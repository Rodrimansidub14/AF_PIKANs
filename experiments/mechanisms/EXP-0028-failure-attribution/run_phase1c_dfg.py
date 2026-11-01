"""EXP-0028 Fase 1c (DFG 2D-1 Re=20, residuo vectorial): ejecutor de los brazos B, F, J, E, I sobre DFGRBFPIKAN (RBF-KAN 2-16-16-3, 11 bases, escalas de salida dentro del forward).

Un JSON por corrida (`dfg-{brazo}-seed{semilla}.json`), reanudable. RECETA FIJADA POR EL INVESTIGADOR (dfg_blockers.md, puntos 1-4):
  Adam    : 2000 PASOS FIJOS (no reloj), lr 1e-3, minibatch de train_dfg (1024 interiores + 256 por frontera, CON reemplazo, torch.randint con Generator del dispositivo sembrado con la semilla),
            float32, pools de Sobol de generate_dfg_pools (interior 16 384, 512 por frontera). El lazo es la copia de `train_dfg` (misma perdida, mismo orden de sorteos).
  Bloque  : recalculo del precondicionador antes del paso 1, cada 1000 pasos de Adam (pasos 0 y 1000) y una vez al arrancar L-BFGS = 3 recalculos por corrida J/E/I.
  L-BFGS  : `sweep.lbfgs_stage` por ITERACIONES (tope 500, strong-Wolfe, hist. 100, chunk 50) sobre el POOL COMPLETO (16 384 interiores + 512 por frontera) de una vez; NO es el worker de 540 s de la campana.
  GN      : 2048 puntos por clave via gram_preconditioning.subset (deterministas, equiespaciados): interior 2048 + 5 x 512 = 4 608 puntos; los 11 terminos de `DFGTerms` (pesos 1 y 1/2).
            El GN NO se calcula sobre el minibatch: es un subconjunto fijo del pool (declarado; B sigue siendo estocastica por paso, igual que los demas brazos: misma secuencia del generador).
Puntuacion (solo despues de entrenar, `prob.score`): contra la referencia de trabajo FEM DFG-L3 (dfg_L3_fields.npz: 9 203 puntos de la malla 221x42 sin el cilindro; presion con ancla en (2.2, 0.2)),
como `dfg_long.score` / score_repair: u, v, p relativos L2 en doble precision sobre una copia del modelo, y los QoI de la campana (`evaluate_dfg_qois`, 512 puntos del cilindro, convencion FEATFLOW):
Cd, Cl, dp y sus errores contra FEATFLOW. Metrica primaria (fijada aqui, antes de ver datos): l2re_flat = L2 relativo aplanado de (u, v, p); el resto es lectura descriptiva (analyze_phase1c_dfg.py).

Modos:
  (normal)   python run_phase1c_dfg.py --arms B,F,J,E,I --seeds 4001,4002 --out DIR [--device cuda:0 --warmup-steps 50]
  --check-id prueba B == ID bit a bit (parametros, historia, puntuacion, L-BFGS) y que el lazo reproduce la perdida y el estado de `train_dfg` PASO A PASO con las mismas semillas en el mismo
             dispositivo (informativo en GPU, exigido en CPU); escribe check_B_ID_dfg.json en --out
  --timing   mide s/paso de Adam, s/iteracion de L-BFGS (pool completo), s/recalculo (J, E, I; chunk 8 y 16) y memoria pico; escribe timing_dfg.json
Solo humo en CPU: --smoke-pool N (submuestrea cada clave a N puntos equiespaciados; marca el JSON), --adam, --lbfgs-iters, --pre-every, --pre-n.
"""
import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
EXP23 = ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning"
sys.path.insert(0, str(BUNDLE or EXP23))
if BUNDLE is None:
    sys.path.insert(1, str(ROOT))

import pre_trainer as pt  # noqa: E402  (EXP-0023; carga t1_trainer / sweep / campana / gram_preconditioning)

gp, sw = pt.gp, pt.sw
rt = gp._residual_terms_module()


def _load_dfg():
    if BUNDLE is not None:                                   # bundle plano: dfg.py copiado de src/sciml/dfg.py
        spec = importlib.util.spec_from_file_location("dfg_flat", str(BUNDLE / "dfg.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dfg_flat"] = mod
        spec.loader.exec_module(mod)
        return mod
    from src.sciml import dfg as mod
    return mod


dfg = _load_dfg()

DFG_REF = (BUNDLE / "dfg_L3_fields.npz") if BUNDLE else ROOT / "var/agent_runs/baseline-competitive-2026-09-27/ns2d-cg-20260930/reference/dfg_L3_fields.npz"
ANCHOR = (2.2, 0.2)
N_PER_KEY = 2048          # puntos del GN por clave del pool (FIJADO)
PRE_CHUNK = 16
LBFGS_ITERS = 500
ADAM_STEPS = 2000         # FIJADO por el investigador
BATCH, BATCH_B = 1024, 256
LR = 1e-3
MAIN_ARMS = ("B", "F", "J", "E", "I")
SEEDS_MAIN = tuple(range(4001, 4011))
PRIMARY = "l2re_flat"
KEYS_BC = ("lower_wall", "upper_wall", "cylinder")


# ------------------------------------------------------------------ perdida (copia exacta de train_dfg) y problema
def dfg_loss(model, interior, inlet, walls, outlet):
    """Perdida de `train_dfg` (8 terminos, peso 1). `walls` = (lower_wall, upper_wall, cylinder). Mismo orden de operaciones que el original."""
    interior = interior.detach().requires_grad_(True)
    residuals = dfg.steady_ns_residual(model(interior), interior)
    losses = {name: value.square().mean() for name, value in residuals.items()}
    inlet_y = inlet[:, 1:2]
    inlet_u = 4.0 * dfg.U_MAX * inlet_y * (dfg.DOMAIN_Y[1] - inlet_y) / (dfg.DOMAIN_Y[1] - dfg.DOMAIN_Y[0]) ** 2
    inlet_target = torch.cat((inlet_u, torch.zeros_like(inlet_u)), dim=1)
    losses["inlet"] = ((model(inlet)[:, :2] - inlet_target) / dfg.U_MEAN).square().mean()
    for name, boundary in zip(KEYS_BC, walls):
        losses[name] = ((model(boundary)[:, :2] - torch.zeros((boundary.shape[0], 2), device=boundary.device, dtype=boundary.dtype)) / dfg.U_MEAN).square().mean()
    outlet = outlet.detach().requires_grad_(True)
    outlet_fields = model(outlet)
    ux, _, vx, _, _, _ = dfg._first_derivatives(outlet_fields, outlet)
    normal_stress_x = (dfg.NU * ux - outlet_fields[:, 2:3]) / dfg.PRESSURE_SCALE
    normal_stress_y = dfg.NU * vx / dfg.PRESSURE_SCALE
    losses["outlet"] = normal_stress_x.square().mean() + normal_stress_y.square().mean()
    total = torch.stack(list(losses.values())).sum()
    return total, losses


_FEM = {}


def _fem():
    if not _FEM:
        z = np.load(DFG_REF)
        _FEM.update({k: z[k] for k in ("xy", "velocity", "pressure_gauge_fixed")})
    return _FEM


class DFGProblem:
    """`prob` de la fase 1c para DFG: loss de POOL COMPLETO (para lbfgs_stage y terms_at), residual_terms (los 11 terminos del GN) y score contra la referencia de trabajo."""
    name = "dfg_2d1_re20"

    def loss(self, model, P):
        return dfg_loss(model, P["interior"], P["inlet"], tuple(P[k] for k in KEYS_BC), P["outlet"])

    def residual_terms(self, model_fn, X):
        return rt.DFGTerms().residual_terms(model_fn, X)

    def score(self, model, device, dtype):
        """Doble precision sobre una COPIA (el modelo de entrenamiento queda en float32). Presion con ancla en (2.2, 0.2) como la referencia (`pressure_gauge_fixed`)."""
        ref = _fem()
        m = copy.deepcopy(model).double()
        dev = torch.device(device)
        xy = torch.as_tensor(ref["xy"], dtype=torch.float64, device=dev)
        with torch.no_grad():
            o = m(xy).cpu().numpy()
            anchor = m(torch.tensor([list(ANCHOR)], dtype=torch.float64, device=dev))[0, 2].item()
        rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        u, v, p = o[:, 0], o[:, 1], o[:, 2] - anchor
        eu, ev, ep = rel(u, ref["velocity"][:, 0]), rel(v, ref["velocity"][:, 1]), rel(p, ref["pressure_gauge_fixed"])
        flat = rel(np.concatenate([u, v, p]), np.concatenate([ref["velocity"][:, 0], ref["velocity"][:, 1], ref["pressure_gauge_fixed"]]))
        q = dfg.evaluate_dfg_qois(m, dev, surface_count=512)
        F = dfg.FEATFLOW_REFERENCE
        return dict(l2re_u=eu, l2re_v=ev, l2re_p=ep, l2re_flat=flat, l2re_mean_components=(eu + ev + ep) / 3.0,
                    Cd=q["drag_coefficient"], Cl=q["lift_coefficient"], dp=q["pressure_difference"],
                    Cd_relerr=abs(q["drag_coefficient"] - F["drag_coefficient"]) / abs(F["drag_coefficient"]),
                    Cl_abserr=abs(q["lift_coefficient"] - F["lift_coefficient"]),
                    dp_relerr=abs(q["pressure_difference"] - F["pressure_difference"]) / abs(F["pressure_difference"]))


def make_problem():
    return DFGProblem()


def make_pools(seed, dev, dt, smoke_pool=0):
    pools = dfg.generate_dfg_pools(seed)                                  # interior 16 384, 512 por frontera (defecto de DFGConfig)
    P = {k: torch.as_tensor(v, dtype=dt, device=dev) for k, v in pools.items()}
    return {k: gp.subset(v, smoke_pool) for k, v in P.items()} if smoke_pool else P


# ------------------------------------------------------------------ trainer propio (lazo de train_dfg + gancho del precondicionador)
def adam_dfg(prob, model, P, steps, seed, dev, pre=None, log_every=100, sync=lambda: None, clock=time.perf_counter, trace=None, batch=BATCH, batch_b=BATCH_B):
    """Lazo de train_dfg por PASOS FIJOS. Con pre=None coincide con train_dfg paso a paso (mismos sorteos, misma perdida, misma actualizacion) en el mismo dispositivo.
    `trace` (lista) recibe (perdida total, terminos) de CADA paso: solo para la prueba contra train_dfg."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def draw(name, requested):
        source = P[name]
        count = min(requested, source.shape[0])
        idx = torch.randint(source.shape[0], (count,), generator=gen, device=dev)
        return source.index_select(0, idx)

    hist, diag, status, step = [], [], "adam_complete", 0
    t0 = clock()
    for step in range(1, steps + 1):
        if pre is not None and pre.due(step - 1):
            d = pre.refresh(model, prob, P, opt, sync=sync, clock=clock)
            d["step"] = step - 1
            diag.append(d)
        opt.zero_grad(set_to_none=True)
        interior = draw("interior", batch)
        inlet = draw("inlet", batch_b)
        walls = tuple(draw(k, batch_b) for k in KEYS_BC)
        outlet = draw("outlet", batch_b)
        total, losses = dfg_loss(model, interior, inlet, walls, outlet)
        total.backward()
        opt.step()
        if trace is not None:
            trace.append((float(total.detach()), {k: float(v.detach()) for k, v in losses.items()}))
        if step % log_every == 0:
            val = float(total.detach())
            if not math.isfinite(val):
                status = "nonfinite_loss"
                break
            hist.append({"step": step, "loss": val, "seconds": clock() - t0})
    sync()
    return dict(hist=hist, pre_diag=diag, seconds=clock() - t0, steps=step, status=status)


def terms_at(prob, model, P, state):
    model.load_state_dict(state)
    total, terms = prob.loss(model, P)
    return float(total.detach()), {k: float(v.detach()) for k, v in terms.items()}


def build(prob, seed, dev, dt):
    torch.manual_seed(seed)                                               # como train_dfg
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return dfg.DFGRBFPIKAN(seed=seed).to(device=dev, dtype=dt)


def run_one(arm, seed, adam_steps, lbfgs_iters, device="cpu", dtype="float32", pre_every=gp.EVERY, pre_n=N_PER_KEY, smoke_pool=0, chunk=PRE_CHUNK):
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = make_problem()
    model = build(prob, seed, dev, dt)
    P = make_pools(seed, dev, dt, smoke_pool)
    pre = None
    if arm != "B":
        pre = gp.GramPreconditioner(arm, every=pre_every, n_points=pre_n, chunk=chunk)
        pre.attach(model)
    t0 = time.perf_counter()
    A = adam_dfg(prob, model, P, adam_steps, seed, dev, pre, sync=sync)
    adam_state = sw.clone_state(model)
    out = dict(problem=prob.name, arm=arm, seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=time.perf_counter() - t0, adam_status=A["status"],
               pre_diag=A["pre_diag"], device=device, dtype=dtype, hist=A["hist"], scientific_accepted=False, pre_every=pre_every, pre_n_per_key=pre_n, pre_chunk=chunk,
               pool_sizes={k: int(v.shape[0]) for k, v in P.items()}, smoke_pool=smoke_pool, primary_metric=PRIMARY, adam_batch=[BATCH, BATCH_B], reference="dfg_L3_fields.npz (referencia de trabajo)")
    if A["status"] != "adam_complete":
        return out
    if pre is not None and pre.due_lbfgs():
        d = pre.refresh(model, prob, P, None, sync=sync)
        d["step"] = "lbfgs_start"
        out["pre_diag"].append(d)
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_iters, chunk=50, snapshot_every=1e12, sync=sync)
    out.update(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], total_seconds=time.perf_counter() - t0,
               pre_recomputes=(pre.n_recomputes if pre else 0), pre_seconds=(pre.seconds if pre else 0.0))
    out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
    out["score_adam"] = sw.score_run(prob, model, dev, dt, adam_state)
    out["score_final"] = sw.score_run(prob, model, dev, dt, L["final_state"])
    out["primary_adam"], out["primary_final"] = out["score_adam"][PRIMARY], out["score_final"][PRIMARY]
    out["loss_adam_end"], out["terms_adam_end"] = terms_at(prob, model, P, adam_state)
    out["loss_final"], out["terms_final"] = terms_at(prob, model, P, L["final_state"])
    return out


def warmup(arm, device, dtype, steps):
    """Corrida desechable (otra semilla, pool reducido para el recalculo) antes de la primera corrida del proceso; no guarda nada (EXP-0016)."""
    run_one(arm, 1, steps, 20, device, dtype, pre_every=max(1, steps // 2), pre_n=64)


# ------------------------------------------------------------------ pruebas y mediciones
def _params_equal(a, b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    return set(pa) == set(pb) and all(torch.equal(pa[k], pb[k]) for k in pa)


def check_vs_train_dfg(seed=4001, steps=10, device="cpu", workdir=None):
    """El lazo propio (B, pools completos) reproduce `train_dfg` PASO A PASO: perdida total y cada termino de cada paso, y el estado final, con las mismas semillas en el mismo dispositivo."""
    dev = dfg.resolve_device(device)
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        dfg.train_dfg(dfg.DFGConfig(seed=seed, max_steps=steps, time_limit_s=600.0, device=str(dev)), td)
        rows = list(csv.DictReader(open(Path(td) / "training_history.csv", encoding="utf-8")))
        ck = torch.load(Path(td) / "dfg_rbf_pikan.pt", map_location="cpu")
    prob = make_problem()
    model = build(prob, seed, dev, torch.float32)
    P = make_pools(seed, dev, torch.float32)
    trace = []
    adam_dfg(prob, model, P, steps, seed, dev, None, log_every=10 ** 9, trace=trace)
    keys = [k for k in rows[0] if k not in ("step", "elapsed_seconds", "total_loss")]
    total_equal = [float(r["total_loss"]) for r in rows] == [t for t, _ in trace]
    terms_equal = all(float(r[k]) == tr[k] for r, (_, tr) in zip(rows, trace) for k in keys)
    state_equal = all(torch.equal(ck["state_dict"][k], v.detach().cpu()) for k, v in model.state_dict().items())
    return dict(steps=steps, seed=seed, device=str(dev), n_rows=len(rows), loss_per_step_equal=bool(total_equal), terms_per_step_equal=bool(terms_equal), final_state_equal=bool(state_equal),
                first_losses=[t for t, _ in trace[:3]], pass_=bool(total_equal and terms_equal and state_equal and len(rows) == steps))


def check_id(out_dir, seed=4001, adam_steps=20, smoke_pool=512, device="cpu", dtype="float32", trainloop_steps=10):
    """B == ID bit a bit (mismo lazo, P = I: parametros finales, historia, puntuacion y L-BFGS) + el lazo reproduce train_dfg paso a paso (exigido en CPU; informativo en GPU)."""
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    runs = {arm: run_one(arm, seed, adam_steps, 5, device, dtype, smoke_pool=smoke_pool) for arm in ("B", "ID")}
    b, i = runs["B"], runs["ID"]
    res = {"B_vs_ID": dict(hist_equal=[h["loss"] for h in b["hist"]] == [h["loss"] for h in i["hist"]], loss_adam_end_equal=b["loss_adam_end"] == i["loss_adam_end"],
                           score_adam_equal=b["score_adam"] == i["score_adam"], score_final_equal=b["score_final"] == i["score_final"],
                           loss_final_equal=b["loss_final"] == i["loss_final"], terms_final_equal=b["terms_final"] == i["terms_final"],
                           lbfgs_iterations=(b["lbfgs_iterations"], i["lbfgs_iterations"]))}
    prob = make_problem()
    P = make_pools(seed, dev, dt, smoke_pool)
    models, traces = {}, {}
    for tag in ("B", "ID"):
        m = build(prob, seed, dev, dt)
        pre = None
        if tag == "ID":
            pre = gp.GramPreconditioner("ID", n_points=N_PER_KEY, chunk=PRE_CHUNK)
            pre.attach(m)
        traces[tag] = []
        adam_dfg(prob, m, P, adam_steps, seed, dev, pre, log_every=10 ** 9, trace=traces[tag])
        models[tag] = m
    res["params_B_equals_ID"] = _params_equal(models["B"], models["ID"])
    res["trace_B_equals_ID"] = traces["B"] == traces["ID"]
    res["B_vs_train_dfg"] = check_vs_train_dfg(seed, trainloop_steps, device)
    res["seed"], res["adam_steps"], res["smoke_pool"], res["device"] = seed, adam_steps, smoke_pool, device
    gate = bool(res["params_B_equals_ID"] and res["trace_B_equals_ID"] and all(v for k, v in res["B_vs_ID"].items() if k != "lbfgs_iterations") and b["lbfgs_iterations"] == i["lbfgs_iterations"])
    if dev.type == "cpu":
        gate = gate and res["B_vs_train_dfg"]["pass_"]
    res["pass"] = gate
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "check_B_ID_dfg.json").write_text(json.dumps(res, indent=1, default=bool))
    return res


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:  # noqa: BLE001
        return None


def timing(out_dir, device="cpu", dtype="float32", adam_n=20, lbfgs_n=20, seed=4001, chunks=(8, 16), arms=("J", "E", "I"), smoke_pool=0):
    """Costo real en este dispositivo: s/paso de Adam (minibatch), s/iteracion de L-BFGS (pool completo), s/recalculo por brazo y chunk con N_PER_KEY por clave, memoria pico (CUDA)."""
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = make_problem()
    P = make_pools(seed, dev, dt, smoke_pool)
    res = dict(device=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"), torch=torch.__version__, dtype=dtype, smoke_pool=smoke_pool, n_per_key=N_PER_KEY,
               pool_sizes={k: int(v.shape[0]) for k, v in P.items()}, rows=[])
    model = build(prob, seed, dev, dt)
    adam_dfg(prob, model, P, 3, seed, dev, None, sync=sync)                               # calentamiento
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    sync()
    t = time.perf_counter()
    adam_dfg(prob, model, P, adam_n, seed, dev, None, sync=sync)
    res["adam_seconds_per_step"] = (time.perf_counter() - t) / adam_n
    res["adam_peak_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    r0 = _rss_mb()
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_n, chunk=min(lbfgs_n, 50), snapshot_every=1e12, sync=sync)
    res["lbfgs_seconds_per_iter"] = L["seconds"] / max(L["iterations"], 1)
    res["lbfgs_iterations_measured"] = L["iterations"]
    res["lbfgs_peak_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else None
    res["rss_mb_before_after_lbfgs"] = [r0, _rss_mb()] if dev.type == "cpu" else None
    for arm in arms:                                              # el primer recalculo de cada proceso paga la compilacion de torch.func: se descarta con uno de calentamiento
        for chunk in chunks:
            m = build(prob, seed, dev, dt)
            pre = gp.GramPreconditioner(arm, n_points=64, chunk=chunk)
            pre.attach(m)
            pre.refresh(m, prob, P, sync=sync)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            m = build(prob, seed, dev, dt)
            pre = gp.GramPreconditioner(arm, n_points=N_PER_KEY, chunk=chunk)
            pre.attach(m)
            d = pre.refresh(m, prob, P, sync=sync)
            res["rows"].append(dict(arm=arm, chunk=chunk, n_points=d["n_points"], seconds=d["seconds"],
                                    peak_mb=(torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else None)))
            print(res["rows"][-1], flush=True)
    t0 = time.perf_counter()                                      # puntuacion (una vez por punto de medida)
    prob.score(model, device, dt)
    res["score_seconds"] = time.perf_counter() - t0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "timing_dfg.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(MAIN_ARMS))
    ap.add_argument("--seeds", default="", help="vacio = 4001-4010")
    ap.add_argument("--adam", type=int, default=ADAM_STEPS, help="pasos fijos de Adam (2000 firmado; otro valor solo para humo)")
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_ITERS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pre-every", type=int, default=gp.EVERY)
    ap.add_argument("--pre-n", type=int, default=N_PER_KEY, help="puntos del GN por clave (firmado 2048)")
    ap.add_argument("--pre-chunk", type=int, default=PRE_CHUNK)
    ap.add_argument("--smoke-pool", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--check-id", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.check_id:
        if a.warmup_steps > 0:
            for arm in ("B", "ID"):
                warmup(arm, a.device, a.dtype, a.warmup_steps)
        steps = a.adam if a.adam != ADAM_STEPS else 20
        print(json.dumps(check_id(out, adam_steps=steps, smoke_pool=a.smoke_pool or 512, device=a.device, dtype=a.dtype)), flush=True)
        return
    if a.timing:
        timing(out, a.device, a.dtype, smoke_pool=a.smoke_pool)
        return
    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS_MAIN)
    if a.warmup_steps > 0:
        for arm in arms:
            warmup(arm, a.device, a.dtype, a.warmup_steps)
    for s in seeds:
        for arm in arms:
            f = out / f"dfg-{arm}-seed{s}.json"
            if f.exists():
                continue
            r = run_one(arm, s, a.adam, a.lbfgs_iters, a.device, a.dtype, a.pre_every, a.pre_n, a.smoke_pool, a.pre_chunk)
            f.write_text(json.dumps(r))
            print("dfg", arm, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s pre {r.get('pre_seconds', 0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
