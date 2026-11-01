"""EXP-0028 Fase 1c (NS2d-CG, residuo vectorial): ejecutor de los brazos B, F, J, E, I sobre PINNacle NS2d-CG, solo PIKAN (RBF-KAN 2-16-16-3, 11 bases).

Un JSON por corrida (`ns-{brazo}-seed{semilla}.json`), reanudable. Receta = la del worker de NS2d-CG de pikan_campaign.py (pools del arnes, perdida de 8 terminos con peso 1, Adam lr 1e-3,
lote completo, float32, RBF16) mas, como en 1a, un tramo L-BFGS por ITERACIONES con `sweep.lbfgs_stage` (tope 500, strong-Wolfe, chunk 50). Cadencia de recalculo del precondicionador: antes del
paso 1 y cada 1000 pasos de Adam, y una vez al arrancar L-BFGS (igual que EXP-0023/pre_trainer.py).

Presupuestos (--recipe):
  full (defecto) : Adam 20 000 (el del worker NS2d-CG); L-BFGS 500 it.         -> ~21 recalculos de J/E/I por corrida
  t1             : Adam  2 000 (10 %, la convencion T1 de 1a para el resto del banco) -> 3 recalculos. NO es el defecto: se elige en la firma de 1c (ver budget_1c.md).
Puntos del Gauss-Newton (FIJADOS): N_PER_KEY = 2048 por clave del pool via gram_preconditioning.subset (indices equiespaciados, deterministas). Las claves del pool NS son
domain (8192 -> 2048), inlet, outlet, other (cada una < 2048: completas), asi que el GN usa 4 095 puntos en total y los 8 terminos de la perdida (3 PDE + 5 de borde), con los pesos de la
perdida (todos 1). Es el mismo N = 2048 de 1-B/1-C y del firmado en EXP-0023; el costo medido (budget_vector.json: E 128/clave 25 s, 2048/clave 171 s en CPU) crece sublineal en N, y a
2048/clave el recalculo es ~6-15 % de una corrida en T4 (budget_1c.md), asi que no se recorta. Los chunks del Jacobiano son 16 (MAX_CHUNK de residual_terms; el resultado no depende del chunk).

Metrica primaria de NS: `l2re_flat` (error L2 relativo aplanado de todas las salidas, la cifra de cabecera de la campana NS: ns2dcg-results.md). NOTA: `sw.primary` devolveria l2re_u (primer valor
del dict); aqui se guarda el dict completo (l2re_u, v, p, media de componentes, aplanado) y `primary_*` = l2re_flat. El analisis reporta tambien la media de componentes (v y p dominan el error de campo).
La referencia (ns_0_obstacle.dat, COMSOL) solo puntua DESPUES de entrenar (prob.score).

Modos:
  (normal)   python run_phase1c.py --arms B,F,J,E,I --seeds 4001,4002 --out DIR [--device cuda:0 --warmup-steps 50]
  --check-id prueba B == ID bit a bit (parametros, historia de perdida, puntuacion) y B == sweep.adam_stage; escribe check_B_ID_ns.json en --out
  --timing   mide s/paso de Adam, s/recalculo (J, E, I; chunk 8 y 16), s por iteracion de L-BFGS y memoria pico; escribe timing_ns.json (CPU o GPU)
Solo humo en CPU: --smoke-pool N (submuestrea cada clave del pool a N puntos equiespaciados; marca el JSON), --adam, --lbfgs-iters, --pre-every, --pre-n.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
EXP23 = ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning"
sys.path.insert(0, str(BUNDLE or EXP23))

import pre_trainer as pt  # noqa: E402  (EXP-0023; carga t1_trainer / sweep / campana / gram_preconditioning)

pc, gp, sw = pt.pc, pt.gp, pt.sw
rt = gp._residual_terms_module()

NS_REF = (BUNDLE / "ns_0_obstacle.dat") if BUNDLE else ROOT / "var/agent_runs/baseline-competitive-2026-09-27/ns2d-cg-20260930/reference/ns_0_obstacle.dat"
WIDTH, BASES = 16, 11
N_PER_KEY = 2048          # puntos del GN por clave del pool (FIJADO)
PRE_CHUNK = 16
LBFGS_ITERS = 500
RECIPES = {"full": 20000, "t1": 2000}
MAIN_ARMS = ("B", "F", "J", "E", "I")
SEEDS_MAIN = tuple(range(4001, 4011))
PRIMARY = "l2re_flat"


def make_problem():
    return rt.install(pc.make_problem("ns", str(NS_REF)))


def smoke_pools(P, n):
    return {k: gp.subset(v, n) for k, v in P.items()} if n else P


def adam_with_pre_vec(prob, model, P, steps, pre=None, log_every=100, sync=lambda: None, clock=time.perf_counter):
    """Mismo lazo que pt.adam_with_pre y sweep.adam_stage (los tres coinciden bit a bit con pre=None), pero el pool que ve el precondicionador es el diccionario P completo
    (NS no tiene P['residual']): gram_preconditioning.subset toma N_PER_KEY puntos de cada clave."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist, diag, status, step = [], [], "adam_complete", 0
    t0 = clock()
    for step in range(1, steps + 1):
        if pre is not None and pre.due(step - 1):
            d = pre.refresh(model, prob, P, opt, sync=sync, clock=clock)
            d["step"] = step - 1
            diag.append(d)
        opt.zero_grad(set_to_none=True)
        total, _ = prob.loss(model, P)
        total.backward()
        opt.step()
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


def run_one(arm, seed, adam_steps, lbfgs_iters, device="cpu", dtype="float32", pre_every=gp.EVERY, pre_n=N_PER_KEY, smoke_pool=0, chunk=PRE_CHUNK):
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = make_problem()
    model = prob.build("rbf", WIDTH, BASES, seed).to(device=dev, dtype=dt)
    P = smoke_pools(prob.pools(seed, dev, dt), smoke_pool)
    pre = None
    if arm != "B":
        pre = gp.GramPreconditioner(arm, every=pre_every, n_points=pre_n, chunk=chunk)
        pre.attach(model)
    t0 = time.perf_counter()
    A = adam_with_pre_vec(prob, model, P, adam_steps, pre, sync=sync)
    adam_state = sw.clone_state(model)
    out = dict(problem="ns2d_cg", arm=arm, seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=time.perf_counter() - t0, adam_status=A["status"],
               pre_diag=A["pre_diag"], device=device, dtype=dtype, hist=A["hist"], scientific_accepted=False, pre_every=pre_every, pre_n_per_key=pre_n, pre_chunk=chunk,
               pool_sizes={k: int(v.shape[0]) for k, v in P.items()}, smoke_pool=smoke_pool, primary_metric=PRIMARY)
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
    """Corrida desechable (otra semilla, pool reducido para el recalculo) antes de la primera corrida del proceso; no guarda nada (EXP-0016: sin calentamiento la primera corrida en GPU no reproduce)."""
    run_one(arm, 1, steps, 20, device, dtype, pre_every=max(1, steps // 2), pre_n=64)


# ------------------------------------------------------------------ pruebas y mediciones
def _params_equal(a, b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    return set(pa) == set(pb) and all(torch.equal(pa[k], pb[k]) for k in pa)


def check_id(out_dir, seed=4001, adam_steps=20, smoke_pool=512, device="cpu", dtype="float32"):
    """B == ID bit a bit (mismo lazo, P = I: parametros finales, historia de perdida, puntuacion y L-BFGS) y B == sweep.adam_stage (el lazo del control)."""
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    res = {}
    runs = {}
    for arm in ("B", "ID"):
        runs[arm] = run_one(arm, seed, adam_steps, 5, device, dtype, smoke_pool=smoke_pool)
    b, i = runs["B"], runs["ID"]
    res["B_vs_ID"] = dict(hist_equal=[h["loss"] for h in b["hist"]] == [h["loss"] for h in i["hist"]], loss_adam_end_equal=b["loss_adam_end"] == i["loss_adam_end"],
                          score_adam_equal=b["score_adam"] == i["score_adam"], score_final_equal=b["score_final"] == i["score_final"],
                          loss_final_equal=b["loss_final"] == i["loss_final"], terms_final_equal=b["terms_final"] == i["terms_final"],
                          lbfgs_iterations=(b["lbfgs_iterations"], i["lbfgs_iterations"]))
    # parametros finales: se repite el entrenamiento Adam (la funcion no devuelve el modelo) y se compara con el lazo del control
    prob = make_problem()
    P = smoke_pools(prob.pools(seed, dev, dt), smoke_pool)
    models = {}
    for tag in ("B", "ID", "control"):
        m = prob.build("rbf", WIDTH, BASES, seed).to(device=dev, dtype=dt)
        pre = None
        if tag == "ID":
            pre = gp.GramPreconditioner("ID", n_points=N_PER_KEY, chunk=PRE_CHUNK)
            pre.attach(m)
        if tag == "control":
            sw.adam_stage(prob, m, P, adam_steps)
        else:
            adam_with_pre_vec(prob, m, P, adam_steps, pre)
        models[tag] = m
    res["params_B_equals_ID"] = _params_equal(models["B"], models["ID"])
    res["params_B_equals_sweep_adam_stage"] = _params_equal(models["B"], models["control"])
    res["seed"], res["adam_steps"], res["smoke_pool"], res["device"] = seed, adam_steps, smoke_pool, device
    res["pass"] = bool(res["params_B_equals_ID"] and res["params_B_equals_sweep_adam_stage"] and all(v for k, v in res["B_vs_ID"].items() if k != "lbfgs_iterations")
                       and b["lbfgs_iterations"] == i["lbfgs_iterations"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "check_B_ID_ns.json").write_text(json.dumps(res, indent=1, default=bool))
    return res


def timing(out_dir, device="cpu", dtype="float32", adam_n=20, lbfgs_n=20, seed=4001, chunks=(8, 16), arms=("J", "E", "I"), smoke_pool=0):
    """Costo real en este dispositivo: s/paso de Adam (lote completo), s/recalculo por brazo y chunk con N_PER_KEY por clave, s/iteracion de L-BFGS, memoria pico (CUDA)."""
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = make_problem()
    P = smoke_pools(prob.pools(seed, dev, dt), smoke_pool)
    res = dict(device=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"), torch=torch.__version__, dtype=dtype, smoke_pool=smoke_pool, n_per_key=N_PER_KEY,
               pool_sizes={k: int(v.shape[0]) for k, v in P.items()}, rows=[])
    model = prob.build("rbf", WIDTH, BASES, seed).to(device=dev, dtype=dt)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for k in range(adam_n + 3):
        if k == 3:
            sync()
            t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        prob.loss(model, P)[0].backward()
        opt.step()
    sync()
    res["adam_seconds_per_step"] = (time.perf_counter() - t) / adam_n
    # L-BFGS: sweep.lbfgs_stage por iteraciones (el primer chunk incluye el calentamiento de la linea de busqueda)
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_n, chunk=min(lbfgs_n, 50), snapshot_every=1e12, sync=sync)
    res["lbfgs_seconds_per_iter"] = L["seconds"] / max(L["iterations"], 1)
    res["lbfgs_iterations_measured"] = L["iterations"]
    for arm in arms:                                              # el primer recalculo de cada proceso paga la compilacion de torch.func: se descarta con un recalculo de calentamiento
        for chunk in chunks:
            m = prob.build("rbf", WIDTH, BASES, seed).to(device=dev, dtype=dt)
            pre = gp.GramPreconditioner(arm, n_points=64, chunk=chunk)
            pre.attach(m)
            pre.refresh(m, prob, P, sync=sync)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            m = prob.build("rbf", WIDTH, BASES, seed).to(device=dev, dtype=dt)
            pre = gp.GramPreconditioner(arm, n_points=N_PER_KEY, chunk=chunk)
            pre.attach(m)
            d = pre.refresh(m, prob, P, sync=sync)
            res["rows"].append(dict(arm=arm, chunk=chunk, n_points=d["n_points"], seconds=d["seconds"],
                                    peak_mb=(torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else None)))
            print(res["rows"][-1], flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "timing_ns.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(MAIN_ARMS))
    ap.add_argument("--seeds", default="", help="vacio = 4001-4010")
    ap.add_argument("--recipe", default="full", choices=tuple(RECIPES))
    ap.add_argument("--adam", type=int, default=0, help="0 = segun --recipe; otro valor solo para humo")
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
        print(json.dumps(check_id(out, adam_steps=a.adam or 20, smoke_pool=a.smoke_pool or 512, device=a.device, dtype=a.dtype)), flush=True)
        return
    if a.timing:
        timing(out, a.device, a.dtype, smoke_pool=a.smoke_pool)
        return
    arms = a.arms.split(",")
    steps = a.adam or RECIPES[a.recipe]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS_MAIN)
    if a.warmup_steps > 0:
        for arm in arms:
            warmup(arm, a.device, a.dtype, a.warmup_steps)
    for s in seeds:
        for arm in arms:
            f = out / f"ns-{arm}-seed{s}.json"
            if f.exists():
                continue
            r = run_one(arm, s, steps, a.lbfgs_iters, a.device, a.dtype, a.pre_every, a.pre_n, a.smoke_pool, a.pre_chunk)
            f.write_text(json.dumps(r))
            print("ns", arm, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s pre {r.get('pre_seconds', 0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
