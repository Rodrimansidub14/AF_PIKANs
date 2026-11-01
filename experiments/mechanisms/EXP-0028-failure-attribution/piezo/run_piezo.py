"""EXP-0028, bloque final de piezo: ejecutor de los brazos B, F, J, E, I sobre el bimorfo PVDF del paper (RBF-PIKAN 4-16-16-8, 11 bases, mismas entradas/salidas que el MLP del paper)
en los DOS problemas, converso y directo. NO LANZADO: piezo no se corre antes de que el investigador firme predictions.md.

RECETA = la del paper: Adam lr 1e-3, 20 000 pasos FIJOS, float64, SIN L-BFGS, remuestreo cada 200 pasos (semilla + k), 2 x 2048 interiores + 256 por frontera/interfaz, pesos
fijos (converso 1/1; directo lambda_BC = 10). Semillas pareadas 20260728-20260732 (cohorte de la linea base congelada; la del paper es 20260728): misma semilla = mismo modelo inicial y mismos puntos para los cinco brazos.
Bloque (mecanismo): el precondicionador se recalcula antes del paso 1 y cada 1000 pasos (20 recalculos por corrida de J/E/I; sin L-BFGS no hay recalculo de arranque de L-BFGS).
GN: gram_preconditioning.subset(pool, 2048) por clave del pool ACTUAL (interior 4096 -> 2048; top/bottom/right/left/interface 256) con los terminos de `piezo_residual_terms`.
Registro del error: cada 500 pasos (u, v, phi relativos L2 contra el FEM del paper, flecha en la punta, potencial del electrodo superior). Metrica primaria: l2re_u en el paso 20 000
(fijada aqui, antes de ver datos; v y phi se reportan igual).

Modos:
  (normal)   python run_piezo.py --problems converse,direct --arms B,F,J,E,I --seeds 20260728,20260729 --out DIR [--device cuda:0 --warmup-steps 50]
  --arms MLP corre el MLP 3x50 SiLU del paper (sin mecanismo): validacion/linea base del paper (semilla 20260728: n=1, 0.107/0.150/0.045 converso y 0.117/0.153/0.067 directo)
  --check-id B == ID bit a bit (parametros y perdida de cada paso, puntuacion) con calentamiento; escribe check_B_ID_piezo.json en --out
  --timing   mide s/paso de Adam, s/recalculo (J, E, I) por problema, puntuacion; escribe timing_piezo.json
Humo en CPU: --adam N, --pre-every, --pre-n, --smoke-pool N (capas de N puntos y fronteras de N/8), --eval-every.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
ROOT = Path(os.environ["AF_ROOT"]) if os.environ.get("AF_ROOT") else (HERE.parents[3] if len(HERE.parents) > 3 else HERE)   # en un contenedor (/kit) no hay 4 niveles
sys.path.insert(0, str(BUNDLE or HERE))
sys.path.insert(1, str(BUNDLE or ROOT / "src/mechanisms"))
if BUNDLE is None:
    sys.path.insert(2, str(ROOT))

import piezo_problem as pp  # noqa: E402
import importlib.util  # noqa: E402


def _load_gp():
    path = (BUNDLE or ROOT / "src/mechanisms") / "gram_preconditioning.py"
    spec = importlib.util.spec_from_file_location("exp0028_gp", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp0028_gp"] = mod
    spec.loader.exec_module(mod)
    return mod


gp = _load_gp()

ADAM_STEPS, LR, RESAMPLE, EVAL_EVERY = 20000, 1e-3, 200, 500
N_PER_KEY, PRE_CHUNK, PRE_EVERY = 2048, 16, gp.EVERY
WIDTH, BASES = 16, 11
MAIN_ARMS = ("B", "F", "J", "E", "I")
SEEDS_MAIN = tuple(range(20260728, 20260733))      # cohorte del paper / linea base congelada (la del paper es 20260728)
REFERENCE_SEED = 20260728
PROBLEMS = ("converse", "direct")
PRIMARY = "l2re_u"


def make_problem(mode, smoke_pool=0):
    kw = dict(interior_per_layer=smoke_pool, boundary=max(8, smoke_pool // 8), interface=max(8, smoke_pool // 8)) if smoke_pool else {}
    return pp.PiezoProblem(mode, **kw)


def build(prob, arm, seed, dt=torch.float64):
    return prob.build("mlp" if arm == "MLP" else "rbf", WIDTH, BASES, seed, dt)


def adam_piezo(prob, model, seed, steps, dev, dt, pre=None, eval_every=EVAL_EVERY, resample_every=RESAMPLE, trace=None, sync=lambda: None, clock=time.perf_counter, log=None, recorder=None, start_step=0):
    """Lazo de run_adam del repo del paper (remuestreo al inicio del paso si epoch > 0 y epoch % 200 == 0, bloque k con semilla + k) + gancho del precondicionador."""
    P = prob.pools(seed, dev, dt, start_step // resample_every)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist, diag, status = [], [], "adam_complete"
    t0 = clock()
    if recorder is not None:
        recorder.start(model, P)
    end = start_step + steps
    for step in range(start_step + 1, end + 1):
        epoch = step - 1
        if epoch > 0 and epoch % resample_every == 0:
            P = prob.pools(seed, dev, dt, epoch // resample_every)
        if pre is not None and pre.due(epoch):
            if recorder is not None:
                with recorder.capture() as cap:
                    d = pre.refresh(model, prob, P, opt, sync=sync, clock=clock)
                if getattr(pre, "last_B", None) is not None:        # brazos S/M: los bloques no pasan por gn_blocks; se entregan al registro
                    cap.blocks, cap.structure, cap.group_traces, cap.blocks_bal = pre.last_B, "edge", pre.last_group_traces, getattr(pre, "last_Bbal", None)
                recorder.on_refresh(epoch, cap, model)
            else:
                d = pre.refresh(model, prob, P, opt, sync=sync, clock=clock)
            d["step"] = epoch
            diag.append(d)
        opt.zero_grad(set_to_none=True)
        loss, terms = prob.loss(model, P)
        loss.backward()
        if recorder is not None:
            recorder.on_step(step, loss, terms, model)
        opt.step()
        val = float(loss.detach())
        if trace is not None:
            trace.append(val)
        if pre is not None and pre.arm in ("S", "M") and math.isfinite(val) and pre.observe(step, val):      # pata P4 (solo brazo S)
            if pre.damp_enabled:
                pre.damp(model, opt)
            else:
                pre.events[-1]["shadow"] = True
        if not math.isfinite(val):
            status = "nonfinite_loss"
            break
        sc = None
        if step % eval_every == 0 or step == end:
            sync()
            sc = prob.score(model, dev, dt)
            hist.append(dict(step=step, loss=val, seconds=clock() - t0, **{k: sc[k] for k in ("l2re_u", "l2re_v", "l2re_phi", "tip_v", "phi_top_mean")}))
            if log:
                log(hist[-1])
        if recorder is not None:
            recorder.after_step(step, model, P, sc)
    sync()
    return dict(hist=hist, pre_diag=diag, seconds=clock() - t0, steps=step, status=status, P_last=P)


def run_one(problem, arm, seed, steps=ADAM_STEPS, device="cpu", dtype="float64", pre_every=PRE_EVERY, pre_n=N_PER_KEY, smoke_pool=0, chunk=None, eval_every=EVAL_EVERY, quiet=False, out_dir=None, artifacts=False, wandb_offline=True, init_npz=None, init_key=None, start_step=0, damp=True):
    dev = torch.device(device)
    if chunk is None:                      # el bloque 'I' (layout por entrada) con float64 y 8 salidas cierra el proceso (segfault de torch en CPU) con chunk >= 4: en CPU se usa 2
        chunk = 2 if (arm == "I" and dev.type == "cpu") else PRE_CHUNK
    dt = torch.float64 if dtype == "float64" else torch.float32
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = make_problem(problem, smoke_pool)
    model = build(prob, arm, seed, dt).to(dev)
    if init_npz is not None:                         # arranque en caliente: estado de la linea base B guardado en `init_key` (p. ej. s004000); Adam parte de cero para todos los brazos
        z = np.load(init_npz, allow_pickle=True)
        sd = {k.split("/state/")[1]: torch.tensor(z[k]) for k in z.keys() if k.startswith(init_key + "/state/")}
        r = model.load_state_dict(sd, strict=False)
        assert not r.missing_keys and not r.unexpected_keys, r
    pre = None
    if arm not in ("B", "MLP"):
        pre = gp.GramPreconditioner(arm, every=pre_every, n_points=pre_n, chunk=chunk)
        pre.damp_enabled = damp
        pre.attach(model)
    rec = None
    if artifacts and out_dir is not None:
        import artifacts as art
        cfg = dict(problem=problem, arm=arm, seed=seed, steps=steps, device=device, dtype=dtype, pre_every=pre_every, pre_n=pre_n, chunk=chunk, weights=prob.w, routing=prob.routing,
                   width=WIDTH, bases=BASES, lr=LR, resample_every=RESAMPLE, smoke_pool=smoke_pool, recipe="paper: Adam 20000 float64 sin L-BFGS",
                   mechanism=dict(arm=arm, eps_rel=gp.EPS_REL, every=pre_every, n_points=pre_n, chunk=chunk, **{k: getattr(gp, k) for k in ("MIX_B", "DENSE_EXTRA", "B_GRID", "B_CAP_W", "COND_P90_MAX", "GUARD_FACTOR", "GUARD_WINDOW", "GUARD_COOLDOWN", "GUARD_MIN_A") if hasattr(gp, k)}),
                   spec=prob.spec if hasattr(prob, "spec") else None, init_npz=init_npz, init_key=init_key, start_step=start_step)
        files = [HERE / "piezo_problem.py", HERE / "run_piezo.py", HERE / "artifacts.py", (BUNDLE or ROOT / "src/mechanisms") / "gram_preconditioning.py",
                 (BUNDLE or ROOT / "src/mechanisms") / "residual_terms.py", (BUNDLE or HERE / "reference") / f"fem_{problem}.npz"]
        rec = art.RunRecorder(prob, out_dir, f"piezo-{problem}-{arm}-seed{seed}", cfg, steps, seed, gp=gp, files=files, wandb_offline=wandb_offline, job_type=problem,
                              state_every=max(1, min(int(os.environ.get("X28_STATE_EVERY", 2000)), steps)), field_every=max(1, min(int(os.environ.get("X28_FIELD_EVERY", 5000)), steps)),
                              hist_every=max(1, min(int(os.environ.get("X28_HIST_EVERY", 100)), steps)), settle_every=max(1, min(1000, steps)))
    t0 = time.perf_counter()
    trace = []
    A = adam_piezo(prob, model, seed, steps, dev, dt, pre, eval_every, sync=sync, log=None if quiet else (lambda h: print(problem, arm, seed, json.dumps(h), flush=True)), recorder=rec, trace=trace, start_step=start_step)
    out = dict(problem=problem, prob_name=prob.name, arm=arm, seed=seed, adam_steps=steps, adam_seconds=A["seconds"], adam_status=A["status"], pre_diag=A["pre_diag"], device=device, dtype=dtype,
               hist=A["hist"], scientific_accepted=False, pre_every=pre_every, pre_n_per_key=pre_n, pre_chunk=chunk, smoke_pool=smoke_pool, primary_metric=PRIMARY, params=sum(q.numel() for q in model.parameters()),
               start_step=start_step, init_key=init_key, loss_trace=trace, pre_events=(pre.events if pre is not None else []),
               weights=prob.w, routing=prob.routing, eval_every=eval_every, resample_every=RESAMPLE, reference="piezo/reference/fem_%s.npz (scikit-fem P2 200x8, 201x21, poling_sign=-1)" % problem)
    if A["status"] == "adam_complete":
        out["score_final"] = prob.score(model, dev, dt)
        out["primary_final"] = out["score_final"][PRIMARY]
        loss, terms = prob.loss(model, A["P_last"])
        out["loss_end"], out["terms_end"] = float(loss.detach()), {k: float(v.detach()) for k, v in terms.items()}
        out["interface_gap"] = prob.interface_gap(model, A["P_last"])
        try:
            out["score_flux"] = prob.score_flux(model, dev)
        except FileNotFoundError:
            out["score_flux"] = None
        out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
        out["pre_recomputes"], out["pre_seconds"] = (pre.n_recomputes, pre.seconds) if pre else (0, 0.0)
        out["total_seconds"] = time.perf_counter() - t0
    out["run_card"] = dict(
        purpose="Medir si el precondicionamiento por arista (E) y el brazo graduado M reducen el error de la PIKAN en el bimorfo PVDF del paper (converso y directo); hipótesis y firma en EXP-0029_spec.md y DISENO_FINAL.md.",
        data="Sintética: puntos de colocación muestreados por semilla; referencia FEM (scikit-fem P2) generada localmente; sin datos personales ni sensibles.",
        evaluation="l2re_u, l2re_v, l2re_phi contra el FEM cada eval_every pasos y al final; flujos y consistencia constitutiva en artefactos; pareo por semilla con B y E.",
        limitations=["n pequeño de semillas (3 a 5): sin p-valores", "referencia FEM propia, no aceptada como referencia de terceros", "hardware mixto entre bloques (T4, H100, A100)", "la guarda P4 de M solo observa; P1 no tiene señal en línea", "float64 con semillas pareadas, no determinismo bit a bit entre GPUs"],
        human_oversight="Hipótesis, predicciones y lanzamiento aprobados por el investigador (ver provenance/EXP-0029_spec.md).",
        compute=dict(adam_seconds=A["seconds"], concurrent_processes=int(os.environ.get("X28_K", 1)), note="Energía estimada = segundos de GPU / procesos concurrentes x potencia de la GPU; sin medir."))
    if rec is not None:
        out["artifacts"] = rec.finalize(model, A["P_last"], extra=dict(adam_status=A["status"]))
    return out


def warmup(problem, arm, device, dtype, steps):
    """Corrida desechable (otra semilla, pool reducido) antes de la primera del proceso; no guarda nada."""
    run_one(problem, arm, 1, steps, device, dtype, pre_every=max(1, steps // 2), pre_n=64, smoke_pool=32, eval_every=10 ** 9, quiet=True)


def _params_equal(a, b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    return set(pa) == set(pb) and all(torch.equal(pa[k], pb[k]) for k in pa)


def check_id(out_dir, problems=PROBLEMS, seed=4001, steps=12, smoke_pool=32, device="cpu", dtype="float64", warmup_steps=0):
    """B == ID bit a bit en cada problema: parametros finales, perdida de CADA paso y puntuacion; `warmup_steps` > 0 calienta el proceso antes (GPU)."""
    dev = torch.device(device)
    dt = torch.float64 if dtype == "float64" else torch.float32
    res, ok = {}, True
    for problem in problems:
        if warmup_steps > 0:
            for arm in ("B", "ID"):
                warmup(problem, arm, device, dtype, warmup_steps)
        prob = make_problem(problem, smoke_pool)
        models, traces, scores = {}, {}, {}
        for tag in ("B", "ID"):
            m = build(prob, tag if tag == "ID" else "B", seed, dt).to(dev)
            pre = None
            if tag == "ID":
                pre = gp.GramPreconditioner("ID", n_points=N_PER_KEY, chunk=PRE_CHUNK)
                pre.attach(m)
            traces[tag] = []
            adam_piezo(prob, m, seed, steps, dev, dt, pre, eval_every=10 ** 9, trace=traces[tag], resample_every=5)
            models[tag], scores[tag] = m, prob.score(m, device, dt)
        r = dict(params_equal=_params_equal(models["B"], models["ID"]), trace_equal=traces["B"] == traces["ID"], score_equal=scores["B"] == scores["ID"], first_losses=traces["B"][:3])
        r["pass"] = bool(r["params_equal"] and r["trace_equal"] and r["score_equal"])
        ok = ok and r["pass"]
        res[problem] = r
    res.update(seed=seed, steps=steps, smoke_pool=smoke_pool, device=device, warmup_steps=warmup_steps, passed=bool(ok))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "check_B_ID_piezo.json").write_text(json.dumps(res, indent=1, default=bool))
    # el JSON que lee el lanzador: clave `pass` en la raiz
    res["pass"] = res["passed"]
    (Path(out_dir) / "check_B_ID_piezo.json").write_text(json.dumps(res, indent=1, default=bool))
    return res


def timing(out_dir, device="cpu", dtype="float64", adam_n=20, seed=4001, chunks=(8, 16), arms=("J", "E", "I"), smoke_pool=0, problems=PROBLEMS):
    """Costo real en este dispositivo y por problema: s/paso de Adam (B y MLP), s/recalculo por brazo y chunk con N_PER_KEY por clave, memoria pico (CUDA), puntuacion."""
    dev = torch.device(device)
    dt = torch.float64 if dtype == "float64" else torch.float32
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    res = dict(device=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"), torch=torch.__version__, dtype=dtype, smoke_pool=smoke_pool, n_per_key=N_PER_KEY, problems={})
    for problem in problems:
        prob = make_problem(problem, smoke_pool)
        row = dict(rows=[])
        for tag in ("B", "MLP"):
            model = build(prob, tag, seed, dt).to(dev)
            adam_piezo(prob, model, seed, 3, dev, dt, None, eval_every=10 ** 9, sync=sync)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            sync()
            t = time.perf_counter()
            adam_piezo(prob, model, seed, adam_n, dev, dt, None, eval_every=10 ** 9, sync=sync)
            row[f"adam_seconds_per_step_{tag}"] = (time.perf_counter() - t) / adam_n
            row[f"adam_peak_mb_{tag}"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else None
        P = prob.pools(seed, dev, dt, 0)
        for arm in arms:
            for chunk in (tuple(chunks) + ((2,) if arm == "I" and 2 not in chunks else ())):
                m = build(prob, "B", seed, dt).to(dev)
                pre = gp.GramPreconditioner(arm, n_points=64, chunk=chunk)
                pre.attach(m)
                pre.refresh(m, prob, P, sync=sync)                               # calentamiento (compilacion de torch.func)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                m = build(prob, "B", seed, dt).to(dev)
                pre = gp.GramPreconditioner(arm, n_points=N_PER_KEY, chunk=chunk)
                pre.attach(m)
                d = pre.refresh(m, prob, P, sync=sync)
                row["rows"].append(dict(arm=arm, chunk=chunk, n_points=d["n_points"], seconds=d["seconds"], peak_mb=(torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else None)))
                print(problem, row["rows"][-1], flush=True)
                res["problems"][problem] = row
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / "timing_piezo.json").write_text(json.dumps(res, indent=1))
        t0 = time.perf_counter()
        prob.score(model, device, dt)
        row["score_seconds"] = time.perf_counter() - t0
        res["problems"][problem] = row
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "timing_piezo.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=",".join(PROBLEMS))
    ap.add_argument("--arms", default=",".join(MAIN_ARMS))
    ap.add_argument("--seeds", default="", help="vacio = 20260728-20260732")
    ap.add_argument("--adam", type=int, default=ADAM_STEPS, help="pasos fijos de Adam (20 000 = receta del paper; otro valor solo para humo)")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pre-every", type=int, default=PRE_EVERY)
    ap.add_argument("--pre-n", type=int, default=N_PER_KEY)
    ap.add_argument("--pre-chunk", type=int, default=None, help="chunk del GN (defecto 16; 2 para I en CPU)")
    ap.add_argument("--smoke-pool", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--timing-chunks", default="8,16", help="chunks del GN a medir en --timing")
    ap.add_argument("--timing-arms", default="J,E,I")
    ap.add_argument("--no-artifacts", action="store_true", help="no guardar el .npz de artefactos ni el registro W&B offline")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--check-id", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--init-dir", default=None, help="carpeta con piezo-<problema>-B-seed<semilla>-artifacts.npz (arranque en caliente)")
    ap.add_argument("--init-key", default="s004000")
    ap.add_argument("--no-damp", action="store_true", help="brazo S: la guarda P4 solo observa (eventos en sombra), no amortigua")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = a.problems.split(",")
    if a.check_id:
        steps = a.adam if a.adam != ADAM_STEPS else 12
        print(json.dumps(check_id(out, problems, steps=steps, smoke_pool=a.smoke_pool or 32, device=a.device, dtype=a.dtype, warmup_steps=a.warmup_steps)), flush=True)
        return
    if a.timing:
        timing(out, a.device, a.dtype, smoke_pool=a.smoke_pool, problems=problems, chunks=tuple(int(c) for c in a.timing_chunks.split(",")), arms=tuple(a.timing_arms.split(",")))
        return
    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS_MAIN)
    for problem in problems:
        if a.warmup_steps > 0:
            for arm in arms:
                warmup(problem, arm, a.device, a.dtype, a.warmup_steps)
        for s in seeds:
            for arm in arms:
                tag = (f"-ws{a.start_step}" if a.init_dir else "") + ("-nodamp" if a.no_damp else "")
                f = out / f"piezo-{problem}-{arm}-seed{s}{tag}.json"
                if f.exists():
                    continue
                init = str(Path(a.init_dir) / f"piezo-{problem}-B-seed{s}-artifacts.npz") if a.init_dir else None
                r = run_one(problem, arm, s, a.adam, a.device, a.dtype, a.pre_every, a.pre_n, a.smoke_pool, a.pre_chunk, a.eval_every, False, str(out), (not a.no_artifacts) and not init, not a.no_wandb, init, a.init_key if init else None, a.start_step if init else 0, not a.no_damp)
                f.write_text(json.dumps(r))
                print("piezo", problem, arm, s, f"final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s pre {r.get('pre_seconds', 0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
