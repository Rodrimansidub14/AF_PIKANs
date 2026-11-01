"""EXP-0028 Fase 1d (Enmienda v1.2): confirmacion con RECETA COMPLETA de la linea base, solo PIKAN (RBF-KAN). Allen-Cahn d=1e-3, Burgers RBF48, Fisher-KPP.
(NS2d-CG usa run_phase1c.py --recipe full.)  Reutiliza SIN modificar run_phase1.py (importa make_problem, loss_at, warmup) y el camino de entrenamiento de EXP-0023
(pt.adam_with_pre, t1.adam_with_hook, sw.lbfgs_stage). Un JSON por corrida (`{problema}-{brazo}-seed{semilla}.json`), reanudable.

Recetas (RECIPES; cada una copia la linea base congelada v1/v2 del problema, no la receta corta T1 de la fase 1):
  ac      : Allen-Cahn d=1e-3, RBF16, 8000+400+800 puntos Sobol, Adam lr 1e-3 x 40 000, float32, L-BFGS strong-Wolfe float32, tope 300 s / 15 000 it., parada por meseta
            (EXP-0008 ac-rerun-20260930; baseline v1/v2: fin de Adam 2.84e-2, tras L-BFGS 2.686e-3).
  burgers : Burgers RBF48, 16 384 puntos Sobol, Adam x 24 000 float32, L-BFGS strong-Wolfe en float64 (modelo y pool convertidos), tope 1 800 s / 15 000 it.
            (EXP-0008 burgers-big-20261003, la fila de baseline v2: 7.165e-5).
  kpp     : Fisher-KPP, RBF16. NO tiene linea base v1/v2: hereda la receta de Allen-Cahn (misma red, mismo arnes, mismos pools; es lo que la ronda 1-C declara como «receta heredada»).
Cadencia del precondicionador como en la fase 1 (cada 1000 pasos de Adam y al arrancar L-BFGS, N = 2048): 41 recalculos (AC, KPP) o 25 (Burgers).
En Burgers el recalculo de L-BFGS se hace DESPUES de pasar a float64 (el GN siempre se calcula en float64); el error al final de Adam se puntua en float32 antes del cambio.
Brazos: B, F, J, E, I, ID, C (C = regrid idea A de EXP-0024, cuantiles puros una vez al 25 % de Adam, escalado al nuevo numero de pasos).
Semillas por defecto 4101-4105 (no usadas en ninguna fase previa; se rechazan las 4001-4010 salvo humo).
Solo se puntua contra la referencia DESPUES de entrenar.

Uso : python run_phase1d.py --problems ac --arms B,E,F,C --seeds 4101 --out DIR [--device cuda:0 --warmup-steps 50]
Humo: --adam N --lbfgs-iters M --lbfgs-seconds S --n-col K --pre-every E --pre-n n   (marca el JSON smoke=true y levanta la guardia de semillas)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase1 as r1  # noqa: E402  (arnes de la fase 1, sin cambios)

pt, gp, sw, t1 = r1.pt, r1.gp, r1.sw, r1.t1

_AC = dict(adam=40000, dtype="float32", lbfgs_dtype="float32", lbfgs_seconds=300.0, lbfgs_max_iter=15000, n_col=None)
RECIPES = {
    "ac": dict(_AC, source="EXP-0008 ac-rerun-20260930 (baseline v1/v2 Allen-Cahn, semillas 601-605)"),
    "kpp": dict(_AC, source="heredada de Allen-Cahn: Fisher-KPP no tiene linea base v1/v2 (ronda 1-C: receta heredada, no firmada)"),
    "burgers": dict(adam=24000, dtype="float32", lbfgs_dtype="float64", lbfgs_seconds=1800.0, lbfgs_max_iter=15000, n_col=16384,
                    source="EXP-0008 burgers-big-20261003 (baseline v2 Burgers RBF48, semillas 301-305)"),
}
ARMS = ("B", "ID", "F", "J", "E", "I", "C")
SEEDS_DEFAULT = tuple(range(4101, 4106))
USED_SEEDS = set(range(4001, 4011))


def run_one_1d(problem, arm, seed, rec, out_dir=None, device="cpu", pre_every=gp.EVERY, pre_n=gp.N_POINTS, smoke=False):
    if arm not in ARMS:
        raise ValueError(f"brazo {arm!r} no soportado en 1d (permitidos: {ARMS})")
    dev = torch.device(device)
    dt = torch.float32 if rec["dtype"] == "float32" else torch.float64
    ldt = torch.float32 if rec["lbfgs_dtype"] == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob, width = r1.make_problem(problem)
    if rec.get("n_col") and hasattr(prob, "n_col"):
        prob.n_col = rec["n_col"]
    model = prob.build("rbf", width, 11, seed).to(device=dev, dtype=dt)
    P = prob.pools(seed, dev, dt)
    pre, extra = None, {}
    t0 = time.perf_counter()
    if arm == "C":
        spec = t1.DESIGNS["D1"]
        A = t1.adam_with_hook(prob, model, P, rec["adam"], regrid=dict(steps=spec["steps"], mix=spec["mix"]), sync=sync)
        extra["regrid_diag"] = A["regrid_diag"]
    else:
        if arm != "B":
            pre = gp.GramPreconditioner(arm, every=pre_every, n_points=pre_n)
            pre.attach(model)
        A = pt.adam_with_pre(prob, model, P, rec["adam"], pre, sync=sync)
    adam_state = sw.clone_state(model)
    adam_seconds = time.perf_counter() - t0
    out = dict(problem=problem, arm=arm, seed=seed, recipe={k: v for k, v in rec.items()}, adam_steps=rec["adam"], adam_seconds=adam_seconds, adam_status=A["status"],
               adam_s_per_step=adam_seconds / max(1, A["steps"]), pre_diag=A.get("pre_diag", []), device=device, dtype=rec["dtype"], lbfgs_dtype=rec["lbfgs_dtype"],
               n_pool=int(P["residual"].shape[0]), hist=A["hist"], smoke=bool(smoke), scientific_accepted=False, phase="1d", **extra)
    if A["status"] != "adam_complete":
        return out
    out["score_adam"] = sw.score_run(prob, model, dev, dt, adam_state)             # en el dtype de Adam, antes de cambiar a float64 (como la linea base)
    out["primary_adam"] = sw.primary(out["score_adam"])
    out["loss_adam_end"] = r1.loss_at(prob, model, P, adam_state)
    if ldt != dt:
        model.to(dtype=ldt)
        P = {k: v.to(ldt) for k, v in P.items()}
    if pre is not None and pre.due_lbfgs():
        d = pre.refresh(model, prob, P["residual"], None, sync=sync)
        d["step"] = "lbfgs_start"
        out["pre_diag"].append(d)
    L = sw.lbfgs_stage(prob, model, P, rec["lbfgs_seconds"], max_iter=rec["lbfgs_max_iter"], chunk=50, snapshot_every=1e12, sync=sync)
    out.update(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], total_seconds=time.perf_counter() - t0,
               pre_recomputes=(pre.n_recomputes if pre else 0), pre_seconds=(pre.seconds if pre else 0.0), lbfgs_traj=L["traj"][:: max(1, len(L["traj"]) // 200)])
    out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
    out["score_final"] = sw.score_run(prob, model, dev, ldt, L["final_state"])
    out["primary_final"] = sw.primary(out["score_final"])
    out["loss_final"] = r1.loss_at(prob, model, P, L["final_state"])
    return out


def warmup(problem, rec, device, arms, steps):
    """Corrida desechable por proceso (EXP-0016): mismos nucleos (incluido el tramo float64 de Burgers), otra semilla, nada se guarda."""
    wrec = dict(rec, adam=steps, lbfgs_max_iter=20, lbfgs_seconds=30.0, n_col=min(rec["n_col"], 1024) if rec.get("n_col") else None)
    for arm in ("B",) + tuple(a for a in arms if a not in ("B", "F")):
        run_one_1d(problem, arm, 1, wrec, None, device, pre_every=max(1, steps // 2), pre_n=64, smoke=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="ac,burgers,kpp")
    ap.add_argument("--arms", default="B,F,E")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pre-every", type=int, default=gp.EVERY)
    ap.add_argument("--pre-n", type=int, default=gp.N_POINTS)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--adam", type=int, default=0, help="solo humo: sustituye los pasos de Adam de la receta")
    ap.add_argument("--lbfgs-iters", type=int, default=0, help="solo humo: sustituye el tope de iteraciones de L-BFGS")
    ap.add_argument("--lbfgs-seconds", type=float, default=0.0, help="solo humo: sustituye el tope de segundos de L-BFGS")
    ap.add_argument("--n-col", type=int, default=0, help="solo humo: sustituye los puntos de colocacion (Burgers)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    smoke = bool(a.adam or a.lbfgs_iters or a.lbfgs_seconds or a.n_col)
    seeds = [int(s) for s in a.seeds.split(",")]
    if not smoke and (set(seeds) & USED_SEEDS or any(s < 4101 for s in seeds)):
        sys.exit(f"semillas {sorted(set(seeds) & USED_SEEDS) or seeds} ya usadas en la fase 1 o fuera de 4101+: use otras o --adam (humo)")
    arms = a.arms.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for problem in a.problems.split(","):
        rec = dict(RECIPES[problem])
        if a.adam:
            rec["adam"] = a.adam
        if a.lbfgs_iters:
            rec["lbfgs_max_iter"] = a.lbfgs_iters
        if a.lbfgs_seconds:
            rec["lbfgs_seconds"] = a.lbfgs_seconds
        if a.n_col:
            rec["n_col"] = a.n_col
        if a.warmup_steps > 0:
            warmup(problem, rec, a.device, arms, a.warmup_steps)
        for s in seeds:
            for arm in arms:
                f = out / f"{problem}-{arm}-seed{s}.json"
                if f.exists():
                    continue
                r = run_one_1d(problem, arm, s, rec, out, a.device, a.pre_every, a.pre_n, smoke)
                f.write_text(json.dumps(r))
                print(problem, arm, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s {r.get('lbfgs_status')}", flush=True)


if __name__ == "__main__":
    main()
