"""EXP-0023: corrida de la PIKAN-PDE con el precondicionamiento por Gram (brazos B/J/E/I/F; ID solo para pruebas).

Misma receta del control (EXP-0016: Adam lr 1e-3 y L-BFGS strong-Wolfe, pérdida de la PDE) que `EXP-0024.../t1_trainer.py`; con arm=B el camino es el de
sweep.adam_stage + sweep.lbfgs_stage. El costo de recalcular los bloques entra en el reloj (Adam y arranque de L-BFGS). El entrenamiento solo usa prob.loss y
la solución de referencia solo puntúa después (sweep.score_run).
Presupuestos: --adam 0 usa el del control completo (Burgers 20 000, Allen-Cahn 40 000); la convención T1 es el 10 % (--adam 2000 / 4000, --lbfgs-iters 500).
Uso: python pre_trainer.py --problem burgers --arms B,E --seeds 1,2 --adam 2000 --lbfgs-iters 500 [--device cpu --dtype float32 --warmup-steps 0]
"""
import argparse, importlib.util, json, math, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_argv, sys.argv = sys.argv, ["x"]
t1 = _load((BUNDLE or ROOT / "experiments/mechanisms/EXP-0024-wind-tunnel-protocol") / "t1_trainer.py", "t1_trainer_for_0023")
sys.argv = _argv
gp = _load((BUNDLE or ROOT / "src/mechanisms") / "gram_preconditioning.py", "gram_preconditioning_0023")
sw, pc, PROBLEMS = t1.sw, t1.pc, t1.PROBLEMS
ARMS = ("B", "J", "E", "I", "F")


def adam_with_pre(prob, model, P, steps, pre=None, log_every=100, sync=lambda: None, clock=time.perf_counter):
    """Mismo lazo que sweep.adam_stage; el precondicionador se recalcula ANTES del paso done_steps+1 cuando pre.due(done_steps) (0, R, 2R, ...)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist, diag, status, step = [], [], "adam_complete", 0
    t0 = clock()
    for step in range(1, steps + 1):
        if pre is not None and pre.due(step - 1):
            d = pre.refresh(model, prob, P["residual"], opt, sync=sync, clock=clock)
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


def warmup_pre(problem, device, dtype, arms, n_points=64):
    """Recalculo desechable por brazo (J/E/I) para calentar los nucleos de los bloques antes de la primera corrida del proceso. No guarda nada."""
    cfg = PROBLEMS[problem]
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    for arm in arms:
        if arm in gp.RECOMPUTED:
            prob = pc.make_problem(cfg["name"], str(cfg["ref"]))
            m = prob.build("rbf", cfg["width"], 11, 2).to(device=dev, dtype=dt)
            P = prob.pools(2, dev, dt)
            pre = gp.GramPreconditioner(arm, n_points=n_points)
            pre.attach(m)
            pre.refresh(m, prob, P["residual"])


def run_one(problem, arm, seed, adam_steps, lbfgs_iters, device="cpu", dtype="float32", n_col=0, pre_kw=None):
    cfg = PROBLEMS[problem]
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = pc.make_problem(cfg["name"], str(cfg["ref"]))
    if n_col and hasattr(prob, "n_col"):
        prob.n_col = n_col
    model = prob.build("rbf", cfg["width"], 11, seed).to(device=dev, dtype=dt)
    P = prob.pools(seed, dev, dt)
    pre = None
    if arm != "B":
        pre = gp.GramPreconditioner(arm, **(pre_kw or {}))
        pre.attach(model)
    t0 = time.perf_counter()
    A = adam_with_pre(prob, model, P, adam_steps, pre, sync=sync)
    adam_state = sw.clone_state(model)
    t_adam = time.perf_counter() - t0
    out = dict(problem=problem, arm=arm, seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=t_adam, adam_status=A["status"],
               pre_diag=A["pre_diag"], device=device, dtype=dtype, hist=A["hist"], scientific_accepted=False)
    if A["status"] != "adam_complete":
        return out
    t_ref = 0.0
    if pre is not None and pre.due_lbfgs():
        d = pre.refresh(model, prob, P["residual"], None, sync=sync)
        d["step"] = "lbfgs_start"
        out["pre_diag"].append(d)
        t_ref = d["seconds"]
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_iters, chunk=50, snapshot_every=1e12, sync=sync)
    out.update(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], lbfgs_start_refresh_seconds=t_ref,
               total_seconds=time.perf_counter() - t0, pre_recomputes=(pre.n_recomputes if pre else 0), pre_seconds=(pre.seconds if pre else 0.0))
    out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
    out["score_adam"] = sw.score_run(prob, model, dev, dt, adam_state)
    out["score_final"] = sw.score_run(prob, model, dev, dt, L["final_state"])
    out["primary_adam"] = sw.primary(out["score_adam"])
    out["primary_final"] = sw.primary(out["score_final"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(PROBLEMS))
    ap.add_argument("--arms", default="B,E")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--adam", type=int, default=0, help="pasos de Adam (0 = control completo)")
    ap.add_argument("--lbfgs-iters", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-col", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results/runs"))
    ap.add_argument("--pre-every", type=int, default=gp.EVERY, help="R: pasos entre recalculos (pruebas cortas; la convencion firmada es 1000)")
    ap.add_argument("--pre-n", type=int, default=gp.N_POINTS, help="puntos del pool para los bloques (pruebas cortas; firmado 2048)")
    ap.add_argument("--warmup-steps", type=int, default=0, help="calentamiento previo (0 = sin; en GPU usar 50)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    steps = a.adam or PROBLEMS[a.problem]["adam_full"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.warmup_steps > 0:
        print(json.dumps({"warmup": t1.warmup_t1(a.problem, a.device, a.dtype, a.warmup_steps)}), flush=True)
        warmup_pre(a.problem, a.device, a.dtype, a.arms.split(","))
    for arm in a.arms.split(","):
        for s in map(int, a.seeds.split(",")):
            f = out / f"{a.problem}-{arm}-seed{s}-a{steps}-l{a.lbfgs_iters}.json"
            if f.exists():
                continue
            r = run_one(a.problem, arm, s, steps, a.lbfgs_iters, a.device, a.dtype, a.n_col, pre_kw=dict(every=a.pre_every, n_points=a.pre_n))
            f.write_text(json.dumps(r))
            print(a.problem, arm, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s pre {r.get('pre_seconds', 0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
