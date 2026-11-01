"""EXP-0026 (v1.1): diagnostico de E en Burgers y Allen-Cahn. Factorial cadencia {500, 1000} x recalculo al arrancar L-BFGS {si, no} para la correccion de Gram E, mas B.
Arnes de 1-B sin cambios (pre_trainer / t1_trainer / sweep): Burgers RBF48, Adam 2000, L-BFGS 500, P normalizado (Enmienda 1), N = 2048 puntos. La unica diferencia con `pre_trainer.run_one`
es que la cadencia (`every`) y el recalculo al arrancar L-BFGS son parametros. Brazos: B, E1000L (= configuracion de 1-B), E1000, E500L, E500 (L = con recalculo al arrancar L-BFGS).
Uso: python diag_e_burgers.py --arms B,E1000L,E1000,E500L,E500 --seeds 3001 --out DIR"""
import argparse, json, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
ROOT = HERE.parents[2]
sys.path.insert(0, str(BUNDLE or ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning"))
import pre_trainer as pt  # noqa: E402
pc, gp, sw = pt.pc, pt.gp, pt.sw
ARMS = {"B": None, "E1000L": (1000, True), "E1000": (1000, False), "E500L": (500, True), "E500": (500, False)}


def run_one(problem, arm, seed, adam_steps, lbfgs_iters=500, device="cpu", dtype="float32", n_points=2048):
    cfg = pt.PROBLEMS[problem]
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = pc.make_problem(cfg["name"], str(cfg["ref"]))
    model = prob.build("rbf", cfg["width"], 11, seed).to(device=dev, dtype=dt)
    P = prob.pools(seed, dev, dt)
    pre, lb_refresh = None, False
    if ARMS[arm] is not None:
        every, lb_refresh = ARMS[arm]
        pre = gp.GramPreconditioner("E", every=every, n_points=n_points)
        pre.attach(model)
    t0 = time.perf_counter()
    A = pt.adam_with_pre(prob, model, P, adam_steps, pre, sync=sync)
    adam_state = sw.clone_state(model)
    out = dict(problem=problem, arm=arm, seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=time.perf_counter() - t0, adam_status=A["status"],
               pre_diag=A["pre_diag"], hist=A["hist"], lbfgs_start_refresh=lb_refresh, scientific_accepted=False)
    if A["status"] != "adam_complete":
        return out
    if pre is not None and lb_refresh:
        d = pre.refresh(model, prob, P["residual"], None, sync=sync)
        d["step"] = "lbfgs_start"
        out["pre_diag"].append(d)
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_iters, chunk=50, snapshot_every=1e12, sync=sync)
    out.update(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], total_seconds=time.perf_counter() - t0)
    out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
    out["score_adam"] = sw.score_run(prob, model, dev, dt, adam_state)
    out["score_final"] = sw.score_run(prob, model, dev, dt, L["final_state"])
    out["primary_adam"], out["primary_final"] = sw.primary(out["score_adam"]), sw.primary(out["score_final"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="burgers", choices=["burgers", "ac"])
    ap.add_argument("--arms", default="B,E1000L,E1000,E500L,E500")
    ap.add_argument("--seeds", default="3001")
    ap.add_argument("--adam", type=int, default=0, help="0 = 2000 en Burgers, 4000 en Allen-Cahn (convencion T1)")
    ap.add_argument("--lbfgs-iters", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pre-n", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in map(int, a.seeds.split(",")):
        for arm in a.arms.split(","):
            f = out / f"{a.problem}-{arm}-seed{s}.json"
            if f.exists():
                continue
            r = run_one(a.problem, arm, s, a.adam or {"burgers": 2000, "ac": 4000}[a.problem], a.lbfgs_iters, a.device, a.dtype, a.pre_n)
            f.write_text(json.dumps(r))
            print(a.problem, arm, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
