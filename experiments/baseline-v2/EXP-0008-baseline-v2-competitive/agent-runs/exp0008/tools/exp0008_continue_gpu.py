"""EXP-0008 fast-track: L-BFGS continuation of an existing checkpoint (CPU or GPU) + signal battery.

Training-side only: this script never imports ``evals`` and never reads a reference solution.
Scoring against references is done afterwards by ``exp0008_score.py`` on the saved checkpoints.

Budget: min(max_outer steps, seconds). A run stopped by the time cap is marked ``censored_time``.
"""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.research_pipeline.sciml.model import build_model, derivatives  # noqa: E402
from src.research_pipeline.sciml import signal_battery as sb  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_losses(model, problem, params, pools, device):
    T = lambda k: torch.tensor(pools[k], dtype=torch.float64, device=device)
    xi = T("interior"); scale = params["residual_scale"]
    res = sb.residual_fn(problem, params)
    conds = {k: T(k) for k in ("left", "right", "initial") if k in pools and problem == "burgers"}

    def pde():
        x = xi.clone().requires_grad_(True)
        u = model(x); ux, ut, uxx = derivatives(u, x)
        return (((res(u, ux, ut, uxx)) / scale) ** 2).mean()

    def ic():
        x0 = conds["initial"]
        return ((model(x0).squeeze(-1) + torch.sin(torch.pi * x0[:, 0])) ** 2).mean()

    def bc():
        return (model(conds["left"]) ** 2).mean() + (model(conds["right"]) ** 2).mean()

    if problem == "burgers":
        total = lambda: pde() + ic() + bc()
        terms = {"pde": pde, "ic": ic, "bc": bc}
    else:
        total = pde
        terms = None
    return total, terms


def run(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(os.environ.get("NT", "4")))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    problem = ck["problem"]
    params = sb.problem_parameters(problem, ck.get("problem_parameters"))
    model = build_model(problem, {"id": ck["model_id"]}).double().to(device)
    model.load_state_dict(ck["state_dict"])
    pools = dict(np.load(args.pools))
    total, terms = build_losses(model, problem, params, pools, device)
    grid = sb.dev_grid(problem)
    pool_pts = pools["interior"]
    region = None
    if problem == "burgers":
        region = (np.abs(pool_pts[:, 0]) <= 5 * params["nu"]) & (pool_pts[:, 1] >= 1 / np.pi)

    def signals(tag):
        import copy
        t0 = time.time()
        m = copy.deepcopy(model).to("cpu")  # battery runs on a CPU copy (device-agnostic, no cuda tensors)
        _, terms_cpu = build_losses(m, problem, params, pools, "cpu")
        dense, scalars = sb.battery(m, problem, grid, grid_shape=(129, 65), parameters=params, terms=terms_cpu)
        pool_feat = sb.dense_features(m, problem, pool_pts, params, sensitivity_points=0)
        scalars.update(sb.pool_residual_stats(pool_feat, region))
        for layer in (0, 1):
            inf = sb.basis_influence(m, problem, grid, params, layer=layer)
            for k, v in inf.items():
                scalars[k + "/max"] = float(v.max()); scalars[k + "/argmax"] = int(v.argmax())
                dense[k] = v
        np.savez_compressed(out / f"dense_signals_{tag}.npz", **dense)
        scalars["signals/seconds"] = time.time() - t0
        (out / f"signals_{tag}.json").write_text(json.dumps(scalars, indent=1, default=float))
        return scalars

    (out / "c_adam.pt").write_bytes(Path(args.ckpt).read_bytes())
    signals("c_adam")
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, max_eval=25, history_size=100,
                            tolerance_grad=1e-8, tolerance_change=0.0, line_search_fn="strong_wolfe")
    if args.resume_optimizer and ck.get("optimizer_state_dict"):
        opt.load_state_dict(ck["optimizer_state_dict"])  # exact continuation of a previous L-BFGS run
    last = {}

    def closure():
        opt.zero_grad(); l = total(); l.backward()
        last["loss"] = float(l)
        last["gnorm"] = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
        return l

    def flat():
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def save(tag, step):
        torch.save({"problem": problem, "model_id": ck["model_id"], "model_family": ck.get("model_family"),
                    "problem_parameters": ck.get("problem_parameters"), "validation_problem": ck.get("validation_problem"),
                    "state_dict": model.state_dict(), "step": step, "phase": "lbfgs", "tag": tag,
                    "optimizer_state_dict": opt.state_dict() if tag == "c_lbfgs_100" else None}, out / f"{tag}.pt")

    fractions = [0.25, 0.5, 0.75]; saved = set()
    t0 = time.time(); k = 0; stop = "max_outer"; nonfinite = False
    with open(out / "trace.jsonl", "w") as trace:
        while k < args.max_outer:
            before = flat()
            opt.step(closure); k += 1
            after = flat()
            st = opt.state[opt._params[0]]
            row = {"outer": k, "sec": round(time.time() - t0, 2), "loss": last["loss"], "gnorm2": last["gnorm"],
                   "update_ratio": float((after - before).norm() / after.norm()), "step_len": float(st.get("t", float("nan"))),
                   "n_iter": int(st.get("n_iter", 0)), "func_evals": int(st.get("func_evals", 0)),
                   "curv_pairs": len(st.get("old_dirs", []))}
            trace.write(json.dumps(row) + "\n"); trace.flush()
            if not np.isfinite(last["loss"]):
                stop = "nonfinite"; nonfinite = True; break
            elapsed = time.time() - t0
            for fr in fractions:
                if fr not in saved and elapsed >= fr * args.seconds:
                    save(f"c_lbfgs_{int(fr * 100)}", k); saved.add(fr)
            if elapsed >= args.seconds:
                stop = "censored_time"; break
            if last["gnorm"] < 1e-8:
                stop = "tolerance_grad"; break
    if nonfinite:
        model.load_state_dict(ck["state_dict"])
    save("c_lbfgs_100", k)
    end = signals("c_lbfgs_100")
    record = {"schema": "af.registry.run/1", "experiment_id": "EXP-0008", "stage": "fast_track_FT",
              "evidence_kind": "research", "problem_id": problem, "model_id": ck["model_id"], "seed": args.seed,
              "recipe_id": "AF-R1+LBFGS(continuation)", "device": device, "platform": platform.platform(),
              "torch": torch.__version__, "pool_sha256": sha256(args.pools), "source_ckpt_sha256": sha256(args.ckpt),
              "phases": [{"name": "lbfgs", "outer_steps": k, "seconds": round(time.time() - t0, 2), "stop": stop,
                          "func_evals": int(opt.state[opt._params[0]].get("func_evals", 0))}],
              "final_loss": last.get("loss"), "status": "censored_time" if stop == "censored_time" else ("failed" if nonfinite else "complete"),
              "technical_pass": None, "scientific_accepted": False,
              "scores": {"V": None, "E": None, "external": None, "null_reasons": {"all": "scoring_job_pending"}}}
    (out / "run_record.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"seed": args.seed, "outer": k, "sec": record["phases"][0]["seconds"], "stop": stop, "loss": last.get("loss")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--pools", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--max-outer", type=int, default=15000); p.add_argument("--device", default="cpu")
    p.add_argument("--resume-optimizer", action="store_true")
    run(p.parse_args())
