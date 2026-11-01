"""EXP-0008 fast-track, Allen-Cahn: exact L-BFGS continuation of the S3 checkpoints (CPU or GPU) + signal battery.

Restores the S3 optimizer state (history included) so the run continues the S3 trajectory, with the
same L-BFGS settings as the S3 worker (max_iter=1, max_eval=8, strong Wolfe, history 100).
Training-side only: no ``evals`` import, no reference access.
"""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.research_pipeline.sciml.model import build_model, derivatives  # noqa: E402
from src.research_pipeline.sciml import signal_battery as sb  # noqa: E402

KO = REPO / "var/agent_runs/baseline-competitive-2026-09-27"
AC_ROOT = KO / "kaggle-cli-fetch/allen_cahn/allen-cahn-t4x2"
S3_ROOT = KO / "kaggle-outputs/closure-s3-retry-20260928/closure-s3-20260928/runs"
sys.path.insert(0, str(AC_ROOT))
import run_campaign as campaign  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(a):
    torch.set_default_dtype(torch.float64); torch.set_num_threads(int(os.environ.get("NT", "4")))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src_ckpt = S3_ROOT / "AC-RBF-Sobol" / f"seed-{a.seed}" / "checkpoint-final.pt"
    pool_path = AC_ROOT / "campaign/pools" / f"seed-{a.seed}.npz"
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    model = build_model("allen_cahn", {"id": "rbf_fixed"}).double(); model.load_state_dict(ck["state_dict"])
    ref_model = campaign.build_model("AC-RBF-Sobol", a.seed).double(); ref_model.load_state_dict(ck["state_dict"])
    probe = torch.rand(64, 2, dtype=torch.float64)
    assert torch.allclose(model(probe), ref_model(probe), atol=1e-12), "src model != campaign model"
    pools = dict(np.load(pool_path)); pts = torch.tensor(pools["interior"], dtype=torch.float64)
    params = sb.problem_parameters("allen_cahn")
    res = sb.residual_fn("allen_cahn", params)

    def loss_fn():
        x = pts.clone().requires_grad_(True); u = model(x); ux, ut, uxx = derivatives(u, x)
        return ((res(u, ux, ut, uxx) / params["residual_scale"]) ** 2).mean()

    grid = sb.dev_grid("allen_cahn")
    pool_pts = pools["interior"]

    def signals(tag):
        t0 = time.time()
        dense, sc = sb.battery(model, "allen_cahn", grid, grid_shape=(129, 65), parameters=params)
        pf = sb.dense_features(model, "allen_cahn", pool_pts, params, sensitivity_points=0)
        sc.update(sb.pool_residual_stats(pf, (np.abs(pool_pts[:, 0]) < 0.8)))  # region proxy: bulk in x
        for layer in (0, 1):
            for k, v in sb.basis_influence(model, "allen_cahn", grid, params, layer=layer).items():
                sc[k + "/max"] = float(v.max()); sc[k + "/argmax"] = int(v.argmax()); dense[k] = v
        np.savez_compressed(out / f"dense_signals_{tag}.npz", **dense)
        sc["signals/seconds"] = time.time() - t0
        (out / f"signals_{tag}.json").write_text(json.dumps(sc, indent=1, default=float))

    def save(tag, step):
        torch.save({"problem": "allen_cahn", "model_id": "rbf_fixed", "model_family": "rbf", "state_dict": model.state_dict(),
                    "step": step, "phase": "lbfgs_continuation", "tag": tag, "seed": a.seed}, out / f"{tag}.pt")

    save("c_adam_plus_s3", 0); signals("c_adam_plus_s3")
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=8, tolerance_grad=1e-8,
                            tolerance_change=0.0, history_size=100, line_search_fn="strong_wolfe")
    opt.load_state_dict(ck["optimizer_state_dict"])
    last = {}

    def closure():
        opt.zero_grad(); l = loss_fn(); l.backward(); last["loss"] = float(l.detach())
        last["gnorm"] = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))); return l

    start_iters = int(opt.state[opt._params[0]].get("n_iter", 0))
    t0 = time.time(); k = 0; saved = set(); stop = "max_outer"
    with open(out / "trace.jsonl", "w") as tr:
        while k < a.max_outer:
            opt.step(closure); k += 1
            st = opt.state[opt._params[0]]
            if k % 10 == 0 or k == 1:
                tr.write(json.dumps({"iter": k, "lbfgs_total_iters": int(st.get("n_iter", 0)), "sec": round(time.time() - t0, 2),
                                     "loss": last["loss"], "gnorm2": last["gnorm"], "func_evals": int(st.get("func_evals", 0)),
                                     "step_len": float(st.get("t", float("nan"))), "curv_pairs": len(st.get("old_dirs", []))}) + "\n"); tr.flush()
            el = time.time() - t0
            for fr in (0.25, 0.5, 0.75):
                if fr not in saved and el >= fr * a.seconds:
                    save(f"c_lbfgs_{int(fr * 100)}", k); saved.add(fr)
            if not np.isfinite(last["loss"]): stop = "nonfinite"; break
            if el >= a.seconds: stop = "censored_time"; break
            if last["gnorm"] < 1e-8: stop = "tolerance_grad"; break
    save("c_lbfgs_100", k); signals("c_lbfgs_100")
    rec = {"schema": "af.registry.run/1", "experiment_id": "EXP-0008", "stage": "fast_track_FT", "evidence_kind": "research",
           "problem_id": "allen_cahn_1d", "model_id": "rbf_fixed", "seed": a.seed, "recipe_id": "AF-R1+S3+LBFGS(continuation)",
           "device": "cpu", "platform": platform.platform(), "torch": torch.__version__,
           "pool_sha256": sha256(pool_path), "source_ckpt_sha256": sha256(src_ckpt),
           "phases": [{"name": "lbfgs_continuation", "iterations": k, "s3_iterations_before": start_iters, "seconds": round(time.time() - t0, 2), "stop": stop}],
           "final_loss": last.get("loss"), "status": "censored_time" if stop == "censored_time" else "complete",
           "technical_pass": None, "scientific_accepted": False, "reference_accepted": False,
           "scores": {"V": None, "E": None, "external": None, "null_reasons": {"all": "scoring_job_pending"}}}
    (out / "run_record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"seed": a.seed, "iters": k, "stop": stop, "loss": last.get("loss")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=900); p.add_argument("--max-outer", type=int, default=100000)
    run(p.parse_args())
