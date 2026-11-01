"""EXP-0008 scoring job (Allen-Cahn): weighted space-time L2 against the two frozen candidate references.
Same evaluator (``run_campaign.evaluate_field``) and same 512x256 interpolation as the S3 scorer.
Writes dense |error| labels on the 129x65 development grid using the ETDRK4 candidate."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.research_pipeline.sciml.model import build_model  # noqa: E402
from src.research_pipeline.sciml import signal_battery as sb  # noqa: E402

KO = REPO / "var/agent_runs/baseline-competitive-2026-09-27"
AC_ROOT = KO / "kaggle-cli-fetch/allen_cahn/allen-cahn-t4x2"
sys.path.insert(0, str(AC_ROOT))
import run_campaign as campaign  # noqa: E402

REFS = {"FD8192": AC_ROOT / "campaign/reference/ac_gpu_reference_candidate.npz",
        "ETDRK4": KO / "kaggle-outputs/closure-s1-20260928/closure-s1/allen_cahn/etdrk4_reference.npz"}


def load_reference(path):
    with np.load(path, allow_pickle=False) as z:
        xs, ts, f = np.array(z["x"]), np.array(z["t"]), np.array(z["u"])
    if f.shape == (len(xs), len(ts)): f = f.T
    xe, te = np.linspace(-1, 1, 512), np.linspace(0, 1, 256)
    ax = np.stack([np.interp(xe, xs, r) for r in f], 0)
    return {"x": xe, "t": te, "u": np.stack([np.interp(te, ts, ax[:, i]) for i in range(len(xe))], 1)}


def score(path, labels=None):
    torch.set_default_dtype(torch.float64)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model("allen_cahn", {"id": "rbf_fixed"}).double(); model.load_state_dict(ck["state_dict"]); model.eval()
    out = {"checkpoint": str(path), "tag": ck.get("tag"), "seed": ck.get("seed")}
    refs = {k: load_reference(p) for k, p in REFS.items()}
    for k, r in refs.items():
        out[k] = campaign.evaluate_field(model, r, torch.device("cpu"))
    if labels:
        dev = sb.dev_grid("allen_cahn")  # x-major ordering, x in [-1,1], t in [0,1]
        r = refs["ETDRK4"]
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((r["t"], r["x"]), r["u"])
        uref = f(np.stack([dev[:, 1], dev[:, 0]], 1))
        with torch.no_grad(): u = model(torch.tensor(dev)).squeeze(-1).numpy()
        np.savez_compressed(labels, abs_error=np.abs(u - uref), x=dev[:, 0], t=dev[:, 1])
        out["V_dev_grid_l2re"] = float(np.linalg.norm(u - uref) / np.linalg.norm(uref))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--labels", action="store_true"); a = p.parse_args(); rows = []
    for c in a.ckpt:
        c = Path(c); rows.append(score(c, c.with_name(f"labels_{c.stem}.npz") if a.labels else None))
        print(json.dumps({"c": str(c.name), "seed": rows[-1]["seed"], "FD": rows[-1]["FD8192"]["relative_weighted_space_time_l2"],
                          "ET": rows[-1]["ETDRK4"]["relative_weighted_space_time_l2"]}), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
