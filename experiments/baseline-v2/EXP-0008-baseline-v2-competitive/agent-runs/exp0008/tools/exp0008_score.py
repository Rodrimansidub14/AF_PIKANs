"""EXP-0008 scoring job (Burgers). Separate from training: this is the only script that touches references.

Scores: Cole-Hopf 1025x129 (E, binding), PINNacle FEM 101x11 (external; also reports the
FEM-vs-Cole-Hopf floor), and dense |error| labels on the 129x65 development grid.
"""
import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.research_pipeline.sciml.model import build_model  # noqa: E402
from src.research_pipeline.sciml import signal_battery as sb  # noqa: E402
from evals.problems.burgers_pinnacle_1d import reference as ref  # noqa: E402

FEM = Path(__file__).resolve().parents[1] / "external" / "burgers1d.dat"
FEM_SHA = "3b6d31d895022b397330091ace59220167f0918c2ed616ed2ae5b1458eede5eb"
NU = ref.NU_DEFAULT
_cache = {}


def _ch_grid():
    if "ch" not in _cache:
        gx, gt = ref.eval_grid()
        _cache["ch"] = (gx, gt, ref.exact(gx, gt))
    return _cache["ch"]


def _fem():
    if "fem" not in _cache:
        assert hashlib.sha256(FEM.read_bytes()).hexdigest() == FEM_SHA, "FEM file hash mismatch"
        rows = [l.split() for l in FEM.read_text(encoding="utf-8").splitlines() if l and not l.startswith("%")]
        arr = np.array(rows, dtype=float)
        xs = arr[:, 0]; ts = np.linspace(0, 1, arr.shape[1] - 1)
        xx, tt = np.meshgrid(xs, ts, indexing="ij")
        pts = np.stack([xx.ravel(), tt.ravel()], 1)
        _cache["fem"] = (pts, arr[:, 1:].reshape(-1))
    return _cache["fem"]


def predict(model, pts):
    with torch.no_grad():
        return model(torch.tensor(pts, dtype=torch.float64)).squeeze(-1).numpy()


def l2re(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def score_checkpoint(path, label_out=None):
    torch.set_default_dtype(torch.float64)
    ck = torch.load(path, map_location="cpu")
    model = build_model(ck["problem"], {"id": ck["model_id"]}).double(); model.load_state_dict(ck["state_dict"])
    gx, gt, uref = _ch_grid()
    u = predict(model, np.stack([gx, gt], 1))
    front = (np.abs(gx) <= 5 * NU) & (gt >= 1 / np.pi)
    pts, ufem = _fem()
    u_at_fem = predict(model, pts)
    ch_at_fem = ref.exact(pts[:, 0], pts[:, 1])
    out = {"checkpoint": str(path), "step": ck.get("step"), "tag": ck.get("tag"),
           "E_ch_l2re_1025x129": l2re(u, uref),
           "E_front_rmse": float(np.sqrt(np.mean((u - uref)[front] ** 2))),
           "E_complement_rmse": float(np.sqrt(np.mean((u - uref)[~front] ** 2))),
           "E_max_abs_error": float(np.abs(u - uref).max()),
           "external_fem_101x11_l2re": l2re(u_at_fem, ufem),
           "ch_at_fem_points_l2re": l2re(u_at_fem, ch_at_fem),
           "fem_floor_fem_vs_ch": l2re(ufem, ch_at_fem)}
    if label_out is not None:
        dev = sb.dev_grid("burgers")
        err = np.abs(predict(model, dev) - ref.exact(dev[:, 0], dev[:, 1]))
        np.savez_compressed(label_out, abs_error=err, x=dev[:, 0], t=dev[:, 1])
        out["V_dev_grid_l2re"] = l2re(predict(model, dev), ref.exact(dev[:, 0], dev[:, 1]))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--out", required=True); p.add_argument("--labels", action="store_true")
    a = p.parse_args(); rows = []
    for c in a.ckpt:
        c = Path(c)
        lab = c.with_name(f"labels_{c.stem}.npz") if a.labels else None
        rows.append(score_checkpoint(c, lab)); print(json.dumps({k: rows[-1][k] for k in ("checkpoint", "E_ch_l2re_1025x129", "external_fem_101x11_l2re")}), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
