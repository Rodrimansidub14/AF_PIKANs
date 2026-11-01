#!/usr/bin/env python3
"""EXP-0011 (pre-registered 2026-10-03): RAD arm on the EXP-0008 Heat RBF control.

Identical to the control (heat_stepmatched_runner.py): same seed-wise initial weights, same saved Sobol pool as the starting point, full-batch 4,096
interior points, float64, Adam lr 1e-3 for 20,000 updates, residual (u_t - 0.1 u_xx)/1.1, hard trace sin(pi x)+t x (1-x) raw.
Only the interior points change: every REFRESH steps the |residual| is evaluated on a fixed 16,384-point Sobol candidate set and 4,096 points are
re-sampled with src/mechanisms/rad.py::sample_rad (power 2, uniform_fraction 0.5, no importance correction). No solution values, evaluation grids or
verifier code are used by the mechanism. After Adam: score (end-of-Adam, the primary metric), then full L-BFGS (stress test) on the last RAD points.
Usage: python heat_rad_runner.py SEED --dir payload --out results [--adam-steps 20000] [--device cuda:0] [--lbfgs-seconds 700]
"""
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from scipy.stats import qmc

ap = argparse.ArgumentParser()
ap.add_argument("seed", type=int)
ap.add_argument("--dir", default="payload")
ap.add_argument("--out", default="results")
ap.add_argument("--adam-steps", type=int, default=20000)
ap.add_argument("--refresh", type=int, default=1000)
ap.add_argument("--candidates", type=int, default=16384)
ap.add_argument("--power", type=float, default=2.0)
ap.add_argument("--uniform", type=float, default=0.5)
ap.add_argument("--lbfgs-seconds", type=float, default=700.0)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

import sys
sys.path.insert(0, a.dir)
from af_sciml_model import build_model  # noqa: E402
from rad import sample_rad  # noqa: E402  (copy of src/mechanisms/rad.py)

ALPHA, RS = 0.1, 1.1
S = a.seed
dev = torch.device(a.device)
torch.manual_seed(S); np.random.seed(S)
torch.set_default_dtype(torch.float64)
out = Path(a.out) / f"seed-{S}"
out.mkdir(parents=True, exist_ok=True)
z = np.load(Path(a.dir) / f"heat-seed-{S}.npz")
pool0 = np.array(z["interior"], dtype=np.float64)
model = build_model("heat", {"id": "rbf_fixed"}).to(device=dev, dtype=torch.float64)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
cand_np = qmc.Sobol(2, scramble=True, seed=10 * S + 7).random_base2(int(np.log2(a.candidates)))
cand = torch.tensor(cand_np, dtype=torch.float64, device=dev)
rng = np.random.default_rng(10 * S + 8)


def residual(pts):
    pts = pts.clone().requires_grad_(True)
    o = model(pts)
    g = torch.autograd.grad(o, pts, torch.ones_like(o), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], pts, torch.ones_like(g[:, 0:1]), create_graph=True, retain_graph=True)[0][:, 0:1]
    return (g[:, 1:2] - ALPHA * uxx) / RS


def loss_on(points):
    return residual(points).square().mean()


def trap(n):
    w = torch.ones(n, dtype=torch.float64, device=dev); w[0] = w[-1] = 0.5
    return w / (n - 1)


def score():
    nx, nt = 512, 256
    x = torch.linspace(0, 1, nx, device=dev, dtype=torch.float64); t = torch.linspace(0, 1, nt, device=dev, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    with torch.no_grad():
        pred = model(torch.stack((xx.reshape(-1), tt.reshape(-1)), 1)).reshape(nx, nt)
    ex = torch.sin(torch.pi * xx) * torch.exp(-ALPHA * torch.pi ** 2 * tt)
    w = trap(nx)[:, None] * trap(nt)[None, :]
    return float(torch.sqrt(torch.sum(w * (pred - ex) ** 2) / torch.sum(w * ex ** 2)))


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


points = torch.tensor(pool0, dtype=torch.float64, device=dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sync(); t0 = time.perf_counter(); refresh_s = 0.0; refreshes = []; hist = []
for step in range(1, a.adam_steps + 1):
    if a.refresh and step > 1 and (step - 1) % a.refresh == 0:
        sync(); tr = time.perf_counter()
        parts = []
        for i in range(0, len(cand), 4096):
            parts.append(residual(cand[i:i + 4096]).detach().abs().cpu())
        r = torch.cat(parts).numpy().reshape(-1)
        s = sample_rad(cand_np, r, len(pool0), rng, power=a.power, uniform_fraction=a.uniform)
        points = torch.tensor(s["points"], dtype=torch.float64, device=dev)
        sync(); dt = time.perf_counter() - tr; refresh_s += dt
        refreshes.append({"step": step - 1, "unique": s["unique_count"], "effective_candidates": s["effective_candidates"], "seconds": dt})
    opt.zero_grad(set_to_none=True)
    l = loss_on(points)
    l.backward()
    opt.step()
    if step % 500 == 0:
        hist.append({"step": step, "loss": float(l.detach())})
sync(); adam_s = time.perf_counter() - t0
adam_score = score()
torch.save({"state_dict": model.state_dict(), "stage": "adam_final_rad"}, out / "checkpoint-adam-final.pt")
print(json.dumps({"seed": S, "params": params, "adam_rel_l2": adam_score, "adam_s": adam_s, "refresh_s": refresh_s}), flush=True)

CHUNK, MAXEVAL, MAXITER = 25, 125, 15000
lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=CHUNK, max_eval=MAXEVAL, tolerance_grad=1e-8, tolerance_change=0.0, history_size=100, line_search_fn="strong_wolfe")


def closure():
    lb.zero_grad(set_to_none=True)
    v = loss_on(points)
    v.backward()
    return v


last = float(loss_on(points).detach()); it = 0; plateau = 0; reason = "max_iterations"; t1 = time.perf_counter()
while it < MAXITER:
    if time.perf_counter() - t1 >= a.lbfgs_seconds:
        reason = "time_cap"; break
    b = int(lb.state[lb._params[0]].get("n_iter", 0))
    lb.step(closure)
    d = int(lb.state[lb._params[0]].get("n_iter", b)) - b
    it += d
    cur = float(loss_on(points).detach())
    if not np.isfinite(cur):
        reason = "nonfinite"; break
    if d == 0:
        reason = "no_progress_chunk"; break
    plateau = plateau + 1 if (last - cur) / max(abs(last), 1e-300) < 1e-10 else 0
    last = cur
    if plateau >= 5:
        reason = "loss_plateau"; break
sync()
final = {"seed": S, "experiment": "EXP-0011", "mechanism": "rad power=%g uniform=%g refresh=%d candidates=%d" % (a.power, a.uniform, a.refresh, a.candidates),
         "params": params, "adam_steps": a.adam_steps, "adam_seconds": adam_s, "refresh_seconds": refresh_s, "end_of_adam_rel_l2_512x256": adam_score,
         "lbfgs_iters": it, "lbfgs_stop_reason": reason, "lbfgs_seconds": time.perf_counter() - t1, "after_lbfgs_rel_l2_512x256": score(),
         "refreshes": refreshes[:3] + refreshes[-2:], "n_refreshes": len(refreshes), "loss_history": hist, "pool_sha256": str(z["pool_sha256"]),
         "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", "scientific_accepted": False, "reference_accepted": False}
torch.save({"state_dict": model.state_dict(), "stage": "lbfgs_final_rad", "iters": it}, out / "checkpoint-lbfgs-final.pt")
json.dump(final, open(out / "result.json", "w"), indent=2)
print("FINAL", json.dumps({k: v for k, v in final.items() if k not in ("refreshes", "loss_history")}), flush=True)
