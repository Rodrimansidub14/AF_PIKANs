"""EXP-0012: L-BFGS on the CONTROL's uniform pool, starting from the RAD end-of-Adam checkpoints of EXP-0011 (no retraining). Adapted from heat_lbfgs_gpu.py.
ORIGINAL: GPU redo of the Heat L-BFGS stage (same algorithm as heat_lbfgs_redo.py, CUDA float64, per-seed time cap).
Starts from the archived S2 checkpoint-adam-final.pt of seeds 901-905 + their saved pools. Chunk max_eval=125; stops on
zero-progress chunk, plateau, iteration cap 15000 or --seconds. Scored on analytic 512x256 (same formula as archived scorer).
Usage: python heat_lbfgs_gpu.py SEED --seconds 700 --dir /kaggle/working/payload --out /kaggle/working/results"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument("seed", type=int); ap.add_argument("--seconds", type=float, default=700.0)
ap.add_argument("--dir", default="payload"); ap.add_argument("--out", default="results"); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
import sys; sys.path.insert(0, a.dir)
from af_sciml_model import build_model
torch.set_default_dtype(torch.float64)
dev = torch.device(a.device)
ALPHA, RS, MAXITER, CHUNK, MAXEVAL = 0.1, 1.1, 15000, 25, 125
S = a.seed; out = Path(a.out) / f"seed-{S}"; out.mkdir(parents=True, exist_ok=True)
ck = torch.load(Path(a.dir) / f"seed-{S}-adam-final.pt", map_location="cpu", weights_only=False)
assert ck["stage"] == "adam_final_rad", ck["stage"]
model = build_model("heat", {"id": "rbf_fixed"}); model.load_state_dict(ck["state_dict"]); model = model.to(dev).double()
z = np.load(Path(a.dir) / f"heat-seed-{S}.npz")
pts = torch.tensor(z["interior"], dtype=torch.float64, device=dev, requires_grad=True)


def trap(n):
    w = torch.ones(n, dtype=torch.float64, device=dev); w[0] = w[-1] = 0.5; return w / (n - 1)


def loss_fn():
    o = model(pts)
    g = torch.autograd.grad(o, pts, torch.ones_like(o), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], pts, torch.ones_like(g[:, 0:1]), create_graph=True, retain_graph=True)[0][:, 0:1]
    return torch.mean(((g[:, 1:2] - ALPHA * uxx) / RS).square())


def score():
    nx, nt = 512, 256
    x = torch.linspace(0, 1, nx, device=dev); t = torch.linspace(0, 1, nt, device=dev)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    with torch.no_grad():
        pred = model(torch.stack((xx.reshape(-1), tt.reshape(-1)), 1)).reshape(nx, nt)
    ex = torch.sin(torch.pi * xx) * torch.exp(-ALPHA * torch.pi ** 2 * tt)
    w = trap(nx)[:, None] * trap(nt)[None, :]
    return float(torch.sqrt(torch.sum(w * (pred - ex) ** 2) / torch.sum(w * ex ** 2)))


opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=CHUNK, max_eval=MAXEVAL, tolerance_grad=1e-8,
                        tolerance_change=0.0, history_size=100, line_search_fn="strong_wolfe")
evals = 0


def closure():
    global evals
    opt.zero_grad(set_to_none=True); v = loss_fn(); v.backward(); evals += 1; return v


(torch.cuda.synchronize() if dev.type == "cuda" else None); t0 = time.perf_counter()
last = float(loss_fn().detach()); curve = [{"iters": 0, "loss": last, "rel_l2": score(), "evals": 0, "sec": 0.0}]
print(json.dumps(curve[0]), flush=True)
iters, plateau, reason = 0, 0, "max_iterations"; marks = [50, 200, 1000, 3000, 7500, 15000]
while iters < MAXITER:
    if time.perf_counter() - t0 >= a.seconds: reason = "time_cap"; break
    before = int(opt.state[opt._params[0]].get("n_iter", 0)); opt.step(closure)
    after = int(opt.state[opt._params[0]].get("n_iter", before)); d = after - before; iters += d
    cur = float(loss_fn().detach())
    if not np.isfinite(cur): reason = "nonfinite"; break
    if d == 0: reason = "no_progress_chunk"; break
    plateau = plateau + 1 if (last - cur) / max(abs(last), 1e-300) < 1e-10 else 0; last = cur
    if any(iters >= m and curve[-1]["iters"] < m for m in marks):
        curve.append({"iters": iters, "loss": cur, "rel_l2": score(), "evals": evals, "sec": time.perf_counter() - t0})
        print(json.dumps(curve[-1]), flush=True)
    if plateau >= 5: reason = "loss_plateau"; break
(torch.cuda.synchronize() if dev.type == "cuda" else None)
final = {"seed": S, "iters": iters, "evals": evals, "stop_reason": reason, "loss": last, "rel_l2_512x256": score(),
         "lbfgs_seconds": time.perf_counter() - t0, "adam_final_rel_l2": curve[0]["rel_l2"], "curve": curve,
         "pool_sha256": str(z["pool_sha256"]), "device": (torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"), "pool": "control uniform pool (EXP-0008) heat-seed-S.npz interior", "experiment": "EXP-0012", "scientific_accepted": False}
torch.save({"state_dict": model.state_dict(), "stage": "lbfgs_uniform_after_rad", "iters": iters}, out / "checkpoint-lbfgs-uniform.pt")
json.dump(final, open(out / "result.json", "w"), indent=2)
print("FINAL", json.dumps({k: v for k, v in final.items() if k != "curve"}), flush=True)
