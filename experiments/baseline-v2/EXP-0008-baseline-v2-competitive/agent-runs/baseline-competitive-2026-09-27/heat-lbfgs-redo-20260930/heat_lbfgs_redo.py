"""Local CPU redo of the Heat L-BFGS stage, starting from the archived S2 Adam-final checkpoints (seeds 901-905).

Why: the archived S2 L-BFGS chunked PyTorch L-BFGS as max_iter=25 / max_eval=32 and treated any chunk with fewer than
25 iterations as 'optimizer_stopped_early' and aborted. With strong-Wolfe line search a 32-evaluation budget usually ends a
chunk before 25 iterations, so 4 of 5 seeds stopped after 22-73 iterations: a harness artefact, not convergence.
Here: same model, same saved pool, same full-batch loss, float64, same L-BFGS settings (lr=1, history 100, strong Wolfe,
tolerance_grad 1e-8, tolerance_change 0) but chunk max_eval=125, stopping only when a chunk makes zero progress, on a loss
plateau, or at the DeepXDE-style cap of 15,000 iterations. Scored on the analytic 512x256 grid (same formula as the archived
scorer). Usage: py -3.13 heat_lbfgs_redo.py SEED
"""
import json, sys, time, hashlib
from pathlib import Path
import numpy as np, torch

SEED = int(sys.argv[1])
ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2"
OUT = Path(__file__).resolve().parent / f"seed-{SEED}"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(S2 / "payload"))
from af_sciml_model import build_model
from heat_common_eval import _trap_weights

torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
ALPHA, RS, MAXITER, CHUNK, MAXEVAL = 0.1, 1.1, 15000, 25, 125
ck = torch.load(S2 / f"runs/tier2/seed-{SEED}/rbf/checkpoint-adam-final.pt", map_location="cpu", weights_only=False)
assert ck["stage"] == "adam_complete" and ck["adam_step"] == 20000
model = build_model("heat", {"id": "rbf_fixed"}); model.load_state_dict(ck["state_dict"]); model = model.double()
z = np.load(S2 / f"pools/heat-seed-{SEED}.npz")
pts = torch.tensor(z["interior"], dtype=torch.float64, requires_grad=True)


def loss_fn():
    out = model(pts)
    g = torch.autograd.grad(out, pts, torch.ones_like(out), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], pts, torch.ones_like(g[:, 0:1]), create_graph=True, retain_graph=True)[0][:, 0:1]
    return torch.mean(((g[:, 1:2] - ALPHA * uxx) / RS).square())


def score():
    nx, nt = 512, 256
    x = torch.linspace(0, 1, nx); t = torch.linspace(0, 1, nt)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    with torch.no_grad():
        pred = model(torch.stack((xx.reshape(-1), tt.reshape(-1)), 1)).reshape(nx, nt)
    ex = torch.sin(torch.pi * xx) * torch.exp(-ALPHA * torch.pi ** 2 * tt)
    w = _trap_weights(nx, torch.device("cpu"))[:, None] * _trap_weights(nt, torch.device("cpu"))[None, :]
    return float(torch.sqrt(torch.sum(w * (pred - ex) ** 2) / torch.sum(w * ex ** 2)))


opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=CHUNK, max_eval=MAXEVAL, tolerance_grad=1e-8,
                        tolerance_change=0.0, history_size=100, line_search_fn="strong_wolfe")
evals = 0


def closure():
    global evals
    opt.zero_grad(set_to_none=True)
    v = loss_fn(); v.backward(); evals += 1
    return v


t0 = time.perf_counter()
with torch.no_grad(): pass
last = float(loss_fn().detach())
curve = [{"iters": 0, "loss": last, "rel_l2_512x256": score(), "evals": 0, "sec": 0.0}]
print(json.dumps(curve[0]), flush=True)
iters, plateau, reason, marks = 0, 0, "max_iterations", [50, 200, 1000, 3000, 7500]
while iters < MAXITER:
    before = int(opt.state[opt._params[0]].get("n_iter", 0))
    opt.step(closure)
    after = int(opt.state[opt._params[0]].get("n_iter", before)); d = after - before
    iters += d
    cur = float(loss_fn().detach())
    if not np.isfinite(cur): reason = "nonfinite"; break
    if d == 0: reason = "no_progress_chunk"; break
    plateau = plateau + 1 if (last - cur) / max(abs(last), 1e-300) < 1e-10 else 0
    last = cur
    if any(iters >= m and curve[-1]["iters"] < m for m in marks):
        curve.append({"iters": iters, "loss": cur, "rel_l2_512x256": score(), "evals": evals, "sec": time.perf_counter() - t0})
        print(json.dumps(curve[-1]), flush=True)
    if plateau >= 5: reason = "loss_plateau"; break
final = {"seed": SEED, "iters": iters, "evals": evals, "stop_reason": reason, "loss": last, "rel_l2_512x256": score(),
         "sec": time.perf_counter() - t0, "adam_final_rel_l2": None, "curve": curve,
         "pool_sha256": str(z["pool_sha256"]), "start_ckpt": "checkpoint-adam-final.pt (20000 Adam)",
         "note": "local CPU float64 redo; scientific_accepted=false"}
torch.save({"state_dict": model.state_dict(), "stage": "lbfgs_redo", "iters": iters}, OUT / "checkpoint-lbfgs-redo.pt")
json.dump(final, open(OUT / "result.json", "w"), indent=2)
print("FINAL", json.dumps({k: v for k, v in final.items() if k != "curve"}), flush=True)
