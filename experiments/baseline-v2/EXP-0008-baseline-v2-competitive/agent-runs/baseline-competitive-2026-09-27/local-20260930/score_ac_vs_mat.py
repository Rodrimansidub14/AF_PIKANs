"""Step 1 (local, CPU, no training): score archived Allen-Cahn RBF/MLP checkpoints against the
third-party DeepXDE `Allen_Cahn.mat` reference, using DeepXDE's own metric (unweighted relative L2 on the
201x101 grid). Two endpoints per arm/seed: Adam endpoint (600 s time-capped, allen-cahn-t4x2-v1) and
after 240 s L-BFGS continuation (closure-s3-retry). Read-only; writes CSV/JSON next to this script.
Run: py -3.13 score_ac_vs_mat.py
"""
import csv
import hashlib
import importlib.util
import json
import statistics as st
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]
KO = ROOT / "kaggle-outputs"
SRC = KO / "allen-cahn-score-only-v2-20260928/allen-cahn-score-only/run-1790600803211630363/run_campaign_source.py"
MAT = ROOT.parent / "baseline-close-2026-09-27/references/Allen_Cahn.mat"
ADAM = KO / "allen-cahn-t4x2-v1-20260928/allen-cahn-t4x2/campaign"
LBFGS = KO / "closure-s3-retry-20260928/closure-s3-20260928/runs"

spec = importlib.util.spec_from_file_location("ac_src", SRC)
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

m = sio.loadmat(MAT)
x = m["x"].reshape(-1)
t = m["t"].reshape(-1)
u_ref = m["u"]  # (nt, nx)
assert u_ref.shape == (len(t), len(x))
xx, tt = np.meshgrid(x, t, indexing="xy")  # (nt, nx)
coords = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=torch.float64)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


rows = []
for arm in ("AC-RBF-Sobol", "AC-DeepXDE-MLP"):
    for seed in range(401, 406):
        for stage, path in (("adam_endpoint_600s", ADAM / arm / f"seed-{seed}" / "checkpoint-final.pt"),
                            ("plus_lbfgs_240s", LBFGS / arm / f"seed-{seed}" / "checkpoint-final.pt")):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = ac.build_model(arm, seed).to(torch.float64)
            model.load_state_dict(ck["state_dict"], strict=True)
            model.eval()
            with torch.no_grad():
                pred = model(coords).squeeze(1).reshape(u_ref.shape).numpy()
            err = pred - u_ref
            rows.append({
                "arm": arm, "seed": seed, "stage": stage,
                "l2_rel_mat": float(np.linalg.norm(err) / np.linalg.norm(u_ref)),
                "max_abs_mat": float(np.abs(err).max()),
                "ckpt_sha256": sha(path),
            })

out = Path(__file__).parent
with open(out / "ac-vs-mat-per-seed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

summ = {}
for arm in ("AC-RBF-Sobol", "AC-DeepXDE-MLP"):
    for stage in ("adam_endpoint_600s", "plus_lbfgs_240s"):
        xs = [r["l2_rel_mat"] for r in rows if r["arm"] == arm and r["stage"] == stage]
        summ[f"{arm}|{stage}"] = {"n": len(xs), "mean": st.mean(xs), "sd": st.stdev(xs), "min": min(xs), "max": max(xs)}
# paired RBF-MLP per seed and stage
for stage in ("adam_endpoint_600s", "plus_lbfgs_240s"):
    d = []
    for s in range(401, 406):
        a = next(r for r in rows if r["arm"] == "AC-RBF-Sobol" and r["seed"] == s and r["stage"] == stage)
        b = next(r for r in rows if r["arm"] == "AC-DeepXDE-MLP" and r["seed"] == s and r["stage"] == stage)
        d.append(a["l2_rel_mat"] - b["l2_rel_mat"])
    summ[f"paired_RBF_minus_MLP|{stage}"] = {"diffs": d, "mean": st.mean(d), "sd": st.stdev(d)}
json.dump(summ, open(out / "ac-vs-mat-summary.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
