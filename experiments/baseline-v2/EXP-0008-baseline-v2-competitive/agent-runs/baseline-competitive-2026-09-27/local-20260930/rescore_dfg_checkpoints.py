"""Step 2b (local, CPU, no training): re-score every archived DFG checkpoint that is on disk with ONE force
formula (sigma.n, src/research_pipeline/sciml/dfg.py::qoi_from_cylinder), 2048 contour points, float64, and
also with the older tangential formula so the discrepancy between scorers is visible per checkpoint.

Sets: 301-305 RBF pre-L-BFGS (r3/r4 final Adam checkpoints), 301-305 RBF post-L-BFGS (current-cohort
refinement), 501-505 RBF and 501-505 tanh-MLP post-L-BFGS (the 501-505 pre-L-BFGS checkpoints are not on disk).
Run: py -3.13 rescore_dfg_checkpoints.py
"""
import csv
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent
sys.path.insert(0, str(ROOT.parents[2] / "src"))
from research_pipeline.sciml import dfg  # noqa: E402

KO = ROOT / "kaggle-outputs"
OLD_SRC = KO / "dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/run_campaign_source.py"
spec = importlib.util.spec_from_file_location("old_dfg", OLD_SRC)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

N = 2048
CX, CY, R = 0.2, 0.2, 0.05
REF = dfg.FEATFLOW_REFERENCE


def load(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.to(torch.float64).eval()


def derivs(model, pts):
    xy = torch.as_tensor(pts, dtype=torch.float64).requires_grad_(True)
    f = model(xy)
    g = [torch.autograd.grad(f[:, i].sum(), xy, create_graph=False, retain_graph=True)[0].detach().numpy()
         for i in range(2)]
    return f.detach().numpy(), g[0][:, 0], g[0][:, 1], g[1][:, 0], g[1][:, 1]


def score(model):
    theta = 2 * math.pi * np.arange(N) / N
    pts = np.column_stack((CX + R * np.cos(theta), CY + R * np.sin(theta)))
    f, ux, uy, vx, vy = derivs(model, pts)
    p = f[:, 2]
    with torch.no_grad():
        pf = float(model(torch.tensor([[0.15, 0.2]], dtype=torch.float64))[0, 2])
        pr = float(model(torch.tensor([[0.25, 0.2]], dtype=torch.float64))[0, 2])
    q = dfg.qoi_from_cylinder(theta, p, ux, uy, vx, vy, pf, pr)
    nx, ny = np.cos(theta), np.sin(theta)
    tx, ty = -ny, nx
    dut = tx * (ux * nx + uy * ny) + ty * (vx * nx + vy * ny)
    ds = 2 * math.pi * R / N
    s = 2.0 / (dfg.U_MEAN ** 2 * dfg.DIAMETER)
    cd_t = s * float(np.sum(dfg.NU * dut * ny - p * nx) * ds)
    cl_t = s * float(-np.sum(dfg.NU * dut * nx + p * ny) * ds)
    cd_press = s * float(np.sum(-p * nx) * ds)
    cd_visc = s * float(np.sum(dfg.NU * (ux * nx + uy * ny)) * ds)
    return {"Cd_sigma_n": q["drag_coefficient"], "Cl_sigma_n": q["lift_coefficient"], "dp": q["pressure_difference"],
            "Cd_tangential_old": cd_t, "Cl_tangential_old": cl_t, "Cd_pressure_part": cd_press,
            "Cd_viscous_part": cd_visc}


sets = []
r3 = ROOT.parent / "baseline-close-2026-09-27/dfg-re20-gpu-r3/download-attempts/attempt-01/pikan-682b323105591987/runs"
r4 = ROOT.parent / "baseline-close-2026-09-27/dfg-re20-gpu-r4/download-attempts/attempt-01/pikan-c349281ede3181b8/runs"
for seed in range(301, 306):
    p = (r3 if seed < 303 else r4) / f"seed-{seed}" / "dfg_rbf_pikan.pt"
    sets.append(("301-305_RBF_pre_LBFGS", seed, lambda s=seed: dfg.DFGRBFPIKAN(s), p))
    p2 = KO / f"dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261/training/seed-{seed}/checkpoint-lbfgs.pt"
    sets.append(("301-305_RBF_post_LBFGS", seed, lambda s=seed: dfg.DFGRBFPIKAN(s), p2))
base = KO / "dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training"
for seed in range(501, 506):
    sets.append(("501-505_RBF_post_LBFGS", seed, lambda s=seed: old.DFGRBFPIKAN(s), base / f"DFG-RBF-PIKAN/seed-{seed}/checkpoint-lbfgs.pt"))
    sets.append(("501-505_MLP5x64_post_LBFGS", seed, lambda s=seed: old.DFGMLP(s), base / f"DFG-Tanh-MLP/seed-{seed}/checkpoint-lbfgs.pt"))

rows = []
for label, seed, mk, path in sets:
    if not path.exists():
        rows.append({"set": label, "seed": seed, "status": "missing_checkpoint", "path": str(path)})
        continue
    m = load(mk(), path)
    r = score(m)
    r.update({"set": label, "seed": seed, "status": "scored", "path": str(path.relative_to(RUNS)),
              "Cd_relerr_sigma_n": abs(r["Cd_sigma_n"] - REF["drag_coefficient"]) / REF["drag_coefficient"],
              "Cl_abserr_sigma_n": abs(r["Cl_sigma_n"] - REF["lift_coefficient"]),
              "dp_relerr": abs(r["dp"] - REF["pressure_difference"]) / REF["pressure_difference"],
              "Cd_relerr_tangential_old": abs(r["Cd_tangential_old"] - REF["drag_coefficient"]) / REF["drag_coefficient"]})
    rows.append(r)
    print(label, seed, {k: round(v, 5) for k, v in r.items() if isinstance(v, float)}, flush=True)

out = Path(__file__).parent
cols = sorted({k for r in rows for k in r})
with open(out / "dfg-rescore-per-checkpoint.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
summ = {}
for label in sorted({r["set"] for r in rows}):
    rs = [r for r in rows if r["set"] == label and r["status"] == "scored"]
    if not rs:
        continue
    summ[label] = {"n": len(rs)}
    for k in ("Cd_sigma_n", "Cd_relerr_sigma_n", "Cd_tangential_old", "Cd_relerr_tangential_old",
              "Cl_sigma_n", "Cl_abserr_sigma_n", "dp", "dp_relerr", "Cd_pressure_part", "Cd_viscous_part"):
        xs = [r[k] for r in rs]
        summ[label][k] = {"mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else None}
json.dump(summ, open(out / "dfg-rescore-summary.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
