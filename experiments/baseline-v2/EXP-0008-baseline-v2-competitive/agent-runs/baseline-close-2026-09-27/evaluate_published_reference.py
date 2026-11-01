from __future__ import annotations
import csv
import io
import json
import pathlib
import statistics
import zipfile
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator

ROOT = pathlib.Path("var/agent_runs/baseline-close-2026-09-27")
mat_path = ROOT / "references/Allen_Cahn.mat"
zip_path = pathlib.Path("var/agent_runs/allen-cahn-exp003/download/pikan-7d9fafd823f3e315/allen-cahn-t4x2-results.zip")
out = ROOT / "allen-cahn-published-reference"
out.mkdir(parents=True, exist_ok=True)

mat = loadmat(mat_path)
x_ref = np.asarray(mat["x"]).reshape(-1).astype(np.float64)
t_ref = np.asarray(mat["t"]).reshape(-1).astype(np.float64)
u_raw = np.asarray(mat["u"], dtype=np.float64)
if u_raw.shape == (len(t_ref), len(x_ref)):
    u_ref = u_raw
elif u_raw.shape == (len(x_ref), len(t_ref)):
    u_ref = u_raw.T
else:
    raise ValueError(f"Unexpected Allen_Cahn.mat array shape {u_raw.shape}")
interp = RegularGridInterpolator((t_ref, x_ref), u_ref, method="linear", bounds_error=True)

rows = []
with zipfile.ZipFile(zip_path) as zf:
    for seed in (101, 102, 103):
        rel = f"runs/seed-{seed}/allen_cahn/validation.npz"
        with np.load(io.BytesIO(zf.read(rel))) as item:
            x = np.asarray(item["x"], dtype=np.float64)
            t = np.asarray(item["t"], dtype=np.float64)
            pred = np.asarray(item["prediction"], dtype=np.float64)
            final_test_used = bool(item["final_test_used"].item())
        if pred.shape != (len(x), len(t)):
            raise ValueError(f"Unexpected prediction shape for seed {seed}: {pred.shape}")
        xx, tt = np.meshgrid(x, t, indexing="ij")
        points = np.column_stack((tt.ravel(), xx.ravel()))
        truth = interp(points).reshape(len(x), len(t))
        # Composite trapezoidal L2 quadrature over the full, uniformly sampled rectangle.
        wx = np.ones(len(x)); wx[[0, -1]] = 0.5
        wt = np.ones(len(t)); wt[[0, -1]] = 0.5
        weights = wx[:, None] * wt[None, :]
        dx = (x[-1] - x[0]) / (len(x) - 1)
        dt = (t[-1] - t[0]) / (len(t) - 1)
        norm_error = np.sqrt(np.sum(weights * (pred - truth) ** 2) * dx * dt)
        norm_truth = np.sqrt(np.sum(weights * truth ** 2) * dx * dt)
        rows.append({
            "problem": "allen_cahn_1d",
            "seed": seed,
            "reference": "DeepXDE published Allen_Cahn.mat; provisional, provenance unspecified",
            "reference_grid_nx": len(x_ref),
            "reference_grid_nt": len(t_ref),
            "prediction_grid_nx": len(x),
            "prediction_grid_nt": len(t),
            "relative_l2_trapezoidal": float(norm_error / norm_truth),
            "max_abs_error": float(np.max(np.abs(pred - truth))),
            "rmse": float(np.sqrt(np.sum(weights * (pred - truth) ** 2) / np.sum(weights))),
            "final_test_used_in_training": final_test_used,
            "scientific_accepted": False,
        })
summary = {
    "metric": "relative_l2_trapezoidal",
    "n": len(rows),
    "mean": statistics.mean(r["relative_l2_trapezoidal"] for r in rows),
    "sample_sd": statistics.stdev(r["relative_l2_trapezoidal"] for r in rows),
    "max_abs_error_mean": statistics.mean(r["max_abs_error"] for r in rows),
    "reference_status": "published_grid_provisional; generating solver and tolerances not documented in source",
    "field_metric_unlocked_in_eval_harness": False,
    "scientific_accepted": False,
    "runs": rows,
}
with (out / "seed-results.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
try:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(r["seed"]) for r in rows],
                         y=[r["relative_l2_trapezoidal"] for r in rows],
                         name="RBF-PIKAN (EXP-0003)"))
    fig.update_yaxes(type="log", title="Relative L2 error (trapezoidal)")
    fig.update_layout(title="Allen–Cahn: RBF-PIKAN vs DeepXDE published grid",
                      xaxis_title="Seed", template="plotly_white")
    fig.write_html(out / "relative-l2-by-seed.html", include_plotlyjs="cdn")
except Exception as exc:
    (out / "plot-error.txt").write_text(type(exc).__name__ + ": " + str(exc), encoding="utf-8")
print(json.dumps(summary, indent=2))
