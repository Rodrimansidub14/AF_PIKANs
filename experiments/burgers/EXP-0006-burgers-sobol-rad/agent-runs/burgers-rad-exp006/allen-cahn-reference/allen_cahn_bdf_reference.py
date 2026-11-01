#!/usr/bin/env python
"""CPU-only candidate Allen-Cahn reference: centered FD in space, SciPy BDF in time.

Outputs from this script are exploratory artifacts, never a canonical verifier
reference. The script does not import or edit evals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

# Keep this small CPU reference reproducible and avoid oversized BLAS pools.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.sparse import diags

D = 0.001
K = 5.0
LEFT = RIGHT = -1.0
X_BOUNDS = (-1.0, 1.0)
T_BOUNDS = (0.0, 1.0)
EVENT_TIMES = np.array([0.25, 0.5, 0.75, 1.0])
FIELD_LIMIT = 0.005
QOI_LIMIT = 0.001
HERE = Path(__file__).resolve().parent


def initial(x):
    x = np.asarray(x, dtype=np.float64)
    return x*x*np.cos(np.pi*x)


def laplacian_dirichlet(y, h, left=LEFT, right=RIGHT):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(h) or h <= 0:
        raise ValueError("expected nonempty vector and positive finite spacing")
    lap = -2.0*y.copy()
    lap[1:] += y[:-1]
    lap[:-1] += y[1:]
    lap[0] += left
    lap[-1] += right
    return lap/(h*h)


def rhs(u, uxx):
    u = np.asarray(u, dtype=np.float64)
    uxx = np.asarray(uxx, dtype=np.float64)
    if u.shape != uxx.shape:
        raise ValueError("u and uxx shapes differ")
    return D*uxx + K*(u-u**3)


def residual(u, ut, uxx):
    u = np.asarray(u, dtype=np.float64)
    ut = np.asarray(ut, dtype=np.float64)
    uxx = np.asarray(uxx, dtype=np.float64)
    if u.shape != ut.shape or u.shape != uxx.shape:
        raise ValueError("u, ut and uxx shapes differ")
    return ut-rhs(u, uxx)


def operator_self_check():
    x = np.linspace(-1.0, 1.0, 9)
    q = x*x
    lap = laplacian_dirichlet(q[1:-1], x[1]-x[0], q[0], q[-1])
    center = float(lap[3])
    diffusion = D*center
    reaction = K*(0.25-0.25**3)
    manufactured = float(residual([0.2], [1.0], [2.0])[0])
    ok = (abs(center-2.0) < 1e-12 and abs(diffusion-0.002) < 1e-14
          and abs(reaction-1.171875) < 1e-14
          and abs(manufactured-0.038) < 1e-12)
    return {
        "quadratic_laplacian_center": center,
        "diffusion_rhs_for_quadratic": float(diffusion),
        "reaction_rhs_at_quarter": float(reaction),
        "manufactured_residual_at_u_0_2": manufactured,
        "operator_signs_ok": bool(ok),
    }


def energy(u, x):
    h = float(x[1]-x[0])
    ux = np.empty_like(u)
    ux[1:-1] = (u[2:]-u[:-2])/(2*h)
    ux[0] = (-3*u[0]+4*u[1]-u[2])/(2*h)
    ux[-1] = (3*u[-1]-4*u[-2]+u[-3])/(2*h)
    density = 0.5*D*ux**2 + 1.25*(1-u**2)**2
    return float(np.trapezoid(density, x))


def relative_l2(a, b, x, t):
    if a.shape != b.shape or a.shape != (len(x), len(t)):
        raise ValueError("field shapes do not match common grid")
    wx = np.ones(len(x)); wt = np.ones(len(t))
    wx[[0, -1]] = 0.5; wt[[0, -1]] = 0.5
    wx *= (x[-1]-x[0])/(len(x)-1)
    wt *= (t[-1]-t[0])/(len(t)-1)
    w = np.multiply.outer(wx, wt)
    den = float(np.sum(w*b*b))
    num = float(np.sum(w*(a-b)**2))
    if den <= 0 or not np.isfinite(den+num):
        raise ValueError("invalid weighted field norm")
    return float(np.sqrt(num/den))


def energy_change(a, b):
    return float(np.max(np.abs(a-b)/np.maximum(np.abs(b), 1e-14)))


def project(x_native, values, x_grid):
    out = np.empty((len(x_grid), values.shape[1]))
    for j in range(values.shape[1]):
        out[:, j] = np.interp(x_grid, x_native, values[:, j])
    return out


def solve_grid(n, x_grid, t_grid, event_times, rtol, atol, label):
    if n < 5 or n % 2 != 1:
        raise ValueError("node count must be odd and >=5")
    x = np.linspace(*X_BOUNDS, n)
    h = float(x[1]-x[0])
    m = n-2
    off = D/(h*h)
    pattern = diags(
        [np.ones(m-1), np.ones(m), np.ones(m-1)],
        [-1, 0, 1], shape=(m, m), format="csc")

    def fun(_t, y):
        return rhs(y, laplacian_dirichlet(y, h))

    def jac(_t, y):
        diag = -2*off + K*(1-3*y*y)
        return diags(
            [np.full(m-1, off), diag, np.full(m-1, off)],
            [-1, 0, 1], shape=(m, m), format="csc")

    t_eval = np.unique(np.concatenate((t_grid, event_times)))
    start = time.perf_counter()
    sol = solve_ivp(
        fun, T_BOUNDS, initial(x[1:-1]), method="BDF", t_eval=t_eval,
        rtol=rtol, atol=atol, jac=jac, jac_sparsity=pattern)
    elapsed = time.perf_counter()-start
    if not sol.success:
        raise RuntimeError(f"{label}: BDF failed: {sol.message}")
    if sol.y.shape != (m, len(t_eval)) or not np.isfinite(sol.y).all():
        raise FloatingPointError(f"{label}: bad BDF output shape or non-finite state")
    full = np.empty((n, len(t_eval)))
    full[0, :] = LEFT
    full[-1, :] = RIGHT
    full[1:-1, :] = sol.y
    ir = np.searchsorted(t_eval, t_grid)
    ie = np.searchsorted(t_eval, event_times)
    native_report = full[:, ir]
    native_events = full[:, ie]
    common = project(x, native_report, x_grid)
    common_events = project(x, native_events, x_grid)
    energies = np.array([energy(native_report[:, j], x) for j in range(len(t_grid))])
    event_energy = np.array([energy(native_events[:, j], x) for j in range(len(event_times))])
    max_increase = max(0.0, float(np.max(np.diff(energies))))
    energy_tol = 1e-8*max(1.0, float(np.max(np.abs(energies))))
    i0 = int(np.searchsorted(t_eval, 0.0))
    return {
        "label": label, "node_count": n, "interior_count": m, "spacing": h,
        "rtol": rtol, "atol": atol, "success": bool(sol.success),
        "message": str(sol.message), "elapsed_sec": elapsed,
        "nfev": int(sol.nfev), "njev": int(sol.njev), "nlu": int(sol.nlu),
        "initial_max_abs_error": float(np.max(np.abs(full[:, i0]-initial(x)))),
        "boundary_max_abs_error": float(max(np.max(np.abs(full[0, :]-LEFT)),
                                             np.max(np.abs(full[-1, :]-RIGHT)))),
        "all_values_finite": bool(np.isfinite(common).all()),
        "energy_monotone_within_tolerance": bool(max_increase <= energy_tol),
        "max_energy_increase_abs": max_increase,
        "energy_monotonicity_tolerance": energy_tol,
        "energy_events": event_energy, "energies": energies,
        "field": common, "field_events": common_events,
    }


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_run(run):
    return {k: v for k, v in run.items()
            if k not in {"energy_events", "energies", "field", "field_events"}}


def run_study(out, smoke=False):
    if smoke:
        nodes, nx, nt = [65], 65, 33
        rtol, atol, do_time = 1e-6, 1e-8, False
    else:
        nodes, nx, nt = [513, 1025, 2049], 512, 256
        rtol, atol, do_time = 1e-8, 1e-10, True
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    xg = np.linspace(*X_BOUNDS, nx)
    tg = np.linspace(*T_BOUNDS, nt)
    runs = []
    for n in nodes:
        label = f"space_{n}_rtol_{rtol:g}_atol_{atol:g}"
        runs.append(solve_grid(n, xg, tg, EVENT_TIMES, rtol, atol, label))
    tight = None
    if do_time:
        tight = solve_grid(nodes[-1], xg, tg, EVENT_TIMES, 1e-10, 1e-12,
                           f"space_{nodes[-1]}_rtol_1e-10_atol_1e-12")
    all_runs = runs + ([tight] if tight else [])
    changes = []
    for a, b in zip(runs, runs[1:]):
        changes.append({
            "coarse": a["label"], "fine": b["label"],
            "weighted_relative_l2_field_change": relative_l2(a["field"], b["field"], xg, tg),
            "max_relative_energy_change": energy_change(a["energy_events"], b["energy_events"]),
        })
    time_delta = None
    if tight:
        a, b = runs[-1], tight
        time_delta = {
            "baseline": a["label"], "tight": b["label"],
            "weighted_relative_l2_field_change": relative_l2(a["field"], b["field"], xg, tg),
            "max_relative_energy_change": energy_change(a["energy_events"], b["energy_events"]),
        }
    last_field = changes[-1]["weighted_relative_l2_field_change"] if changes else None
    last_qoi = changes[-1]["max_relative_energy_change"] if changes else None
    ratio = None
    if len(changes) >= 2 and changes[-2]["weighted_relative_l2_field_change"] > 0:
        ratio = changes[-1]["weighted_relative_l2_field_change"] / changes[-2]["weighted_relative_l2_field_change"]
    checks = operator_self_check()
    checks.update({
        "all_solvers_succeeded": all(r["success"] for r in all_runs),
        "all_values_finite": all(r["all_values_finite"] for r in all_runs),
        "max_boundary_abs_error": max(r["boundary_max_abs_error"] for r in all_runs),
        "max_initial_abs_error": max(r["initial_max_abs_error"] for r in all_runs),
        "energy_monotone_within_tolerance": all(r["energy_monotone_within_tolerance"] for r in all_runs),
    })
    if not checks["operator_signs_ok"] or not checks["all_solvers_succeeded"] or not checks["all_values_finite"]:
        raise RuntimeError("candidate reference sanity check failed")
    primary = tight if tight else runs[-1]
    src = Path(__file__).resolve()
    report = {
        "study_status": "candidate_only",
        "canonical_reference": False,
        "scientific_acceptance": False,
        "verifier_unlocked": False,
        "contract": {
            "pde": "u_t - 0.001*u_xx - 5*(u-u^3) = 0",
            "x_bounds": list(X_BOUNDS), "t_bounds": list(T_BOUNDS),
            "initial_condition": "x^2*cos(pi*x)",
            "dirichlet_boundary": -1.0,
            "space_method": "second-order centered finite differences",
            "time_method": "SciPy solve_ivp method=BDF with sparse analytic Jacobian",
            "report_grid": {"nx": nx, "nt": nt, "include_boundary": True},
            "evaluation_times": EVENT_TIMES.tolist(),
            "projection": "linear interpolation in x onto the fixed report grid",
        },
        "convergence_contract": {
            "declared_minimum_resolutions": 3,
            "spatial_node_counts": nodes,
            "base_temporal_tolerances": {"rtol": rtol, "atol": atol},
            "tight_temporal_tolerances": ({"rtol": 1e-10, "atol": 1e-12} if do_time else None),
            "field_change_metric": "trapezoid-weighted relative L2 between consecutive fields on report grid",
            "field_change_limit": FIELD_LIMIT,
            "last_spatial_field_change": last_field,
            "last_spatial_field_threshold_met": bool(last_field is not None and last_field < FIELD_LIMIT),
            "qoi_change_metric": "maximum relative change in energy at four declared evaluation times",
            "qoi_change_limit": QOI_LIMIT,
            "last_spatial_energy_change": last_qoi,
            "last_spatial_energy_threshold_met": bool(last_qoi is not None and last_qoi < QOI_LIMIT),
            "temporal_tolerance_difference": time_delta,
            "subsequent_change_fraction": {
                "declared_fraction": 0.1,
                "observed_successive_field_change_ratio": ratio,
                "decision": "unresolved: problem.yaml does not define effect or comparison",
            },
            "full_reference_convergence_claim": False,
            "reason_not_to_unlock": "candidate artifact; subsequent-change criterion unresolved and human verifier review required",
        },
        "sanity_checks": checks,
        "runs": [clean_run(r) for r in runs],
        "spatial_refinement_differences": changes,
        "energy_by_run_at_evaluation_times": {
            r["label"]: [float(v) for v in r["energy_events"]] for r in all_runs
        },
        "primary_candidate_field": primary["label"],
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "thread_limits": {k: os.environ[k] for k in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")},
            "source_script": src.name, "source_script_sha256": digest(src),
        },
    }
    field_arrays = {f"u_common_n{r['node_count']}": r["field"] for r in runs}
    if tight:
        field_arrays["u_common_n2049_tight_time"] = tight["field"]
    npz = out/"allen_cahn_candidate_reference.npz"
    np.savez_compressed(
        npz, x=xg, t=tg, event_times=EVENT_TIMES,
        u_primary=primary["field"], u_primary_at_evaluation_times=primary["field_events"],
        **field_arrays)
    report["provenance"]["candidate_npz"] = npz.name
    report["provenance"]["candidate_npz_sha256"] = digest(npz)
    (out/"metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    (out/"study_report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(r):
    c, s = r["convergence_contract"], r["sanity_checks"]
    lines = [
        "# Allen-Cahn CPU reference study (candidate artifact)", "",
        "Candidate only: not canonical, not scientifically accepted, and does not unlock evals.", "",
        "## Contract and solver", "",
        f"- PDE: {r['contract']['pde']}",
        f"- Spatial nodes: {c['spatial_node_counts']}",
        f"- Base BDF tolerances: {c['base_temporal_tolerances']}",
        f"- Tight BDF tolerances: {c['tight_temporal_tolerances']}",
        f"- Common grid: {r['contract']['report_grid']}",
        "",
        "## Sanity checks", "",
        f"- Operator sign checks: {s['operator_signs_ok']}",
        f"- All solves succeeded / values finite: {s['all_solvers_succeeded']} / {s['all_values_finite']}",
        f"- Max boundary / initial errors: {s['max_boundary_abs_error']:.3e} / {s['max_initial_abs_error']:.3e}",
        f"- Energy monotone within tolerance: {s['energy_monotone_within_tolerance']}",
        "",
        "## Refinement differences", "",
        "| Comparison | Weighted relative field change | Maximum relative energy change |",
        "|---|---:|---:|",
    ]
    for d in r["spatial_refinement_differences"]:
        lines.append(f"| {d['coarse']} -> {d['fine']} | {d['weighted_relative_l2_field_change']:.6e} | {d['max_relative_energy_change']:.6e} |")
    td = c["temporal_tolerance_difference"]
    if td:
        lines.append(f"| {td['baseline']} -> tighter BDF | {td['weighted_relative_l2_field_change']:.6e} | {td['max_relative_energy_change']:.6e} |")
    lines.extend([
        "",
        f"- Last spatial field limit (<{FIELD_LIMIT}): {c['last_spatial_field_threshold_met']}",
        f"- Last spatial energy limit (<{QOI_LIMIT}): {c['last_spatial_energy_threshold_met']}",
        f"- Subsequent-change criterion: {c['subsequent_change_fraction']['decision']}",
        "",
        "The configured phrase 'subsequent change below 10% of effect' has no",
        "operational definition of effect in problem.yaml. The observed ratio is",
        "recorded, but this study does not claim that criterion passed.",
        "",
        "## Provenance", "",
        f"- Python {r['provenance']['python'].split()[0]}, NumPy {r['provenance']['numpy']}, SciPy {r['provenance']['scipy']}",
        f"- Generated UTC: {r['provenance']['generated_at_utc']}",
        f"- Script SHA-256: {r['provenance']['source_script_sha256']}",
        f"- NPZ SHA-256: {r['provenance']['candidate_npz_sha256']}", "",
    ])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE/"study")
    p.add_argument("--smoke", action="store_true", help="one small CPU solve")
    p.add_argument("--self-check", action="store_true", help="analytic operator checks only")
    a = p.parse_args()
    if a.self_check:
        checks = operator_self_check()
        print(json.dumps(checks, indent=2, sort_keys=True, allow_nan=False))
        return 0 if checks["operator_signs_ok"] else 2
    r = run_study(a.output, smoke=a.smoke)
    print(json.dumps({
        "study_status": r["study_status"],
        "output": str(a.output.resolve()),
        "all_solvers_succeeded": r["sanity_checks"]["all_solvers_succeeded"],
        "all_values_finite": r["sanity_checks"]["all_values_finite"],
        "energy_monotone_within_tolerance": r["sanity_checks"]["energy_monotone_within_tolerance"],
        "last_spatial_field_threshold_met": r["convergence_contract"]["last_spatial_field_threshold_met"],
        "last_spatial_energy_threshold_met": r["convergence_contract"]["last_spatial_energy_threshold_met"],
        "reference_unlocked": False,
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
