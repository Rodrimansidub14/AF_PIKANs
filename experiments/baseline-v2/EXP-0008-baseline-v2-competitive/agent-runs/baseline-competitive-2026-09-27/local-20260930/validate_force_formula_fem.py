"""Step 2a (local, CPU, no training): validate the FEATFLOW force formula in src/research_pipeline/sciml/dfg.py
(`qoi_from_cylinder`) on the archived Taylor-Hood P2/P1 FEM field DFG-L3.

No dolfinx here, so the isoparametric P2 velocity / P1 pressure fields are evaluated directly from
mesh.h5 + fem_coefficients.npz at the cylinder contour points. Expected: Cd~5.5794, Cl~0.010619,
dp~0.11750 (S1b's own dolfinx force integrals; FEATFLOW 5.57953523, 0.010618948, 0.11752017).
Also reports the *tangential* formula used by the older 501-505 scorer for comparison.
Run: py -3.13 validate_force_formula_fem.py
"""
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "src"))
from research_pipeline.sciml import dfg  # noqa: E402

L3 = ROOT / "kaggle-outputs/dfg-fem-field-followup-retry5-20260929/dfg-fem-field-followup-retry5/dfg/DFG-L3"
with h5py.File(L3 / "mesh.h5", "r") as f:
    X = f["Mesh/mesh/geometry"][:]
    T = f["Mesh/mesh/topology"][:]
co = np.load(L3 / "fem_coefficients.npz")
vel_xy = co["velocity_dof_coordinates"][:, :2]
vel = co["velocity_coefficients"].reshape(-1, 2)  # blocked (u,v) per dof
p_xy = co["pressure_dof_coordinates"][:, :2]
p_val = co["pressure_coefficients"]

# node ordering check: dofmap = [v0,v1,v2, mid(1,2), mid(0,2), mid(0,1)]
P = X[T]  # (nc,6,2)
mid12 = 0.5 * (P[:, 1] + P[:, 2])
straight = np.abs(P[:, 3] - mid12).max()
print("max |node3 - mid(v1,v2)| (should be ~ curvature only):", straight)

vtree = cKDTree(vel_xy)
dv, vi = vtree.query(X)
print("geometry->velocity dof match max dist:", dv.max())
vel_at_geom = vel[vi]  # (nn,2)
ptree = cKDTree(p_xy)
dp_, pi_ = ptree.query(X[T[:, :3]].reshape(-1, 2))
print("vertex->pressure dof match max dist:", dp_.max())
p_at_vert = p_val[pi_].reshape(-1, 3)  # per cell, vertices 0..2

# no-slip / inlet sanity on the field
cyl = np.abs(np.hypot(vel_xy[:, 0] - 0.2, vel_xy[:, 1] - 0.2) - 0.05) < 1e-6
print("|u| on cylinder dofs max:", np.abs(vel[cyl]).max(), "n=", cyl.sum())


def shape(xi, eta):
    l0, l1, l2 = 1 - xi - eta, xi, eta
    N = np.array([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l1 * l2, 4 * l0 * l2, 4 * l0 * l1])
    dN = np.array([  # d/dxi, d/deta
        [-(4 * l0 - 1), -(4 * l0 - 1)],
        [4 * l1 - 1, 0.0],
        [0.0, 4 * l2 - 1],
        [4 * l2, 4 * l1],
        [-4 * l2, 4 * (l0 - l2)],
        [4 * (l0 - l1), -4 * l1],
    ])
    return N, dN


def locate(pt):
    """Return (cell, xi, eta) for a point; Newton on the isoparametric map among candidate cells."""
    lo = P[:, :3].min(1) - 1e-3
    hi = P[:, :3].max(1) + 1e-3
    cand = np.where((pt >= lo).all(1) & (pt <= hi).all(1))[0]
    best = None
    for c in cand:
        xi = np.array([1 / 3, 1 / 3])
        for _ in range(30):
            N, dN = shape(*xi)
            r = N @ P[c] - pt
            J = (dN.T @ P[c]).T  # (2,2): dx/dxi columns
            step = np.linalg.solve(J, r)
            xi = xi - step
            if np.abs(step).max() < 1e-14:
                break
        inside = min(xi[0], xi[1], 1 - xi[0] - xi[1])
        if best is None or inside > best[0]:
            best = (inside, c, xi.copy())
        if inside > -1e-9:
            break
    return best[1], best[2], best[0]


def eval_at(pt):
    c, xi, inside = locate(pt)
    N, dN = shape(*xi)
    J = (dN.T @ P[c]).T
    grad_ref = dN  # (6,2)
    gphys = np.linalg.solve(J.T, grad_ref.T).T  # (6,2) d/dx,d/dy of each shape fn
    vv = vel_at_geom[T[c]]  # (6,2)
    u_val = N @ vv
    grad = gphys.T @ vv  # (2 dims, 2 comps): [d/dx, d/dy] x [u,v]
    l = np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
    p = l @ p_at_vert[c]
    return p, grad[0, 0], grad[1, 0], grad[0, 1], grad[1, 1], u_val, inside


def contour(n):
    theta = 2.0 * math.pi * np.arange(n) / n
    pts = np.column_stack((0.2 + 0.05 * np.cos(theta), 0.2 + 0.05 * np.sin(theta)))
    out = np.array([eval_at(pt)[:5] for pt in pts])
    ins = min(eval_at(pt)[6] for pt in pts[:: max(1, n // 16)])
    return theta, out, ins


def tangential_formula(theta, p, ux, uy, vx, vy, nu=dfg.NU, u_mean=dfg.U_MEAN, d=dfg.DIAMETER):
    """Formula from the older 501-505 scorer (run_campaign_source.qoi_for_model)."""
    nx, ny = np.cos(theta), np.sin(theta)
    tx, ty = -ny, nx
    ndu = ux * nx + uy * ny
    ndv = vx * nx + vy * ny
    dut = tx * ndu + ty * ndv
    ds = 2 * math.pi * 0.05 / len(theta)
    fd = float(np.sum(nu * dut * ny - p * nx) * ds)
    fl = float(-np.sum(nu * dut * nx + p * ny) * ds)
    s = 2.0 / (u_mean**2 * d)
    return s * fd, s * fl


res = {}
front = eval_at(np.array([0.15, 0.2]))[0]
rear = eval_at(np.array([0.25, 0.2]))[0]
for n in (512, 2048):
    theta, o, ins = contour(n)
    p, ux, uy, vx, vy = o.T
    q = dfg.qoi_from_cylinder(theta, p, ux, uy, vx, vy, front, rear)
    cd_t, cl_t = tangential_formula(theta, p, ux, uy, vx, vy)
    ref = dfg.FEATFLOW_REFERENCE
    res[str(n)] = {
        "min_inside_ref_coord": ins,
        "sigma_n_dfg_py": {k: q[k] for k in ("drag_coefficient", "lift_coefficient", "pressure_difference")},
        "tangential_older_scorer": {"drag_coefficient": cd_t, "lift_coefficient": cl_t},
        "rel_err_sigma_n_vs_featflow": {
            "Cd": abs(q["drag_coefficient"] - ref["drag_coefficient"]) / ref["drag_coefficient"],
            "Cl_abs": abs(q["lift_coefficient"] - ref["lift_coefficient"]),
            "dp": abs(q["pressure_difference"] - ref["pressure_difference"]) / ref["pressure_difference"],
        },
        "rel_err_tangential_Cd_vs_featflow": abs(cd_t - ref["drag_coefficient"]) / ref["drag_coefficient"],
    }
res["s1b_dolfinx_own_qois"] = json.load(open(L3 / "qois.json"))["Cd"], json.load(open(L3 / "qois.json"))["Cl"]
json.dump(res, open(Path(__file__).parent / "force-formula-fem-validation.json", "w"), indent=2)
print(json.dumps(res, indent=2))
