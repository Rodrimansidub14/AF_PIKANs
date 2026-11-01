"""EXP-0013 — ETIQUETAS (resultados): errores contra la referencia, por región, y auditoría de alineación residuo–error.

Este es el ÚNICO módulo del estudio que lee valores de referencia. `signals.py` no lo importa. Las etiquetas solo se usan como variable a explicar.
Referencias: analítica (Heat), Cole-Hopf (Burgers), Allen_Cahn.mat (aceptado 2026-10-01), COMSOL (NS2d-CG), FEM L3 (DFG), FEM de Daniel poling_sign=-1 (piezo).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

from common import BL, EX
from signals import entropy_norm, gini, spearman, top_mass

HEAT_ALPHA = 0.1
_CACHE: dict = {}


def reference(problem: str, pts: np.ndarray) -> dict:
    """Devuelve {'comps': {nombre: (N,)}, 'primary': (N,) campo para escalas de estructura}."""
    if problem in _CACHE:
        return _CACHE[problem]
    if problem == "heat_1d":
        x, t = pts[:, 0], pts[:, 1]
        u = np.sin(math.pi * x) * np.exp(-HEAT_ALPHA * math.pi ** 2 * t)
        out = {"comps": {"u": u}, "primary": u}
    elif problem == "burgers_1d":
        z = np.load(BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz")
        U = z["u"][::2, ::4].T          # (nt,nx) -> submuestreo -> (nx,nt)
        out = {"comps": {"u": U.reshape(-1)}, "primary": U.reshape(-1)}
    elif problem == "allen_cahn_1d":
        m = sio.loadmat(BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat")
        U = m["u"].T                    # (nt,nx) -> (nx,nt)
        out = {"comps": {"u": U.reshape(-1)}, "primary": U.reshape(-1)}
    elif problem == "ns2d_cg":
        ref = np.loadtxt(BL / "ns2d-cg-20260930/reference/ns_0_obstacle.dat", comments="%", encoding="utf-8")
        out = {"comps": {"u": ref[:, 2], "v": ref[:, 3], "p": ref[:, 4]}, "primary": np.hypot(ref[:, 2], ref[:, 3])}
    elif problem == "dfg_2d":
        z = np.load(BL / "ns2d-cg-20260930/reference/dfg_L3_fields.npz")
        vel = z["velocity"]
        out = {"comps": {"u": vel[:, 0], "v": vel[:, 1], "p": z["pressure_gauge_fixed"]}, "primary": np.hypot(vel[:, 0], vel[:, 1])}
    elif problem == "piezo_2d":
        z = np.load(EX / "piezo/EXP-0009-piezo-indirect-rbf/results/reference/fem-grid-indirect-polingminus.npz")
        out = {"comps": {"u": z["u"], "v": z["v"], "phi": z["phi"]}, "primary": z["v"]}
    else:
        raise KeyError(problem)
    _CACHE[problem] = out
    return out


def model_components(problem: str, fields: np.ndarray, anchor_p: float | None = None) -> dict:
    if problem in ("heat_1d", "burgers_1d", "allen_cahn_1d"):
        return {"u": fields[:, 0]}
    if problem == "ns2d_cg":
        return {"u": fields[:, 0], "v": fields[:, 1], "p": fields[:, 2]}
    if problem == "dfg_2d":
        return {"u": fields[:, 0], "v": fields[:, 1], "p": fields[:, 2] - (anchor_p or 0.0)}
    if problem == "piezo_2d":
        return {"u": fields[:, 0], "v": fields[:, 1], "phi": fields[:, 2]}
    raise KeyError(problem)


def knn_gradient(pts: np.ndarray, f: np.ndarray, k: int = 6) -> np.ndarray:
    """Magnitud local del gradiente por vecinos más cercanos (sirve en mallas regulares y no estructuradas)."""
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=k + 1)
    dd = d[:, 1:]
    df = np.abs(f[idx[:, 1:]] - f[:, None])
    # normaliza cada dimensión por su extensión para que x e y/t sean comparables en mallas anisótropas
    return (df / np.maximum(dd, 1e-300)).max(axis=1)


def regions(problem: str, pts: np.ndarray, primary: np.ndarray) -> dict:
    scale = pts.max(axis=0) - pts.min(axis=0)
    g = knn_gradient(pts / scale, primary)
    steep = g >= np.percentile(g, 95)
    out = {"steep5": steep}
    x = pts[:, 0]
    y = pts[:, 1]
    if problem == "allen_cahn_1d":
        out["interface"] = np.abs(primary) < 0.9
    if problem == "burgers_1d":
        out["late_time"] = y >= 1.0 / math.pi
    if problem == "heat_1d":
        for q in range(4):
            lo, hi = q / 4, (q + 1) / 4
            out[f"t_q{q+1}"] = (y >= lo) & (y <= hi if q == 3 else y < hi)
    if problem == "ns2d_cg":
        out["near_corner"] = np.hypot(x - 2.0, y - 1.0) < 0.5
    if problem == "dfg_2d":
        out["near_cyl"] = np.hypot(x - 0.2, y - 0.2) < 0.15
        out["wake"] = (x > 0.3) & (x < 1.0) & (np.abs(y - 0.2) < 0.1)
    if problem == "piezo_2d":
        out["layer_bottom"] = y < 0.5e-3
        out["layer_top"] = y >= 0.5e-3
        out["near_clamp"] = x < 0.01
        out["near_tip"] = x > 0.09
    return out


def label_row(problem: str, pts: np.ndarray, fields: np.ndarray, resid_all: np.ndarray, anchor_p: float | None) -> dict:
    ref = reference(problem, pts)
    mc = model_components(problem, fields, anchor_p)
    comps = ref["comps"]
    out = {}
    e2 = np.zeros(len(pts))
    flat_num = flat_den = 0.0
    for c, r in comps.items():
        e = mc[c] - r
        nr = float(np.linalg.norm(r))
        out[f"err/{c}_rel_l2"] = float(np.linalg.norm(e) / nr) if nr > 0 else float("nan")
        e2 += (e / (np.sqrt(np.mean(r ** 2)) + 1e-300)) ** 2
        flat_num += float(np.sum(e ** 2))
        flat_den += float(np.sum(r ** 2))
    eall = np.sqrt(e2)
    out["err/flat_rel_l2"] = math.sqrt(flat_num / flat_den) if flat_den > 0 else float("nan")
    out["err/all_rms_normalized"] = float(np.sqrt((eall ** 2).mean()))
    sq = eall ** 2
    out["err/gini"] = gini(eall)
    out["err/top1_mass"] = top_mass(sq, 0.01)
    out["err/top5_mass"] = top_mass(sq, 0.05)
    out["err/entropy_norm"] = entropy_norm(sq)
    regs = regions(problem, pts, ref["primary"])
    n = len(pts)
    for name, mask in regs.items():
        frac_pts = float(mask.mean())
        out[f"region/{name}/point_fraction"] = frac_pts
        out[f"region/{name}/err_mass"] = float(sq[mask].sum() / sq.sum()) if sq.sum() > 0 else float("nan")
        out[f"region/{name}/err_concentration"] = out[f"region/{name}/err_mass"] / frac_pts if frac_pts > 0 else float("nan")
        out[f"region/{name}/res_mass"] = float((resid_all[mask] ** 2).sum() / (resid_all ** 2).sum()) if (resid_all ** 2).sum() > 0 else float("nan")
        out[f"region/{name}/res_concentration"] = out[f"region/{name}/res_mass"] / frac_pts if frac_pts > 0 else float("nan")
    # --- auditoría de alineación residuo–error (usa etiquetas: NO es una señal)
    out["audit/spearman_res_err"] = spearman(resid_all, eall)
    k = max(1, n // 20)
    top_e = np.argsort(eall)[-k:]
    top_r = np.argsort(resid_all)[-k:]
    out["audit/top5_overlap"] = float(len(np.intersect1d(top_e, top_r)) / k)
    out["audit/err_mass_in_top5res"] = float(sq[top_r].sum() / sq.sum()) if sq.sum() > 0 else float("nan")
    out["audit/res_mass_in_top5err"] = float((resid_all[top_e] ** 2).sum() / (resid_all ** 2).sum()) if (resid_all ** 2).sum() > 0 else float("nan")
    out["audit/spearman_res_err_in_steep5"] = spearman(resid_all[regs["steep5"]], eall[regs["steep5"]])
    return out, eall
