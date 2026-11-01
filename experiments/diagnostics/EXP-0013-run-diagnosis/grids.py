"""EXP-0013 — mallas de evaluación de señales. Solo COORDENADAS (nunca valores de referencia).

Las señales se calculan sobre estas mallas; las etiquetas (errores) se calculan aparte, en `labels.py`, sobre los mismos puntos.
"""
from __future__ import annotations

import numpy as np

from common import BL, EX


def _tensor_grid(xs, ts):
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    return np.stack([xx.ravel(), tt.ravel()], axis=1)


def get_grid(problem: str) -> dict:
    """Devuelve {'points': (N,d), 'shape': (nx,nt) o None, 'axes': nombres, 'source': descripción}."""
    if problem == "heat_1d":
        xs, ts = np.linspace(0, 1, 257), np.linspace(0, 1, 129)
        return {"points": _tensor_grid(xs, ts), "shape": (257, 129), "axes": ["x", "t"], "source": "tensor uniforme 257x129 en [0,1]^2 (sin referencia)"}
    if problem == "burgers_1d":
        xs, ts = np.linspace(-1, 1, 1025)[::4], np.linspace(0, 1, 129)[::2]
        return {"points": _tensor_grid(xs, ts), "shape": (len(xs), len(ts)), "axes": ["x", "t"],
                "source": "submuestreo (x[::4], t[::2]) de la malla Cole-Hopf 1025x129 (solo coordenadas)"}
    if problem == "allen_cahn_1d":
        import scipy.io as sio

        mat = sio.loadmat(BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat")
        xs, ts = mat["x"].reshape(-1), mat["t"].reshape(-1)
        return {"points": _tensor_grid(xs, ts), "shape": (len(xs), len(ts)), "axes": ["x", "t"],
                "source": "coordenadas (x,t) de Allen_Cahn.mat 201x101 (solo coordenadas)"}
    if problem == "ns2d_cg":
        ref = np.loadtxt(BL / "ns2d-cg-20260930/reference/ns_0_obstacle.dat", comments="%", encoding="utf-8")
        return {"points": ref[:, :2].astype(np.float64), "shape": None, "axes": ["x", "y"], "source": "nodos COMSOL ns_0_obstacle.dat (solo coordenadas)"}
    if problem == "dfg_2d":
        z = np.load(BL / "ns2d-cg-20260930/reference/dfg_L3_fields.npz")
        return {"points": z["xy"].astype(np.float64), "shape": None, "axes": ["x", "y"], "source": "malla 221x42 del FEM L3 (solo coordenadas)"}
    if problem == "piezo_2d":
        z = np.load(EX / "piezo/EXP-0009-piezo-indirect-rbf/results/reference/fem-grid-indirect-polingminus.npz")
        return {"points": z["xy"].astype(np.float64), "shape": (201, 21), "axes": ["x", "y"], "source": "rejilla 201x21 del FEM de Daniel (solo coordenadas)"}
    raise KeyError(problem)
