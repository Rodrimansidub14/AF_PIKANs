"""Genera la referencia FEM del paper (scikit-fem P2, malla 200x8, 20 451 DOF) sobre la rejilla 201x21 para los DOS problemas, con el solver del repo de
Gonzalez (PINNs_piezoelectricity @ e7856f1, solo se importa; no se modifica):  reference/fem_converse.npz (100 V) y reference/fem_direct.npz (0.1 N hacia abajo,
electrodo superior flotante con carga neta cero).  poling_sign = -1 (CANONICAL_POLING_SIGN del repo: +e en la capa inferior).  Requiere scikit-fem 12.0.2 (no se necesita en el bundle).
Uso: python make_fem_reference.py [--repo RUTA_DEL_CLON]"""
import argparse, os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=str(ROOT / "var/agent_runs/piezo-ref-20260930/daniel-ref-20261003"))
a = ap.parse_args()
os.environ.setdefault("PINN_PIEZO_OUTPUTS_DIR", str(HERE / "_tmp_outputs"))
sys.path[:0] = [str(Path(a.repo) / "src")]
import numpy as np
from pinn_piezo import fem
xs, ys = np.linspace(0, 0.1, 201), np.linspace(0, 1e-3, 21)
X, Y = np.meshgrid(xs, ys, indexing="xy")
P = np.stack((X.ravel(), Y.ravel()), 1)
out = {}
for name, kw in (("converse", dict(case="indirect", voltage=100.0)), ("direct", dict(case="direct", force=-0.1))):
    r = fem.solve_piezo(nx=200, ny=8, poling_sign=-1.0, **kw)
    o = r.probe(P)
    np.savez(HERE / "reference" / f"fem_{name}.npz", xy=P, u=o["u"], v=o["v"], phi=o["phi"])
    tip = int(np.argmin(np.abs(P[:, 0] - 0.1) + P[:, 1]))
    top = P[:, 1] == 1e-3
    out[name] = dict(tip_v_um=float(o["v"][tip]) * 1e6, phi_top_mean=float(o["phi"][top].mean()), phi_top_spread=float(np.ptp(o["phi"][top])), n_points=int(P.shape[0]))
    print(name, out[name], flush=True)
(HERE / "reference" / "fem_reference.json").write_text(json.dumps(out, indent=1))
