"""py3.12 step: Daniel's scikit-fem solver (vendored skfem 12.0.2) -> FEM fields on a regular grid, saved to npz.
Uses the same torch-free stub trick as check_daniel.py. Output: fem-grid-<case>.npz"""
import sys, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "teoria-visual-v2" / "vendor"))
pkg = types.ModuleType("pinn_piezo"); pkg.__path__ = [str(HERE / "daniel-ref" / "src" / "pinn_piezo")]; sys.modules["pinn_piezo"] = pkg
cfg = types.ModuleType("pinn_piezo.config"); cfg.WIDTH = 100e-3; cfg.HEIGHT = 1e-3; cfg.H2 = 0.5e-3; cfg.CENTER = 0.5e-3; sys.modules["pinn_piezo.config"] = cfg
import numpy as np
from pinn_piezo import fem
xs = np.linspace(0, 0.1, 201); ys = np.linspace(0, 1e-3, 21)
X, Y = np.meshgrid(xs, ys, indexing="xy"); P = np.stack((X.ravel(), Y.ravel()), 1)
for sgn in (1.0, -1.0):
  for case, kw in (("indirect", dict(voltage=100.0)), ("direct", dict(force=0.1))):
      r = fem.solve_piezo(case, nx=400, ny=16, poling_sign=sgn, **kw)
      o = r.probe(P)
      np.savez(HERE / f"fem-grid-{case}{"" if sgn>0 else "-polingminus"}.npz", xy=P, u=o["u"], v=o["v"], phi=o["phi"], force=kw.get("force", np.nan))
      print(sgn, case, "v_max", np.abs(o["v"]).max(), "phi range", o["phi"].min(), o["phi"].max())
