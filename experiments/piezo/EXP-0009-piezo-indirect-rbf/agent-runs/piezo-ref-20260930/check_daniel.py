"""Read-only checks on Daniel's repo (cloned in ./daniel-ref, not modified): (1) material coefficients vs the
ProyectoGrad P5 table, (2) run his scikit-fem indirect+direct solver on CPU with the vendored skfem 12.0.2.
Outputs: daniel-checks.json. Nothing is installed; no GPU."""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "teoria-visual-v2" / "vendor"))
sys.path.insert(1, str(HERE / "daniel-ref" / "src"))
import types
# config.py imports torch at top level (unused by FEM). Register a stub package + config with the same three constants
# (WIDTH=100e-3, HEIGHT=1e-3, CENTER=HEIGHT/2, copied from daniel-ref/src/pinn_piezo/config.py) so torch is not needed.
pkg = types.ModuleType("pinn_piezo"); pkg.__path__ = [str(HERE / "daniel-ref" / "src" / "pinn_piezo")]
sys.modules["pinn_piezo"] = pkg
cfg = types.ModuleType("pinn_piezo.config"); cfg.WIDTH = 100e-3; cfg.HEIGHT = 1e-3; cfg.H2 = 0.5e-3; cfg.CENTER = 0.5e-3
sys.modules["pinn_piezo.config"] = cfg
import numpy as np
from pinn_piezo import materials as m
from pinn_piezo import fem
out = {}
C, e, kap = fem.constitutive_matrices()
out["materials"] = {"C11": m.C11, "C12": m.C12, "G": m.G, "e31(=e21 thesis)": float(m.pze_E[0,1]), "e33(=e22 thesis)": float(m.pze_E[1,1]),
                    "kappa_x": float(kap[0,0]), "kappa_y": float(kap[1,1])}
out["thesis_table"] = {"C11": 2.1836e9, "C12": 0.6332e9, "C66": 0.775e9, "e21": 0.02904, "e22": -0.05157, "kappa_x": 1.062e-10, "kappa_y": 1.041e-10}
res = {}
for case, kw in (("indirect", dict(voltage=100.0)), ("direct", dict(force=1.0))):
    for nx, ny in ((100, 4), (200, 8), (400, 16)):
        t = time.perf_counter()
        r = fem.solve_piezo(case, nx=nx, ny=ny, **kw)
        pts = r.points
        tip = np.argmin(np.abs(pts[:,0]-0.1) + np.abs(pts[:,1]-0.0005))
        top = np.isclose(pts[:,1], 1e-3)
        res[f"{case}_{nx}x{ny}"] = {"dofs": int(r.n_dofs), "sec": time.perf_counter()-t,
            "u_tip_mid": float(r.u[tip]), "v_tip_mid": float(r.v[tip]), "v_max_abs": float(np.abs(r.v).max()),
            "phi_min": float(r.phi.min()), "phi_max": float(r.phi.max()),
            "phi_top_mean": float(r.phi[top].mean())}
        print(case, nx, ny, res[f"{case}_{nx}x{ny}"], flush=True)
out["fem"] = res
json.dump(out, open(HERE / "daniel-checks.json", "w"), indent=2)
