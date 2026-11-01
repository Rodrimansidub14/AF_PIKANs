"""Referencia FEM de tensiones y desplazamiento electrico (paper, sec. 2.6): P3 refinada cerca de ambos extremos, 336 x 32 divisiones (293 619 DOF), evaluada con pesos de area en
0.001 L < x < 0.999 L sobre la rejilla 201 x 21. El paper NO publica la ley de refinamiento: aqui x_i = L (s_i - a sin(2 pi s_i) / (2 pi)), s_i uniforme, a = 0.6 (agrupa nodos en los dos extremos);
es una eleccion declarada, no la del paper. Salida: reference/fem_flux_<problema>.npz (xy, sigma_xx, sigma_yy, tau_xy, D_x, D_y en la rejilla) y reference/fem_flux_reference.json con los
maximos de referencia (paper: sigma_xx max 59.46 kPa directo / 4.416 kPa converso; D_y max 0.3351 / 10.6244 uC/m2). Uso: python make_fem_flux_reference.py [--nx 336 --ny 32 --a 0.6]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=str(HERE.parents[3] / "var/agent_runs/piezo-ref-20260930/daniel-ref-20261003"))
ap.add_argument("--nx", type=int, default=336); ap.add_argument("--ny", type=int, default=32); ap.add_argument("--a", type=float, default=0.6)
ap.add_argument("--uniform", action="store_true")
a = ap.parse_args()
os.environ.setdefault("PINN_PIEZO_OUTPUTS_DIR", str(HERE / "_tmp_outputs"))
sys.path[:0] = [str(Path(a.repo) / "src")]
from pinn_piezo import fem
from skfem import MeshTri
W, H = 0.1, 1e-3
if not a.uniform:
    def graded(nx, ny, *, width=W, height=H):
        s = np.linspace(0, 1, nx + 1)
        return MeshTri.init_tensor(width * (s - a.a * np.sin(2 * np.pi * s) / (2 * np.pi)), np.linspace(0, height, ny + 1))
    fem._structured_tri_mesh = graded
xs, ys = np.linspace(0, W, 201), np.linspace(0, H, 21)
X, Y = np.meshgrid(xs, ys, indexing="xy")
P = np.stack((X.ravel(), Y.ravel()), 1)
inside = (P[:, 0] > 0.001 * W) & (P[:, 0] < 0.999 * W)
wx = np.gradient(xs); wy = np.gradient(ys)
out = {}
for name, kw in (("converse", dict(case="indirect", voltage=100.0)), ("direct", dict(case="direct", force=-0.1))):
    t = time.time()
    r = fem.solve_piezo(nx=a.nx, ny=a.ny, poling_sign=-1.0, element_order=3, **kw)
    Pm = P.copy(); Pm[:, 1] = np.clip(Pm[:, 1], 0, H * (1 - 1e-12)) if False else Pm[:, 1]
    f = r.probe_flux(P)
    np.savez_compressed(HERE / "reference" / f"fem_flux_{name}.npz", xy=P, weights=np.outer(wy, wx).ravel(), inside=inside, **{k: np.asarray(v) for k, v in f.items()})
    out[name] = dict(n_dofs=r.n_dofs, seconds=time.time() - t, sigma_xx_max_kPa=float(np.abs(f["sigma_xx"][inside]).max()) / 1e3, D_y_max_uC_m2=float(np.abs(f["D_y"][inside]).max()) * 1e6, graded=not a.uniform, a=a.a)
    print(name, out[name], flush=True)
(HERE / "reference" / "fem_flux_reference.json").write_text(json.dumps(out, indent=1))
