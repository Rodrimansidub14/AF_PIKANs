"""Training-visible signal battery for the steady DFG 2D-1 (Re=20) fields ``(u, v, p)``.

Features only: computed from the model and the Navier-Stokes operator, never from a reference field.
Works with any ``model(xy) -> (n, 3)`` in physical ``(x, y)`` coordinates (``dfg.DFGRBFPIKAN`` or
the campaign classes). Forces use the sigma-n traction (``dfg.qoi_from_cylinder``); the decomposition
into pressure and viscous parts is reported separately. No ``evals`` import.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import dfg

H = dfg.DOMAIN_Y[1]
L = dfg.DOMAIN_X[1]
CX, CY = dfg.CYLINDER_CENTER
R = dfg.CYLINDER_RADIUS


def dev_grid(nx: int = 220, ny: int = 41) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, L, nx)
    ys = np.linspace(0.0, H, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    fluid = np.hypot(pts[:, 0] - CX, pts[:, 1] - CY) > R
    return pts, fluid


def _fields_and_grads(model, pts: np.ndarray):
    dtype = next(model.parameters()).dtype
    xy = torch.tensor(pts, dtype=dtype, requires_grad=True)
    f = model(xy)
    return xy, f


def dense_features(model, pts: np.ndarray, chunk: int = 4096) -> dict[str, np.ndarray]:
    out: dict[str, list] = {k: [] for k in ("u", "v", "p", "continuity", "momentum_x", "momentum_y", "vorticity", "grad_residual_norm")}
    for s in range(0, len(pts), chunk):
        xy, f = _fields_and_grads(model, pts[s:s + chunk])
        res = dfg.steady_ns_residual(f, xy)
        ux, uy, vx, vy, _, _ = dfg._first_derivatives(f, xy)
        total = sum(r.square() for r in res.values()).sqrt()
        g = torch.autograd.grad(total.sum(), xy, retain_graph=False, allow_unused=True)[0]
        for k, v in (("u", f[:, 0]), ("v", f[:, 1]), ("p", f[:, 2]), ("continuity", res["continuity"][:, 0]),
                     ("momentum_x", res["momentum_x"][:, 0]), ("momentum_y", res["momentum_y"][:, 0]),
                     ("vorticity", (vx - uy)[:, 0]), ("grad_residual_norm", g.norm(dim=1) if g is not None else torch.zeros(len(xy)))):
            out[k].append(v.detach().numpy().astype(np.float64))
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["x"], d["y"] = pts[:, 0].copy(), pts[:, 1].copy()
    d["cyl_distance"] = np.hypot(pts[:, 0] - CX, pts[:, 1] - CY) - R
    d["wall_distance"] = np.minimum(pts[:, 1], H - pts[:, 1])
    return d


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")


def scalar_signals(model, dense: dict[str, np.ndarray], fluid: np.ndarray) -> dict[str, float]:
    s: dict[str, float] = {}
    near = fluid & (dense["cyl_distance"] > 0) & (dense["cyl_distance"] < 0.05)
    wake = fluid & (dense["x"] >= 0.25) & (dense["x"] <= 1.0) & (np.abs(dense["y"] - CY) <= 0.1)
    for name, mask in (("global", fluid), ("near_cylinder", near), ("wake", wake)):
        for k in ("continuity", "momentum_x", "momentum_y"):
            s[f"dfg/{k}_rms/{name}"] = _rms(dense[k][mask])
        s[f"dfg/region_point_fraction/{name}"] = float(mask.mean())
    tot = dense["continuity"] ** 2 + dense["momentum_x"] ** 2 + dense["momentum_y"] ** 2
    s["dfg/residual_mass_near_cylinder"] = float(tot[near].sum() / max(tot[fluid].sum(), 1e-300))
    dtype = next(model.parameters()).dtype

    def ev(pts, grad=False):
        xy = torch.tensor(np.asarray(pts), dtype=dtype, requires_grad=grad)
        return xy, model(xy)

    # no-slip on walls and cylinder, inlet profile, outlet natural traction
    xs = np.linspace(0, L, 400)
    _, fb = ev(np.column_stack([xs, np.zeros_like(xs)])); _, ft = ev(np.column_stack([xs, np.full_like(xs, H)]))
    th = 2 * math.pi * np.arange(512) / 512
    _, fc = ev(np.column_stack([CX + R * np.cos(th), CY + R * np.sin(th)]))
    noslip = torch.cat([fb[:, :2], ft[:, :2], fc[:, :2]]).detach().numpy()
    s["dfg/noslip_rms_normalized"] = _rms(noslip / dfg.U_MEAN)
    ys = np.linspace(0, H, 200)
    _, fi = ev(np.column_stack([np.zeros_like(ys), ys]))
    inlet = 4 * dfg.U_MAX * ys * (H - ys) / H**2
    s["dfg/inlet_profile_rmse_over_umax"] = _rms(fi[:, 0].detach().numpy() - inlet) / dfg.U_MAX
    s["dfg/inlet_v_rms_over_umax"] = _rms(fi[:, 1].detach().numpy()) / dfg.U_MAX
    xy_o, fo = ev(np.column_stack([np.full_like(ys, L), ys]), grad=True)
    gu = torch.autograd.grad(fo[:, 0].sum(), xy_o, retain_graph=True)[0]; gv = torch.autograd.grad(fo[:, 1].sum(), xy_o)[0]
    t_x = (dfg.NU * gu[:, 0] - fo[:, 2]) / dfg.PRESSURE_SCALE; t_y = dfg.NU * gv[:, 0] / dfg.PRESSURE_SCALE
    s["dfg/outlet_traction_rms"] = _rms(np.concatenate([t_x.detach().numpy(), t_y.detach().numpy()]))
    s["dfg/outlet_pressure_mean"] = float(fo[:, 2].mean())
    # flux balance
    trap = getattr(np, "trapezoid", None) or np.trapz
    q_in = float(trap(fi[:, 0].detach().numpy(), ys)); q_out = float(trap(fo[:, 0].detach().numpy(), ys))
    s["dfg/flux_in"] = q_in; s["dfg/flux_out"] = q_out; s["dfg/flux_imbalance_rel"] = (q_out - q_in) / max(abs(q_in), 1e-300)
    # forces: sigma-n and decomposition; wall shear on cylinder
    xy_c, f_c = ev(np.column_stack([CX + R * np.cos(th), CY + R * np.sin(th)]), grad=True)
    ux, uy, vx, vy, _, _ = dfg._first_derivatives(f_c, xy_c)
    nx, ny = np.cos(th), np.sin(th); ds = 2 * math.pi * R / len(th); cs = 2.0 / (dfg.U_MEAN**2 * dfg.DIAMETER)
    p_c = f_c[:, 2].detach().numpy(); ux_, uy_, vx_, vy_ = (a[:, 0].detach().numpy() for a in (ux, uy, vx, vy))
    ndu = ux_ * nx + uy_ * ny; ndv = vx_ * nx + vy_ * ny
    s["dfg/cd_pressure_part"] = float(cs * np.sum(-p_c * nx) * ds); s["dfg/cd_viscous_part"] = float(cs * np.sum(dfg.NU * ndu) * ds)
    s["dfg/cl_pressure_part"] = float(cs * np.sum(-p_c * ny) * ds); s["dfg/cl_viscous_part"] = float(cs * np.sum(dfg.NU * ndv) * ds)
    tau = -ny * ndu + nx * ndv
    s["dfg/wall_shear_cyl_rms"] = _rms(dfg.NU * tau); s["dfg/wall_shear_cyl_max"] = float(np.abs(dfg.NU * tau).max())
    s["dfg/cylinder_divergence_rms"] = _rms((ux_ + vy_) / dfg.CONTINUITY_SCALE)
    q = dfg.qoi_from_cylinder(th, p_c, ux_, uy_, vx_, vy_, float(ev([[0.15, 0.2]])[1][0, 2]), float(ev([[0.25, 0.2]])[1][0, 2]))
    s.update({f"dfg/qoi_{k}": float(v) for k, v in q.items() if k in ("drag_coefficient", "lift_coefficient", "pressure_difference")})
    # recirculation length behind the cylinder on y=0.2
    xr = np.linspace(0.25, 1.0, 600)
    _, fr = ev(np.column_stack([xr, np.full_like(xr, CY)]))
    ur = fr[:, 0].detach().numpy(); neg = np.where(ur < 0)[0]
    s["dfg/recirc_length"] = float(xr[neg.max()] - 0.25) if len(neg) else 0.0
    return s
