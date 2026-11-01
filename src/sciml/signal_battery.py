"""Battery of training-visible signals for 1D-time SciML checkpoints (Heat, Burgers, Allen-Cahn).

Everything here is a *feature*: it is computed from the model, the PDE operator and the
training pool only. Reference solutions never enter this module; labels (errors) are attached
by a separate scoring step. The module does not import ``evals``.

Grid convention: coordinates are ``(x, t)`` columns, matching ``model.py``.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from .model import derivatives

PROBLEM_DEFAULTS = {
    "heat": {"alpha": 0.1, "residual_scale": 1.1, "x": (0.0, 1.0)},
    "burgers": {"nu": 0.003183098861837907, "residual_scale": 2.0031830988618378, "x": (-1.0, 1.0)},
    "allen_cahn": {"diffusion": 0.001, "reaction": 5.0, "residual_scale": 6.001, "x": (-1.0, 1.0)},
}


def problem_parameters(problem: str, overrides: dict | None = None) -> dict:
    parameters = dict(PROBLEM_DEFAULTS[problem])
    for key, value in (overrides or {}).items():
        if key in parameters or key in ("nu", "alpha", "diffusion", "reaction", "residual_scale"):
            parameters[key] = value
    return parameters


def dev_grid(problem: str, nx: int = 129, nt: int = 65) -> np.ndarray:
    """Development grid (never the protected evaluation grid): uniform tensor, boundary included."""
    low, high = PROBLEM_DEFAULTS[problem]["x"]
    xs = np.linspace(low, high, nx)
    ts = np.linspace(0.0, 1.0, nt)
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    return np.stack([xx.ravel(), tt.ravel()], axis=1)


def residual_fn(problem: str, parameters: dict) -> Callable:
    def burgers(u, ux, ut, uxx):
        return ut + u * ux - parameters["nu"] * uxx

    def heat(u, ux, ut, uxx):
        return ut - parameters["alpha"] * uxx

    def allen_cahn(u, ux, ut, uxx):
        return ut - parameters["diffusion"] * uxx - parameters["reaction"] * (u - u**3)

    return {"burgers": burgers, "heat": heat, "allen_cahn": allen_cahn}[problem]


def _entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(1e-300)
    return -(p * torch.log(p.clamp_min(1e-300))).sum(dim=dim)


def dense_features(model, problem: str, points: np.ndarray, parameters: dict | None = None,
                   sensitivity_points: int = 512, chunk: int = 4096) -> dict[str, np.ndarray]:
    """Per-point training-visible features on ``points`` (N,2) in (x,t)."""
    parameters = problem_parameters(problem, parameters)
    res = residual_fn(problem, parameters)
    scale = parameters["residual_scale"]
    device = next(model.parameters()).device
    params = tuple(model.parameters())
    out: dict[str, list] = {}

    def push(name, value):
        out.setdefault(name, []).append(value.detach().cpu().numpy())

    for start in range(0, len(points), chunk):
        xy = torch.tensor(points[start:start + chunk], dtype=torch.float64, device=device, requires_grad=True)
        u = model(xy)
        ux, ut, uxx = derivatives(u, xy)
        r = res(u, ux, ut, uxx)
        gr = torch.autograd.grad(r, xy, torch.ones_like(r), retain_graph=True)[0]
        push("u", u.squeeze(-1)); push("u_x", ux.squeeze(-1)); push("u_t", ut.squeeze(-1))
        push("u_xx", uxx.squeeze(-1)); push("residual", r.squeeze(-1) / scale)
        push("grad_residual_norm", gr.norm(dim=1) / scale)
        with torch.no_grad():
            _, traces = model.forward_with_trace(xy.detach())
            if "basis" in traces[0]:
                basis = traces[0]["basis"]  # (N, inputs, G)
                push("basis0_entropy", _entropy(basis, dim=-1).mean(dim=1))
                push("basis0_peak", basis.amax(dim=-1).mean(dim=1))
            for i, trace in enumerate(traces[:-1]):
                push(f"saturation_l{i}", (trace["activation"].abs() > 0.99).double().mean(dim=1))
    features = {k: np.concatenate(v) for k, v in out.items()}
    features["x"] = points[:, 0].copy()
    features["t"] = points[:, 1].copy()

    # Parameter sensitivity on an evenly spaced subset (cost = subset size backward passes).
    n = min(sensitivity_points, len(points))
    idx = np.linspace(0, len(points) - 1, n).astype(int)
    xy = torch.tensor(points[idx], dtype=torch.float64, device=device)
    sens = np.empty(n)
    for k in range(n):
        point = xy[k:k + 1]
        g = torch.autograd.grad(model(point).sum(), params, allow_unused=True)
        sens[k] = math.sqrt(sum(float(v.square().sum()) for v in g if v is not None))
    full = np.full(len(points), np.nan)
    full[idx] = sens
    features["sensitivity"] = full
    return features


def feature_distance_to_interface(features: dict[str, np.ndarray], nx: int, nt: int) -> np.ndarray:
    """Distance (in x) from each point to the nearest zero crossing of the model's own u in its t-slice."""
    x = features["x"].reshape(nx, nt)
    u = features["u"].reshape(nx, nt)
    out = np.full((nx, nt), np.nan)
    for j in range(nt):
        s = np.sign(u[:, j])
        crossings = np.where(s[:-1] * s[1:] < 0)[0]
        if len(crossings) == 0:
            continue
        xs = x[crossings, j] - u[crossings, j] * (x[crossings + 1, j] - x[crossings, j]) / (u[crossings + 1, j] - u[crossings, j])
        out[:, j] = np.abs(x[:, j][:, None] - xs[None, :]).min(axis=1)
    return out.reshape(-1)


def jacobian_spectrum(model, points: np.ndarray, count: int = 256) -> dict[str, float]:
    device = next(model.parameters()).device
    params = tuple(model.parameters())
    idx = np.linspace(0, len(points) - 1, min(count, len(points))).astype(int)
    rows = []
    for k in idx:
        point = torch.tensor(points[k:k + 1], dtype=torch.float64, device=device)
        g = torch.autograd.grad(model(point).sum(), params, allow_unused=True)
        rows.append(torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1) for v, p in zip(g, params)]))
    sv = torch.linalg.svdvals(torch.stack(rows))
    p = sv / sv.sum()
    return {"jac/effective_rank": float(torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum())),
            "jac/cond": float(sv[0] / sv[-1].clamp_min(1e-300)), "jac/sigma_max": float(sv[0])}


def grad_conflict(model, terms: dict[str, Callable[[], torch.Tensor]]) -> dict[str, float]:
    """Cosine and norm ratio between the gradient of each loss term and the first ('pde') term."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = {}
    for name, fn in terms.items():
        g = torch.autograd.grad(fn(), params, allow_unused=True)
        grads[name] = torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1) for v, p in zip(g, params)])
    names = list(terms)
    base = grads[names[0]]
    out = {f"grad/norm_{n}": float(grads[n].norm()) for n in names}
    for n in names[1:]:
        denom = (base.norm() * grads[n].norm()).clamp_min(1e-300)
        out[f"grad/conflict_cos_{n}"] = float((base @ grads[n]) / denom)
        out[f"grad/norm_ratio_{n}"] = float(grads[n].norm() / base.norm().clamp_min(1e-300))
    return out


def basis_influence(model, problem: str, points: np.ndarray, parameters: dict | None = None,
                    layer: int = 0, max_points: int = 2048) -> dict[str, np.ndarray]:
    """Zero one basis function at a time in ``layer`` and record the change in u and residual RMS.
    Only defined for KAN layers exposing ``coefficients``."""
    lyr = model.layers[layer]
    if not hasattr(lyr, "coefficients"):
        return {}
    parameters = problem_parameters(problem, parameters)
    res = residual_fn(problem, parameters)
    idx = np.linspace(0, len(points) - 1, min(max_points, len(points))).astype(int)

    def measure():
        xy = torch.tensor(points[idx], dtype=torch.float64, requires_grad=True)
        u = model(xy)
        ux, ut, uxx = derivatives(u, xy)
        return u.detach().squeeze(-1), res(u, ux, ut, uxx).detach().squeeze(-1)

    u0, r0 = measure()
    G = lyr.coefficients.shape[-1]
    du = np.zeros(G); dr = np.zeros(G)
    with torch.no_grad():
        original = lyr.coefficients.clone()
    for g in range(G):
        with torch.no_grad():
            lyr.coefficients.copy_(original)
            lyr.coefficients[..., g] = 0.0
        u1, r1 = measure()
        du[g] = float(torch.sqrt(torch.mean((u1 - u0) ** 2)))
        dr[g] = float(torch.sqrt(torch.mean((r1 - r0) ** 2)))
    with torch.no_grad():
        lyr.coefficients.copy_(original)
    return {f"basis/influence_u_layer{layer}": du, f"basis/influence_residual_layer{layer}": dr}


def basis_occupancy(model, points: np.ndarray) -> dict[str, float]:
    xy = torch.tensor(points, dtype=torch.float64)
    stats = {}
    with torch.no_grad():
        _, traces = model.forward_with_trace(xy)
    for i, trace in enumerate(traces):
        if "basis" not in trace:
            continue
        occ = trace["basis"].mean(dim=0)  # (inputs, G)
        per_basis = occ.mean(dim=0)
        stats[f"basis/layer{i}/occupancy_min"] = float(per_basis.min())
        stats[f"basis/layer{i}/occupancy_max"] = float(per_basis.max())
        stats[f"basis/layer{i}/effective_count"] = float(torch.exp(_entropy(per_basis)))
        stats[f"basis/layer{i}/dead_fraction"] = float((per_basis < 0.01).double().mean())
    return stats


def physics_summary(model, problem: str, parameters: dict | None = None, nx: int = 1025, nt: int = 65) -> dict[str, float]:
    """Problem-specific physical signals computed from the model alone (no reference)."""
    parameters = problem_parameters(problem, parameters)
    low, high = PROBLEM_DEFAULTS[problem]["x"]
    xs = np.linspace(low, high, nx)
    ts = np.linspace(0.0, 1.0, nt)
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    xy = torch.tensor(np.stack([xx.ravel(), tt.ravel()], 1), dtype=torch.float64, requires_grad=True)
    u = model(xy)
    ux = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=False)[0][:, 0]
    u = u.detach().squeeze(-1).numpy().reshape(nx, nt)
    ux = ux.detach().numpy().reshape(nx, nt)
    out: dict[str, float] = {}
    _trapezoid = getattr(np, "trapezoid", None) or np.trapz
    trap = lambda f: _trapezoid(f, xs, axis=0)
    energy = trap(0.5 * u**2)  # (nt,)
    if problem == "burgers":
        nu = parameters["nu"]
        dissipation = nu * trap(ux**2)
        dEdt = np.gradient(energy, ts)
        bal = dEdt + dissipation
        late = ts >= 1.0 / math.pi
        out["burgers/ux_max_late"] = float(np.abs(ux[:, late]).max())
        out["burgers/front_width_late"] = float(np.median(2 * np.abs(u[:, late]).max(axis=0) / np.abs(ux[:, late]).max(axis=0)))
        out["burgers/antisym_err"] = float(np.linalg.norm(u + u[::-1]) / max(np.linalg.norm(u), 1e-300))
        out["burgers/energy_balance_rms"] = float(np.sqrt(np.mean(bal**2)))
        out["burgers/energy_balance_rel"] = float(np.sqrt(np.mean(bal**2)) / max(np.sqrt(np.mean(dissipation**2)), 1e-300))
        j = int(np.argmin(np.abs(ts - 1.0)))
        s = np.sign(u[:, j]); c = np.where(s[:-1] * s[1:] < 0)[0]
        out["burgers/zero_crossing_abs_t1"] = float(np.min(np.abs(xs[c])) if len(c) else float("nan"))
    elif problem == "allen_cahn":
        D, R = parameters["diffusion"], parameters["reaction"]
        pot = trap(D * ux**2 / 2 + R * (u**2 - 1) ** 2 / 4)
        out["ac/energy_t0"] = float(pot[0]); out["ac/energy_t1"] = float(pot[-1])
        out["ac/energy_increments"] = float((np.diff(pot) > 1e-12).sum())
        out["ac/evenness_err"] = float(np.linalg.norm(u - u[::-1]) / max(np.linalg.norm(u), 1e-300))
        for tq in (0.25, 0.5, 0.75, 1.0):
            j = int(np.argmin(np.abs(ts - tq)))
            s = np.sign(u[:, j]); c = np.where(s[:-1] * s[1:] < 0)[0]
            out[f"ac/n_interfaces_t{tq}"] = float(len(c))
            out[f"ac/max_abs_ux_t{tq}"] = float(np.abs(ux[:, j]).max())
        out["ac/interface_band_fraction"] = float((np.abs(u) < 0.8).mean())
    elif problem == "heat":
        alpha = parameters["alpha"]
        good = energy > 0
        slope = np.polyfit(ts[good], np.log(energy[good]), 1)[0]
        out["heat/decay_rate"] = float(-slope)
        out["heat/decay_rate_exact"] = float(2 * alpha * math.pi**2)
        out["heat/energy_t0"] = float(energy[0]); out["heat/energy_t1"] = float(energy[-1])
    return out


def pool_residual_stats(features_pool: dict[str, np.ndarray], region_mask: np.ndarray | None = None) -> dict[str, float]:
    r = np.abs(features_pool["residual"])
    q = np.sort(r)[::-1]
    top = max(1, len(q) // 10)
    out = {"pool/residual_mean": float(r.mean()), "pool/residual_p95": float(np.quantile(r, 0.95)),
           "pool/residual_max": float(r.max()),
           "pool/residual_top_decile_mass": float(q[:top].sum() / max(q.sum(), 1e-300)),
           "pool/residual_ess_fraction": float((r.sum() ** 2) / (len(r) * (r**2).sum() + 1e-300))}
    if region_mask is not None:
        out["pool/region_point_fraction"] = float(region_mask.mean())
        out["pool/region_residual_mass"] = float(r[region_mask].sum() / max(r.sum(), 1e-300))
    return out


def battery(model, problem: str, points: np.ndarray, grid_shape: tuple[int, int] | None = None,
            parameters: dict | None = None, terms: dict | None = None, sensitivity_points: int = 512) -> tuple[dict, dict]:
    """Run the whole battery. Returns (dense arrays, scalar summary)."""
    dense = dense_features(model, problem, points, parameters, sensitivity_points)
    if grid_shape is not None:
        dense["interface_distance"] = feature_distance_to_interface(dense, *grid_shape)
    scalars: dict[str, float] = {}
    scalars.update(jacobian_spectrum(model, points))
    scalars.update(basis_occupancy(model, points[:: max(1, len(points) // 2048)]))
    scalars.update(physics_summary(model, problem, parameters))
    if terms:
        scalars.update(grad_conflict(model, terms))
    r = np.abs(dense["residual"])
    scalars["residual/rms"] = float(np.sqrt(np.mean(r**2)))
    scalars["residual/p95"] = float(np.quantile(r, 0.95))
    scalars["residual/max"] = float(r.max())
    scalars["grad_residual/p95"] = float(np.quantile(dense["grad_residual_norm"], 0.95))
    scalars["sensitivity/median"] = float(np.nanmedian(dense["sensitivity"]))
    scalars["sensitivity/p95"] = float(np.nanquantile(dense["sensitivity"], 0.95))
    return dense, scalars
