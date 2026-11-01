"""GPU-capable fixed-RBF-PIKAN baseline for the steady DFG 2D-1 Re=20 flow.

The implementation is deliberately self-contained: it does not import the
``evals`` package or change any verifier contract.  It follows the published
FEATFLOW geometry, boundary conditions and force convention.  In particular,
the outflow uses the benchmark's do-nothing traction ``nu * grad(u) n - p n``
and no extra ``p=0`` point constraint is imposed.  The reported pressure
difference is front minus rear, ``p(0.15, 0.2) - p(0.25, 0.2)``.

FEATFLOW defines ``sigma = nu * grad(u) - p I`` and integrates ``sigma n``
around the cylinder (rho=1).  Its equivalent no-slip 2-D formulas are
``F_D = integral(nu * d(u_tau)/dn * n_y - p*n_x) ds`` and
``F_L = -integral(nu * d(u_tau)/dn * n_x + p*n_y) ds``.  Here ``n`` points
radially outward from the cylinder and ``tau=(-n_y,n_x)``.  Coefficients use
``2/(U_mean**2 * D)`` with ``U_mean=0.2`` and ``D=0.1``.  These conventions
are documented at the official FEATFLOW DFG 2D-1 benchmark page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn


DOMAIN_X = (0.0, 2.2)
DOMAIN_Y = (0.0, 0.41)
CYLINDER_CENTER = (0.2, 0.2)
CYLINDER_RADIUS = 0.05
NU = 0.001
RHO = 1.0
U_MAX = 0.3
U_MEAN = 0.2
DIAMETER = 2.0 * CYLINDER_RADIUS
MOMENTUM_SCALE = U_MEAN**2 / DIAMETER
CONTINUITY_SCALE = U_MEAN / DIAMETER
PRESSURE_SCALE = RHO * U_MEAN**2
MAX_TRAINING_SECONDS = 1200.0
FEATFLOW_REFERENCE = {
    "drag_coefficient": 5.57953523384,
    "lift_coefficient": 0.010618948146,
    "pressure_difference": 0.11752016697,
}
FEATFLOW_URL = (
    "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/"
    "cfdbenchmarking/flow/dfg_benchmark1_re20.html"
)


class RBFEdgeLayer(nn.Module):
    """KAN edge layer with fixed Gaussian centers and trainable coefficients."""

    def __init__(self, in_features: int, out_features: int, basis_count: int = 11):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, basis_count, dtype=torch.float32))
        self.register_buffer("width", torch.tensor(0.35, dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.empty(out_features, in_features, basis_count, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        nn.init.normal_(self.coefficients, mean=0.0, std=0.1 / math.sqrt(in_features))
        nn.init.normal_(self.base, mean=0.0, std=0.1 / math.sqrt(in_features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        bases = torch.exp(-((values.unsqueeze(-1) - self.centers) / self.width).square())
        return torch.einsum("big,oig->bo", bases, self.coefficients) + torch.nn.functional.linear(
            torch.nn.functional.silu(values), self.base
        )


class DFGRBFPIKAN(nn.Module):
    """Fixed-center RBF KAN with layer widths ``[2, 16, 16, 3]`` for ``(u,v,p)``.

    Coordinates are mapped to [-1,1]^2.  Network outputs are rescaled to
    physical velocity and kinematic-pressure units.  The pressure has no hard
    point gauge: the FEATFLOW natural outflow condition includes pressure, and
    the pressure-difference QoI is evaluated directly as a gauge-invariant
    difference.
    """

    architecture = (2, 16, 16, 3)
    model_id = "dfg_rbf_fixed_2_16_16_3"
    basis_count = 11

    def __init__(self, seed: int = 101):
        super().__init__()
        # Preserve the caller's RNG state while making initial weights reproducible.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.layers = nn.ModuleList(
                [
                    RBFEdgeLayer(2, 16, self.basis_count),
                    RBFEdgeLayer(16, 16, self.basis_count),
                    RBFEdgeLayer(16, 3, self.basis_count),
                ]
            )
        self.register_buffer(
            "output_scales", torch.tensor([U_MEAN, U_MEAN, PRESSURE_SCALE], dtype=torch.float32)
        )
        self.seed = int(seed)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("DFG input must have shape (n, 2) in physical (x, y) coordinates")
        x = 2.0 * (xy[:, 0:1] - DOMAIN_X[0]) / (DOMAIN_X[1] - DOMAIN_X[0]) - 1.0
        y = 2.0 * (xy[:, 1:2] - DOMAIN_Y[0]) / (DOMAIN_Y[1] - DOMAIN_Y[0]) - 1.0
        hidden = torch.cat((x, y), dim=1)
        for layer in self.layers[:-1]:
            hidden = torch.tanh(layer(hidden))
        raw = self.layers[-1](hidden)
        return raw * self.output_scales


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed_value(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _sobol_1d(count: int, seed: int) -> np.ndarray:
    # Scrambling almost surely avoids endpoints; clipping makes the segment
    # contract explicit and keeps corner nodes out of multiple boundary pools.
    values = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(int(math.ceil(math.log2(count))))[:, 0]
    epsilon = 1e-12
    return np.clip(values, epsilon, 1.0 - epsilon)


def _interior_sobol(count: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    exponent = max(1, int(math.ceil(math.log2(count))))
    max_candidates = max(1024, count * 8)
    while (candidate_count := 2**exponent) <= max_candidates:
        unit = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(exponent)
        x = DOMAIN_X[0] + unit[:, 0] * (DOMAIN_X[1] - DOMAIN_X[0])
        y = DOMAIN_Y[0] + unit[:, 1] * (DOMAIN_Y[1] - DOMAIN_Y[0])
        radius2 = (x - CYLINDER_CENTER[0]) ** 2 + (y - CYLINDER_CENTER[1]) ** 2
        accepted_mask = (
            (x > DOMAIN_X[0])
            & (x < DOMAIN_X[1])
            & (y > DOMAIN_Y[0])
            & (y < DOMAIN_Y[1])
            & (radius2 > CYLINDER_RADIUS**2)
        )
        accepted = np.column_stack((x[accepted_mask], y[accepted_mask]))
        if accepted.shape[0] >= count:
            return accepted[:count].astype(np.float64, copy=False), {
                "method": "scrambled_sobol_rejection",
                "dimension": 2,
                "scramble_seed": seed,
                "candidate_count": candidate_count,
                "accepted_count": int(accepted.shape[0]),
                "rejected_count": int(candidate_count - accepted.shape[0]),
            }
        exponent += 1
    raise RuntimeError("Sobol rejection sampler could not fill the requested DFG interior pool")


def generate_dfg_pools(
    seed: int,
    interior_count: int = 16384,
    boundary_count: int = 512,
    *,
    return_provenance: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate fixed scrambled-Sobol pools for the fluid domain and each boundary.

    The interior uses rejection sampling outside the closed cylinder.  Every
    boundary has an independent 1-D scrambled-Sobol parameter sequence; cylinder
    points are mapped to angle and omit a duplicated endpoint.
    """
    seed = _seed_value(seed)
    interior_count = _positive_int("interior_count", interior_count)
    boundary_count = _positive_int("boundary_count", boundary_count)
    if interior_count > 1_000_000 or boundary_count > 1_000_000:
        raise ValueError("pool counts may not exceed 1,000,000")

    interior, interior_meta = _interior_sobol(interior_count, seed)
    seed_modulus = 2**32 - 1
    offsets = {"inlet": 101, "lower_wall": 211, "upper_wall": 307, "cylinder": 401, "outlet": 503}
    unit = {name: _sobol_1d(boundary_count, (seed + offset) % seed_modulus) for name, offset in offsets.items()}
    pools = {
        "interior": interior,
        "inlet": np.column_stack((np.full(boundary_count, DOMAIN_X[0]), DOMAIN_Y[0] + unit["inlet"] * (DOMAIN_Y[1] - DOMAIN_Y[0]))),
        "lower_wall": np.column_stack((DOMAIN_X[0] + unit["lower_wall"] * (DOMAIN_X[1] - DOMAIN_X[0]), np.full(boundary_count, DOMAIN_Y[0]))),
        "upper_wall": np.column_stack((DOMAIN_X[0] + unit["upper_wall"] * (DOMAIN_X[1] - DOMAIN_X[0]), np.full(boundary_count, DOMAIN_Y[1]))),
        "cylinder": np.column_stack(
            (
                CYLINDER_CENTER[0] + CYLINDER_RADIUS * np.cos(2.0 * math.pi * unit["cylinder"]),
                CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(2.0 * math.pi * unit["cylinder"]),
            )
        ),
        "outlet": np.column_stack((np.full(boundary_count, DOMAIN_X[1]), DOMAIN_Y[0] + unit["outlet"] * (DOMAIN_Y[1] - DOMAIN_Y[0]))),
    }
    provenance = {
        "interior": interior_meta,
        "boundary": {
            "method": "independent_scrambled_sobol_parameterization",
            "scramble_seeds": {name: (seed + offset) % seed_modulus for name, offset in offsets.items()},
            "points_per_segment": boundary_count,
            "cylinder_parameterization": "theta=2*pi*q; q in (0,1); no duplicated endpoint",
            "corner_points_excluded": True,
        },
        "pool_sha256": {name: hashlib.sha256(values.tobytes(order="C")).hexdigest() for name, values in pools.items()},
        "seed": seed,
    }
    if return_provenance:
        return pools, provenance
    return pools


def _grad_scalar(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    if value.requires_grad:
        gradient = torch.autograd.grad(
            value,
            xy,
            grad_outputs=torch.ones_like(value),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is not None:
            return gradient
    return torch.zeros_like(xy)


def _first_derivatives(fields: torch.Tensor, xy: torch.Tensor) -> tuple[torch.Tensor, ...]:
    grads = [_grad_scalar(fields[:, index : index + 1], xy) for index in range(3)]
    return grads[0][:, 0:1], grads[0][:, 1:2], grads[1][:, 0:1], grads[1][:, 1:2], grads[2][:, 0:1], grads[2][:, 1:2]


def steady_ns_residual(
    fields: torch.Tensor,
    xy: torch.Tensor,
    nu: float = NU,
    velocity_scale: float = U_MEAN,
    length_scale: float = DIAMETER,
) -> dict[str, torch.Tensor]:
    """Return dimensionless steady incompressible Navier--Stokes residuals.

    ``fields`` contains physical ``(u,v,p)`` values and ``xy`` physical
    ``(x,y)`` coordinates. Momentum is normalized by ``U_ref^2/L_ref`` and
    continuity by ``U_ref/L_ref``.
    """
    if fields.ndim != 2 or fields.shape[1] != 3 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("fields and xy must have shapes (n,3) and (n,2)")
    if fields.shape[0] != xy.shape[0] or not xy.requires_grad:
        raise ValueError("fields and xy must have matching rows and xy.requires_grad=True")
    if nu <= 0.0 or velocity_scale <= 0.0 or length_scale <= 0.0:
        raise ValueError("nu and residual reference scales must be positive")

    u, v, p = (fields[:, i : i + 1] for i in range(3))
    ux, uy, vx, vy, px, py = _first_derivatives(fields, xy)
    uxx = _grad_scalar(ux, xy)[:, 0:1]
    uyy = _grad_scalar(uy, xy)[:, 1:2]
    vxx = _grad_scalar(vx, xy)[:, 0:1]
    vyy = _grad_scalar(vy, xy)[:, 1:2]
    momentum_scale = velocity_scale**2 / length_scale
    continuity_scale = velocity_scale / length_scale
    return {
        "momentum_x": (u * ux + v * uy + px - nu * (uxx + uyy)) / momentum_scale,
        "momentum_y": (u * vx + v * vy + py - nu * (vxx + vyy)) / momentum_scale,
        "continuity": (ux + vy) / continuity_scale,
    }


def resolve_device(device: str | torch.device = "cuda") -> torch.device:
    """Resolve only explicit CPU/CUDA requests; CUDA requests never silently fall back."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        index = torch.cuda.current_device() if resolved.index is None and torch.cuda.is_available() else resolved.index
        if not torch.cuda.is_available() or (index is not None and index >= torch.cuda.device_count()):
            raise RuntimeError(f"CUDA device {resolved} was requested but is unavailable")
        return torch.device("cuda", index)
    if resolved.type == "cpu":
        return resolved
    raise ValueError("device must be 'cpu' or an available CUDA device")


@dataclass(frozen=True)
class DFGConfig:
    seed: int = 101
    interior_count: int = 16384
    boundary_count: int = 512
    batch_size: int = 1024
    boundary_batch_size: int = 256
    learning_rate: float = 1.0e-3
    max_steps: int = 50000
    time_limit_s: float = MAX_TRAINING_SECONDS
    device: str = "cuda"

    def __post_init__(self) -> None:
        _seed_value(self.seed)
        for name in ("interior_count", "boundary_count", "batch_size", "boundary_batch_size", "max_steps"):
            _positive_int(name, getattr(self, name))
        if not math.isfinite(float(self.time_limit_s)):
            raise ValueError("time_limit_s must be finite")
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if self.time_limit_s > MAX_TRAINING_SECONDS:
            raise ValueError(f"time_limit_s exceeds the campaign maximum of {MAX_TRAINING_SECONDS:g} seconds")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


def pressure_difference(p_front: float | np.ndarray, p_rear: float | np.ndarray) -> float:
    """FEATFLOW pressure QoI: front-side pressure minus rear-side pressure."""
    front = np.asarray(p_front, dtype=np.float64)
    rear = np.asarray(p_rear, dtype=np.float64)
    if front.size == 0 or rear.size == 0 or not np.isfinite(front).all() or not np.isfinite(rear).all():
        raise ValueError("front and rear pressure samples must be finite and non-empty")
    return float(np.mean(front) - np.mean(rear))


def qoi_from_cylinder(
    theta: np.ndarray,
    pressure: np.ndarray,
    du_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dx: np.ndarray,
    dv_dy: np.ndarray,
    p_front: float | np.ndarray,
    p_rear: float | np.ndarray,
    *,
    nu: float = NU,
    u_mean: float = U_MEAN,
    diameter: float = DIAMETER,
) -> dict[str, float]:
    """Evaluate FEATFLOW cylinder drag, lift and front-minus-rear pressure difference.

    Derivatives are physical Cartesian derivatives at the cylinder surface.
    The tangential-velocity normal derivative is ``tau dot (grad(u) n)`` for
    ``tau=(-n_y,n_x)``. Equiangular, endpoint-free samples use the trapezoidal
    line rule; a constant pressure offset integrates to zero on this closed
    contour and also cancels in the front/rear difference.
    """
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (theta, pressure, du_dx, du_dy, dv_dx, dv_dy)]
    if len(arrays[0]) < 3 or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("cylinder angles, pressure and velocity gradients must be same-shaped vectors of length >=3")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("cylinder samples must be finite")
    if nu <= 0.0 or u_mean <= 0.0 or diameter <= 0.0:
        raise ValueError("nu, u_mean and diameter must be positive")
    theta, pressure, du_dx, du_dy, dv_dx, dv_dy = arrays
    nx, ny = np.cos(theta), np.sin(theta)
    tx, ty = -ny, nx
    normal_du = du_dx * nx + du_dy * ny
    normal_dv = dv_dx * nx + dv_dy * ny
    du_tau_dn = tx * normal_du + ty * normal_dv
    ds = 2.0 * math.pi * CYLINDER_RADIUS / len(theta)
    drag_force = float(np.sum(nu * du_tau_dn * ny - pressure * nx) * ds)
    lift_force = float(-np.sum(nu * du_tau_dn * nx + pressure * ny) * ds)
    coefficient_scale = 2.0 / (u_mean**2 * diameter)
    return {
        "drag_force": drag_force,
        "lift_force": lift_force,
        "drag_coefficient": coefficient_scale * drag_force,
        "lift_coefficient": coefficient_scale * lift_force,
        "pressure_difference": pressure_difference(p_front, p_rear),
    }


def evaluate_dfg_qois(model: nn.Module, device: str | torch.device = "cuda", surface_count: int = 512) -> dict[str, float]:
    """Evaluate QoIs from the model on a deterministic cylinder contour."""
    device = resolve_device(device)
    surface_count = _positive_int("surface_count", surface_count)
    if surface_count < 3:
        raise ValueError("surface_count must be at least 3")
    model.eval()
    theta = 2.0 * math.pi * np.arange(surface_count, dtype=np.float64) / surface_count
    surface = np.column_stack(
        (CYLINDER_CENTER[0] + CYLINDER_RADIUS * np.cos(theta), CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(theta))
    )
    xy = torch.as_tensor(surface, dtype=next(model.parameters()).dtype, device=device).requires_grad_(True)
    fields = model(xy)
    du_dx, du_dy, dv_dx, dv_dy, _, _ = _first_derivatives(fields, xy)
    p = fields[:, 2].detach().cpu().numpy()
    front = torch.tensor([[0.15, 0.2]], dtype=xy.dtype, device=device)
    rear = torch.tensor([[0.25, 0.2]], dtype=xy.dtype, device=device)
    p_front = float(model(front)[0, 2].detach().cpu())
    p_rear = float(model(rear)[0, 2].detach().cpu())
    return qoi_from_cylinder(
        theta,
        p,
        du_dx.detach().cpu().numpy(),
        du_dy.detach().cpu().numpy(),
        dv_dx.detach().cpu().numpy(),
        dv_dy.detach().cpu().numpy(),
        p_front,
        p_rear,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def train_dfg(config: DFGConfig, out_dir: str | Path) -> dict[str, Any]:
    """Train one bounded DFG run and save exact pools, checkpoint and metrics.

    The wall-clock budget is checked at every optimizer-step boundary. One
    synchronous GPU update can finish after the deadline; this is recorded as
    ``cap_overrun_seconds`` and the next update is never started after expiry.
    The requested device is strict: ``cuda`` errors instead of silently using
    CPU when CUDA is unavailable.
    """
    device = resolve_device(config.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pools, provenance = generate_dfg_pools(
        config.seed,
        config.interior_count,
        config.boundary_count,
        return_provenance=True,
    )
    pool_path = out / "dfg_training_pools.npz"
    np.savez_compressed(pool_path, **pools)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = DFGRBFPIKAN(seed=config.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    pool_tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in pools.items()
    }

    def batch(name: str, requested: int) -> torch.Tensor:
        source = pool_tensors[name]
        count = min(requested, source.shape[0])
        indices = torch.randint(source.shape[0], (count,), generator=generator, device=device)
        return source.index_select(0, indices)

    def normalized_velocity_loss(xy: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted = model(xy)[:, :2]
        return ((predicted - target) / U_MEAN).square().mean()

    started = time.perf_counter()
    deadline = started + config.time_limit_s
    steps = 0
    last_losses: dict[str, float] = {}
    history_rows: list[dict[str, float | int]] = []
    while steps < config.max_steps and time.perf_counter() < deadline:
        optimizer.zero_grad(set_to_none=True)
        interior = batch("interior", config.batch_size).detach().requires_grad_(True)
        residuals = steady_ns_residual(model(interior), interior)
        losses: dict[str, torch.Tensor] = {name: value.square().mean() for name, value in residuals.items()}

        inlet = batch("inlet", config.boundary_batch_size)
        inlet_y = inlet[:, 1:2]
        inlet_u = 4.0 * U_MAX * inlet_y * (DOMAIN_Y[1] - inlet_y) / (DOMAIN_Y[1] - DOMAIN_Y[0]) ** 2
        inlet_target = torch.cat((inlet_u, torch.zeros_like(inlet_u)), dim=1)
        losses["inlet"] = normalized_velocity_loss(inlet, inlet_target)

        for name in ("lower_wall", "upper_wall", "cylinder"):
            boundary = batch(name, config.boundary_batch_size)
            losses[name] = normalized_velocity_loss(
                boundary,
                torch.zeros((boundary.shape[0], 2), device=device, dtype=boundary.dtype),
            )

        outlet = batch("outlet", config.boundary_batch_size).detach().requires_grad_(True)
        outlet_fields = model(outlet)
        ux, _, vx, _, _, _ = _first_derivatives(outlet_fields, outlet)
        normal_stress_x = (NU * ux - outlet_fields[:, 2:3]) / PRESSURE_SCALE
        normal_stress_y = NU * vx / PRESSURE_SCALE
        losses["outlet"] = normal_stress_x.square().mean() + normal_stress_y.square().mean()

        total_loss = torch.stack(list(losses.values())).sum()
        total_loss.backward()
        optimizer.step()
        steps += 1
        last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
        history_rows.append({
            "step": steps,
            "elapsed_seconds": time.perf_counter() - started,
            "total_loss": float(total_loss.detach().cpu()),
            **last_losses,
        })
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    training_seconds = time.perf_counter() - started
    qois = evaluate_dfg_qois(model, device, surface_count=512)
    qoi_errors = {
        name: {
            "absolute": abs(qois[name] - reference),
            "relative": abs(qois[name] - reference) / max(abs(reference), np.finfo(float).tiny),
        }
        for name, reference in FEATFLOW_REFERENCE.items()
    }
    config_data = asdict(config)
    run_info: dict[str, Any] = {
        "schema_version": 1,
        "problem": "dfg_2d_1_steady_re20",
        "model": {"id": model.model_id, "architecture": list(model.architecture), "basis_count": model.basis_count},
        "seed": config.seed,
        "device": str(device),
        "training_config": config_data,
        "training_steps": steps,
        "training_seconds": training_seconds,
        "training_time_cap_seconds": config.time_limit_s,
        "cap_overrun_seconds": max(0.0, training_seconds - config.time_limit_s),
        "stop_reason": "step_limit" if steps >= config.max_steps else "time_limit",
        "sampling": provenance,
        "pool_archive": pool_path.name,
        "pool_archive_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "outlet_condition": "nu * grad(velocity) n - p n = 0; n=(1,0)",
        "pressure_gauge": "no point pin; FEATFLOW do-nothing outlet includes pressure; pressure difference is front minus rear",
        "qoi_convention": "FEATFLOW sigma=nu*grad(u)-pI; n points radially outward; coefficient 2/(U_mean^2*D)",
        "qoi_reference": {"source": FEATFLOW_URL, "values": FEATFLOW_REFERENCE, "field_reference": None},
        "scientific_accepted": False,
    }
    metrics = {
        "schema_version": 1,
        "problem": run_info["problem"],
        "seed": config.seed,
        "device": str(device),
        "steps": steps,
        "training_seconds": training_seconds,
        "training_time_cap_seconds": config.time_limit_s,
        "cap_overrun_seconds": run_info["cap_overrun_seconds"],
        "stop_reason": run_info["stop_reason"],
        "last_training_losses": last_losses,
        "qoi": qois,
        "qoi_absolute_error_vs_featflow": {name: value["absolute"] for name, value in qoi_errors.items()},
        "qoi_relative_error_vs_featflow": {name: value["relative"] for name, value in qoi_errors.items()},
        "field_relative_l2": None,
        "field_reference_status": "not_available_in_this_runner",
        "scientific_accepted": False,
    }
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "run": run_info,
    }
    history_path = out / "training_history.csv"
    history_columns = ["step", "elapsed_seconds", "total_loss", *last_losses.keys()]
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_columns)
        writer.writeheader()
        writer.writerows(history_rows)
    torch.save(checkpoint, out / "dfg_rbf_pikan.pt")
    _write_json(out / "run.json", run_info)
    _write_json(out / "metrics.json", metrics)
    return {"run": run_info, "metrics": metrics, "out_dir": str(out.resolve())}


