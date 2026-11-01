"""Common Heat scorer for development and post-training confirmation.

DRAFT_UNVERIFIED. This module deliberately knows nothing about evals/ or the
protected 512x256 AF score grid definition. It evaluates numerical grids and
metrics on the assigned CUDA device; CPU is used only to serialize artifacts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


GRID_ID = "heat-dev-257x129-v1"
DEFAULT_NX = 257
DEFAULT_NT = 129
ALPHA = 0.1


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _trap_weights(count: int, device: torch.device) -> torch.Tensor:
    weights = torch.ones(count, dtype=torch.float64, device=device)
    weights[[0, -1]] = 0.5
    return weights


def _predict(model: torch.nn.Module, coordinates: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        values = model(coordinates).reshape(-1)
    return values


def finite_smoke(model: torch.nn.Module, *, device: str = "cuda:0") -> dict[str, Any]:
    """Check finite outputs only; intentionally computes no quality metric."""
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the finite-only GPU smoke check.")
    model = model.to(device=selected_device, dtype=torch.float64)
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    started = time.perf_counter()
    x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64, device=selected_device)
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64, device=selected_device)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    coordinates = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
    values = _predict(model, coordinates)
    finite = bool(torch.isfinite(values).all().item())
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    gpu_seconds = max(0.0, time.perf_counter() - started)
    return {
        "finite_output_check": finite,
        "points_checked": int(values.numel()),
        "quality_score_computed": False,
        "grid_id": None,
        "evaluation_device": str(selected_device),
        "evaluation_gpu_seconds_estimate": gpu_seconds,
    }


def score_torch_model(
    model: torch.nn.Module,
    out: Path | str,
    *,
    nx: int = DEFAULT_NX,
    nt: int = DEFAULT_NT,
    alpha: float = ALPHA,
    device: str = "cuda:0",
    grid_id: str | None = None,
) -> dict[str, float]:
    """Score the common development grid on CUDA; CPU is used only for artifact writes."""
    if nx < 3 or nt < 3:
        raise ValueError("Development grid dimensions must each be at least three.")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The common numerical evaluator requires the assigned CUDA T4.")
    model = model.to(device=selected_device, dtype=torch.float64)
    model.eval()

    torch.cuda.synchronize(selected_device)
    started = time.perf_counter()
    x_gpu = torch.linspace(0.0, 1.0, nx, dtype=torch.float64, device=selected_device)
    t_gpu = torch.linspace(0.0, 1.0, nt, dtype=torch.float64, device=selected_device)
    xx_gpu, tt_gpu = torch.meshgrid(x_gpu, t_gpu, indexing="ij")
    coordinates_gpu = torch.stack((xx_gpu.reshape(-1), tt_gpu.reshape(-1)), dim=1)
    with torch.no_grad():
        predicted_gpu = model(coordinates_gpu).reshape(nx, nt)
    exact_gpu = torch.sin(torch.pi * xx_gpu) * torch.exp(-alpha * torch.pi**2 * tt_gpu)
    error_gpu = predicted_gpu - exact_gpu

    wx = _trap_weights(nx, selected_device)[:, None]
    wt = _trap_weights(nt, selected_device)[None, :]
    quadrature_weights = wx * wt
    field_relative_l2 = torch.sqrt(
        torch.sum(quadrature_weights * error_gpu**2) / torch.sum(quadrature_weights * exact_gpu**2)
    )
    field_rmse = torch.sqrt(torch.mean(error_gpu**2))
    field_max_abs = torch.max(torch.abs(error_gpu))

    ic_coordinates = torch.stack((x_gpu, torch.zeros_like(x_gpu)), dim=1)
    ic_predicted = _predict(model, ic_coordinates)
    ic_exact = torch.sin(torch.pi * x_gpu)
    ic_error = ic_predicted - ic_exact

    left_coordinates = torch.stack((torch.zeros_like(t_gpu), t_gpu), dim=1)
    right_coordinates = torch.stack((torch.ones_like(t_gpu), t_gpu), dim=1)
    boundary_error = torch.cat(
        (_predict(model, left_coordinates), _predict(model, right_coordinates))
    )

    energy = torch.sum(wx * predicted_gpu**2, dim=0) / (nx - 1)
    exact_energy = 0.5 * torch.exp(-2.0 * alpha * torch.pi**2 * t_gpu)
    energy_relative_l2 = torch.sqrt(torch.mean((energy - exact_energy) ** 2) / torch.mean(exact_energy**2))

    # A deterministic 2,048-point subset of interior nodes bounds the CUDA
    # autograd cost while keeping PDE residual points identical by grid ID.
    ix = torch.arange(1, nx - 1, dtype=torch.long, device=selected_device)
    it = torch.arange(1, nt, dtype=torch.long, device=selected_device)
    grid_ix, grid_it = torch.meshgrid(ix, it, indexing="ij")
    residual_indices = torch.stack((grid_ix.reshape(-1), grid_it.reshape(-1)), dim=1)
    count = min(2048, residual_indices.shape[0])
    selected = torch.linspace(0, residual_indices.shape[0] - 1, count, dtype=torch.float64, device=selected_device).long()
    selected_indices = residual_indices.index_select(0, selected)
    residual_coordinates = torch.stack(
        (x_gpu.index_select(0, selected_indices[:, 0]), t_gpu.index_select(0, selected_indices[:, 1])), dim=1
    ).detach().requires_grad_(True)
    residual_values = model(residual_coordinates)
    gradient = torch.autograd.grad(
        residual_values,
        residual_coordinates,
        grad_outputs=torch.ones_like(residual_values),
        create_graph=True,
        retain_graph=True,
    )[0]
    ux = gradient[:, 0:1]
    ut = gradient[:, 1:2]
    uxx = torch.autograd.grad(
        ux,
        residual_coordinates,
        grad_outputs=torch.ones_like(ux),
        create_graph=False,
        retain_graph=False,
    )[0][:, 0:1]
    pde_residual = ut - alpha * uxx
    pde_residual_rms = torch.sqrt(torch.mean(pde_residual.detach() ** 2))
    torch.cuda.synchronize(selected_device)
    evaluation_gpu_seconds = max(0.0, time.perf_counter() - started)

    metric_tensors = torch.stack(
        (
            field_relative_l2,
            field_rmse,
            field_max_abs,
            torch.sqrt(torch.mean(ic_error**2)),
            torch.max(torch.abs(ic_error)),
            torch.sqrt(torch.mean(boundary_error**2)),
            torch.max(torch.abs(boundary_error)),
            pde_residual_rms,
            energy_relative_l2,
        )
    )
    if not bool(torch.isfinite(metric_tensors).all().item()):
        raise FloatingPointError("A common development metric is non-finite on CUDA.")
    metric_values = [float(value) for value in metric_tensors.detach().cpu().tolist()]
    metrics = dict(
        zip(
            (
                "field_relative_l2",
                "field_rmse",
                "field_max_abs",
                "ic_rmse",
                "ic_max_abs",
                "bc_rmse",
                "bc_max_abs",
                "pde_residual_rms",
                "energy_relative_l2",
            ),
            metric_values,
        )
    )

    payload = {
        "status": "DRAFT_UNVERIFIED",
        "grid_id": grid_id or (GRID_ID if (nx, nt) == (DEFAULT_NX, DEFAULT_NT) else f"heat-dev-{nx}x{nt}-custom"),
        "grid": {"nx": int(nx), "nt": int(nt), "x": [0.0, 1.0], "t": [0.0, 1.0]},
        "alpha": float(alpha),
        "residual_points": int(count),
        "residual_sampling": "deterministic linspace indices into interior development-grid nodes",
        "scoring_device": str(selected_device),
        "evaluation_gpu_seconds_estimate": evaluation_gpu_seconds,
        "cpu_role": "artifact serialization only",
        "protected_512x256_grid_loaded": False,
        "protected_512x256_grid_scored": (nx, nt) == (512, 256),
        "metrics": metrics,
        "evaluation_wall_seconds": max(0.0, time.perf_counter() - started),
    }
    _json_write(destination / "validation.json", payload)
    _json_write(destination / "metrics.json", {f"heat/{key}": value for key, value in metrics.items()})
    np.savez_compressed(
        destination / "validation.npz",
        x=x_gpu.detach().cpu().numpy(),
        t=t_gpu.detach().cpu().numpy(),
        prediction=predicted_gpu.detach().cpu().numpy(),
        reference=exact_gpu.detach().cpu().numpy(),
        error=error_gpu.detach().cpu().numpy(),
        energy=energy.detach().cpu().numpy(),
        exact_energy=exact_energy.detach().cpu().numpy(),
        residual_coordinates=residual_coordinates.detach().cpu().numpy(),
        residual_values=pde_residual.detach().cpu().numpy(),
    )
    return metrics
