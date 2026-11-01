"""Endpoint-only CUDA score for one frozen canonical Burgers checkpoint.

DRAFT_UNVERIFIED. Numerical grids, Cole-Hopf values, derivatives, quadrature,
and metrics run on the assigned T4. CPU is used only for checkpoint/artifact IO.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


NU = 0.01 / math.pi
NX = 1025
NT = 129
QUADRATURE_ORDER = 128
CHUNK_SIZE = 2048


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _model(arm: str, model_id: str) -> torch.nn.Module:
    if arm == "rbf" and model_id == "rbf_fixed":
        from src.research_pipeline.sciml.model import build_model

        return build_model("burgers", {"id": "rbf_fixed", "basis_count": 11})
    if arm == "vanilla" and model_id == "vanilla_mlp_tanh_5x100":
        from burgers_paired_training import VanillaMLP

        return VanillaMLP()
    raise ValueError(f"arm/checkpoint model mismatch: {arm!r}, {model_id!r}")


def _gauss_hermite(order: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Golub-Welsch eigensolve for physicists' Hermite polynomials, on CUDA.
    off_diagonal = torch.sqrt(torch.arange(1, order, dtype=torch.float64, device=device) / 2.0)
    jacobi = torch.zeros((order, order), dtype=torch.float64, device=device)
    indices = torch.arange(order - 1, device=device)
    jacobi[indices, indices + 1] = off_diagonal
    jacobi[indices + 1, indices] = off_diagonal
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    weights = math.sqrt(math.pi) * eigenvectors[0, :].square()
    return nodes, weights


def _cole_hopf(x: torch.Tensor, t: torch.Tensor, nodes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values = -torch.sin(math.pi * x)
    evolving = t > 0.0
    selected = torch.nonzero(evolving, as_tuple=False).reshape(-1)
    kappa = 1.0 / (2.0 * math.pi * NU)
    for start in range(0, selected.numel(), 4096):
        indices = selected[start : start + 4096]
        x_part = x.index_select(0, indices)
        t_part = t.index_select(0, indices)
        coordinates = x_part[:, None] + 2.0 * torch.sqrt(NU * t_part[:, None]) * nodes[None, :]
        potential = torch.exp(-kappa * (torch.cos(math.pi * coordinates) + 1.0))
        weighted = potential * weights[None, :]
        phi = weighted.sum(dim=1)
        phi_x = (weighted * (kappa * math.pi * torch.sin(math.pi * coordinates))).sum(dim=1)
        values.index_copy_(0, indices, -2.0 * NU * phi_x / phi)
    values = torch.where(torch.isclose(x.abs(), torch.ones_like(x), rtol=0.0, atol=1e-14), torch.zeros_like(values), values)
    return values


def _weighted_rmse(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(weights * values.square()) / torch.sum(weights))


def score(checkpoint_path: Path, arm: str, seed: int, out: Path) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each score worker must see exactly one assigned CUDA T4.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    if "t4" not in device_name.lower():
        raise RuntimeError(f"Expected T4, got {device_name!r}.")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != seed or checkpoint.get("requested_gpu_seconds") not in ("final", None):
        raise ValueError("Scorer accepts only the declared final endpoint checkpoint for this seed.")
    model = _model(arm, str(checkpoint.get("model_id"))).to(device=device, dtype=torch.float64).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    xs = torch.linspace(-1.0, 1.0, NX, dtype=torch.float64, device=device)
    ts = torch.linspace(0.0, 1.0, NT, dtype=torch.float64, device=device)
    xx, tt = torch.meshgrid(xs, ts, indexing="ij")
    points = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
    prediction_parts: list[torch.Tensor] = []
    ux_parts: list[torch.Tensor] = []
    ut_parts: list[torch.Tensor] = []
    uxx_parts: list[torch.Tensor] = []
    for begin in range(0, points.shape[0], CHUNK_SIZE):
        coordinates = points[begin : begin + CHUNK_SIZE].detach().clone().requires_grad_(True)
        prediction = model(coordinates).reshape(-1)
        gradient = torch.autograd.grad(prediction.sum(), coordinates, create_graph=True, retain_graph=True)[0]
        second = torch.autograd.grad(gradient[:, 0].sum(), coordinates, create_graph=False, retain_graph=False)[0]
        prediction_parts.append(prediction.detach())
        ux_parts.append(gradient[:, 0].detach())
        ut_parts.append(gradient[:, 1].detach())
        uxx_parts.append(second[:, 0].detach())
    prediction = torch.cat(prediction_parts)
    ux = torch.cat(ux_parts)
    ut = torch.cat(ut_parts)
    uxx = torch.cat(uxx_parts)
    nodes, quadrature_weights = _gauss_hermite(QUADRATURE_ORDER, device)
    flat_x, flat_t = points[:, 0], points[:, 1]
    truth = _cole_hopf(flat_x, flat_t, nodes, quadrature_weights)
    residual = ut + prediction * ux - NU * uxx
    error = prediction - truth

    wx = torch.ones(NX, dtype=torch.float64, device=device)
    wt = torch.ones(NT, dtype=torch.float64, device=device)
    wx[[0, -1]] = 0.5
    wt[[0, -1]] = 0.5
    grid_weights = (wx[:, None] * (2.0 / (NX - 1))) * (wt[None, :] * (1.0 / (NT - 1)))
    weights = grid_weights.reshape(-1)
    truth_norm = torch.sum(weights * truth.square())
    if not bool(torch.isfinite(truth_norm).item()) or float(truth_norm.item()) <= 0.0:
        raise ValueError("Common Cole-Hopf reference norm is invalid on CUDA.")
    metrics: dict[str, torch.Tensor] = {
        "field_relative_l2": torch.sqrt(torch.sum(weights * error.square()) / truth_norm),
        "field_rmse_abs": _weighted_rmse(error, weights),
        "field_p95_abs": torch.quantile(error.abs(), 0.95),
        "field_max_abs": error.abs().max(),
        "initial_max_abs_error": error[flat_t == 0.0].abs().max(),
        "boundary_max_abs_error": error[(flat_x == -1.0) | (flat_x == 1.0)].abs().max(),
        "residual_rmse_abs": _weighted_rmse(residual, weights),
        "residual_mean_abs": torch.sum(weights * residual.abs()) / torch.sum(weights),
    }
    critical = (flat_x.abs() <= 5.0 * NU) & (flat_t >= 1.0 / math.pi)
    if not bool(critical.any().item()) or bool(critical.all().item()):
        raise ValueError("CUDA validation grid does not contain front and complement points.")
    for label, mask in (("front", critical), ("complement", ~critical)):
        local_weights = weights[mask]
        local_error = error[mask]
        local_residual = residual[mask]
        metrics[f"{label}_field_rmse_abs"] = _weighted_rmse(local_error, local_weights)
        metrics[f"{label}_residual_rmse_abs"] = _weighted_rmse(local_residual, local_weights)
        metrics[f"{label}_residual_mean_abs"] = torch.sum(local_weights * local_residual.abs()) / torch.sum(local_weights)
    metric_tensors = torch.stack(list(metrics.values()))
    if not bool(torch.isfinite(metric_tensors).all().item()):
        raise FloatingPointError("CUDA score contains non-finite metrics.")
    metric_values = {key: float(value.item()) for key, value in metrics.items()}
    torch.cuda.synchronize(device)
    score_seconds = max(0.0, time.perf_counter() - started)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "validation.npz",
        x=xs.detach().cpu().numpy(),
        t=ts.detach().cpu().numpy(),
        prediction=prediction.reshape(NX, NT).detach().cpu().numpy(),
        reference=truth.reshape(NX, NT).detach().cpu().numpy(),
        error=error.reshape(NX, NT).detach().cpu().numpy(),
        residual=residual.reshape(NX, NT).detach().cpu().numpy(),
        trapezoid_weights=grid_weights.detach().cpu().numpy(),
    )
    report = {
        "draft_status": "DRAFT_UNVERIFIED",
        "problem": "burgers_pinnacle_1d",
        "seed": seed,
        "arm": arm,
        "checkpoint": str(checkpoint_path),
        "grid": {"nx": NX, "nt": NT},
        "reference": "CUDA Cole-Hopf/Gauss-Hermite order 128",
        "scoring_device": device_name,
        "evaluation_gpu_seconds_estimate": score_seconds,
        "endpoint_only": True,
        "scores_used_for_tuning": False,
        "scientific_accepted": False,
        "metrics": metric_values,
    }
    _write_json(out / "score.json", report)
    _write_json(out / "metrics.json", {f"burgers/{key}": value for key, value in metric_values.items()})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("rbf", "vanilla"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.checkpoint, args.arm, args.seed, args.out)
    print(json.dumps({"seed": args.seed, "arm": args.arm, "score_seconds": report["evaluation_gpu_seconds_estimate"]}))


if __name__ == "__main__":
    main()
