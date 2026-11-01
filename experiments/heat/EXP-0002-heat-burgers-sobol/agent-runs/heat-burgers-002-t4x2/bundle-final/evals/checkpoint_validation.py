"""Independent CPU validation for saved Heat and Burgers SciML checkpoints.

This evaluator uses a small, fixed validation grid that is separate from the
problem modules' held-out final evaluation grids. Derivatives are recomputed
from the loaded model with autograd; candidate-provided residuals are never
accepted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evals.problems.burgers_1d import reference as burgers_reference
from evals.problems.heat_1d import reference as heat_reference
from src.research_pipeline.sciml.model import RBFPIKAN


_GRID_NX = 129
_GRID_NT = 65
_CHUNK_SIZE = 256
_PROBLEMS = {"heat", "burgers"}


def _grid(problem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_bounds = (0.0, 1.0) if problem == "heat" else (-1.0, 1.0)
    xs = np.linspace(*x_bounds, _GRID_NX, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, _GRID_NT, dtype=np.float64)
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    return xs, ts, np.column_stack((xx.ravel(), tt.ravel()))


def _trapezoid_tensor_weights(xs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    wx = np.ones(xs.size, dtype=np.float64)
    wt = np.ones(ts.size, dtype=np.float64)
    wx[[0, -1]] = 0.5
    wt[[0, -1]] = 0.5
    wx *= (xs[-1] - xs[0]) / (xs.size - 1)
    wt *= (ts[-1] - ts[0]) / (ts.size - 1)
    return np.multiply.outer(wx, wt).ravel()


def _autograd_fields(model: torch.nn.Module, points: np.ndarray) -> tuple[np.ndarray, ...]:
    predictions: list[np.ndarray] = []
    space_derivatives: list[np.ndarray] = []
    time_derivatives: list[np.ndarray] = []
    second_space_derivatives: list[np.ndarray] = []

    model.to(device="cpu", dtype=torch.float64)
    model.eval()
    with torch.enable_grad():
        for start in range(0, points.shape[0], _CHUNK_SIZE):
            point_chunk = torch.tensor(
                points[start : start + _CHUNK_SIZE],
                dtype=torch.float64,
                device="cpu",
                requires_grad=True,
            )
            try:
                prediction = model(point_chunk)
            except Exception as exc:
                raise ValueError(f"candidate evaluation failed: {exc}") from exc
            if not isinstance(prediction, torch.Tensor):
                raise ValueError("candidate must return a torch.Tensor")
            if prediction.numel() != point_chunk.shape[0]:
                raise ValueError("candidate output must contain one scalar per input point")
            prediction = prediction.reshape(-1)
            if not torch.isfinite(prediction).all():
                raise ValueError("candidate produced non-finite predictions")
            if not prediction.requires_grad:
                raise ValueError("candidate output is not differentiable with respect to its inputs")

            try:
                gradient = torch.autograd.grad(
                    prediction,
                    point_chunk,
                    grad_outputs=torch.ones_like(prediction),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                u_x = gradient[:, 0]
                u_t = gradient[:, 1]
                if not torch.isfinite(gradient).all():
                    raise ValueError("candidate produced non-finite first derivatives")
                if u_x.requires_grad:
                    second_gradient = torch.autograd.grad(
                        u_x,
                        point_chunk,
                        grad_outputs=torch.ones_like(u_x),
                        create_graph=False,
                        retain_graph=False,
                    )[0]
                    u_xx = second_gradient[:, 0]
                else:
                    u_xx = torch.zeros_like(u_x)
            except Exception as exc:
                if isinstance(exc, ValueError) and "non-finite" in str(exc):
                    raise
                raise ValueError(f"could not compute candidate derivatives with autograd: {exc}") from exc

            predictions.append(prediction.detach().cpu().numpy())
            space_derivatives.append(u_x.detach().cpu().numpy())
            time_derivatives.append(u_t.detach().cpu().numpy())
            second_space_derivatives.append(u_xx.detach().cpu().numpy())

    values = (
        np.concatenate(predictions),
        np.concatenate(space_derivatives),
        np.concatenate(time_derivatives),
        np.concatenate(second_space_derivatives),
    )
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("candidate produced non-finite predictions or derivatives")
    return values


def _weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.square(values)) / np.sum(weights)))


def _residual_metrics(prefix: str, residual: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    absolute = np.abs(residual)
    return {
        f"{prefix}rmse_abs": _weighted_rmse(residual, weights),
        f"{prefix}p95_abs": float(np.quantile(absolute, 0.95)),
        f"{prefix}max_abs": float(np.max(absolute)),
    }


def _plot_validation(
    out_dir: Path,
    xs: np.ndarray,
    ts: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> None:
    try:
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Plotly is required to write the offline validation plot") from exc

    error = np.abs(prediction - reference)
    panels = (
        ("Prediction", prediction),
        ("Reference", reference),
        ("Absolute error", error),
    )
    figure = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _ in panels])
    for column, (_, values) in enumerate(panels, start=1):
        figure.add_trace(
            go.Heatmap(x=ts, y=xs, z=values, coloraxis='coloraxis' if column<3 else 'coloraxis2'),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="t", row=1, col=column)
        figure.update_yaxes(title_text="x", row=1, col=column)
    figure.update_layout(title_text="Checkpoint validation", width=1500, height=500,
        coloraxis={'cmin':float(min(prediction.min(),reference.min())),
                   'cmax':float(max(prediction.max(),reference.max())),
                   'colorbar':{'x':.64,'len':.78,'title':'u'}},
        coloraxis2={'cmin':0.,'colorbar':{'x':1.03,'len':.78,'title':'|error|'}})
    figure.write_html(out_dir / "validation.html", include_plotlyjs=True, full_html=True)


def evaluate_model(
    model: torch.nn.Module,
    problem: str,
    out_dir: str | Path,
) -> dict[str, float]:
    """Evaluate a model on the fixed 129 x 65 validation grid and save artifacts."""
    if problem not in _PROBLEMS:
        raise ValueError(f"unknown checkpoint problem: {problem!r}")
    torch.set_num_threads(2)

    xs, ts, points = _grid(problem)
    prediction, u_x, u_t, u_xx = _autograd_fields(model, points)
    x = points[:, 0]
    t = points[:, 1]

    if problem == "heat":
        reference = heat_reference.exact(x, t)
        residual = heat_reference.residual(u_t, u_xx)
        critical_mask = None
    else:
        reference = burgers_reference.exact(x, t)
        residual = burgers_reference.residual(prediction, u_t, u_x, u_xx)
        critical_mask = burgers_reference.critical_mask(x, t)

    if not np.isfinite(reference).all() or not np.isfinite(residual).all():
        raise ValueError("validation produced non-finite reference values or residuals")

    weights = _trapezoid_tensor_weights(xs, ts)
    error = prediction - reference
    reference_norm_sq = float(np.sum(weights * np.square(reference)))
    if not math.isfinite(reference_norm_sq) or reference_norm_sq <= 0:
        raise ValueError("reference has no finite, nonzero weighted L2 norm")

    metrics: dict[str, float] = {
        "field_relative_l2": float(np.sqrt(np.sum(weights * np.square(error)) / reference_norm_sq)),
        "field_rmse_abs": _weighted_rmse(error, weights),
        "field_p95_abs": float(np.quantile(np.abs(error), 0.95)),
        "field_max_abs": float(np.max(np.abs(error))),
        "initial_max_abs_error": float(np.max(np.abs(error[t == 0.0]))),
        "boundary_max_abs_error": float(np.max(np.abs(error[(x == xs[0]) | (x == xs[-1])]))),
    }
    metrics.update(_residual_metrics("residual_", residual, weights))
    field_scale=1. if problem=='heat' else .5
    residual_scale=1.1 if problem=='heat' else .76
    metrics['field_rmse_normalized']=metrics['field_rmse_abs']/field_scale
    metrics['residual_mean_abs']=float(np.sum(weights*np.abs(residual))/np.sum(weights))
    metrics['residual_mean_abs_normalized']=metrics['residual_mean_abs']/residual_scale

    if critical_mask is not None:
        if not critical_mask.any() or critical_mask.all():
            raise ValueError("Burgers validation grid must contain front and complement points")
        for name, mask in (("front", critical_mask), ("complement", ~critical_mask)):
            metrics[f'{name}_field_rmse_abs']=_weighted_rmse(error[mask],weights[mask])
            metrics[f'{name}_field_rmse_normalized']=metrics[f'{name}_field_rmse_abs']/field_scale
            metrics[f"{name}_n_points"] = float(np.count_nonzero(mask))
            metrics.update(
                _residual_metrics(
                    f"{name}_residual_",
                    residual[mask],
                    weights[mask],
                )
            )

    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("validation metrics contain a non-finite value")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    prediction_grid = prediction.reshape(_GRID_NX, _GRID_NT)
    reference_grid = reference.reshape(_GRID_NX, _GRID_NT)
    np.savez_compressed(
        out_path / "validation.npz",
        x=xs,
        t=ts,
        prediction=prediction_grid,
        reference=reference_grid,
        error=(prediction_grid - reference_grid),
        absolute_error=np.abs(prediction_grid - reference_grid),
        residual=residual.reshape(_GRID_NX, _GRID_NT),
        trapezoid_weights=weights.reshape(_GRID_NX, _GRID_NT),
        final_test_used=np.asarray(False),
        scientific_accepted=np.asarray(False),
        **({"front_mask": critical_mask.reshape(_GRID_NX, _GRID_NT)} if critical_mask is not None else {}),
    )
    report = {
        "problem": problem,
        "grid": {"nx": _GRID_NX, "nt": _GRID_NT, "kind": "fixed_uniform_tensor"},
        "final_test_used": False,
        "scientific_accepted": False,
        "metrics": metrics,
    }
    (out_path / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_validation(out_path, xs, ts, prediction_grid, reference_grid)
    return metrics


def evaluate(checkpoint_path: str | Path, out_dir: str | Path) -> dict[str, float]:
    """Load a standard SciML checkpoint safely and run independent validation."""
    torch.set_num_threads(2)
    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dictionary")
    problem = payload.get("problem")
    state_dict = payload.get("state_dict")
    step = payload.get("step")
    if problem not in _PROBLEMS:
        raise ValueError(f"checkpoint problem must be one of {sorted(_PROBLEMS)}")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing a state_dict")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")

    model = RBFPIKAN(problem).double()
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError(f"checkpoint state_dict does not match {problem!r} model: {exc}") from exc
    metrics = evaluate_model(model, problem, out_dir)

    report_path = Path(out_dir) / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "step": step,
            "checkpoint": Path(checkpoint_path).name,
            "metrics": metrics,
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics
