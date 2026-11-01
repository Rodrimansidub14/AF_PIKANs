"""Held-out, physics-only diagnostics for the exploratory Allen–Cahn pilot.

No reference field or field-error calculation is used here. Derivatives are
recomputed from the saved model on points that are disjoint from training.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from evals.graders.exploratory import residual_score
from .model import derivatives
from .train import allen_cahn_residual, json_write


def allen_cahn_energy(x, u, ux, diffusion=0.001, reaction=5.0):
    """Integrate E[u]=∫(D|u_x|²/2 + λ(u²−1)²/4) dx on a 1D grid."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    ux = np.asarray(ux, dtype=np.float64).reshape(-1)
    if len(x) < 2 or x.shape != u.shape or x.shape != ux.shape:
        raise ValueError("x, u, and ux must be finite vectors of equal length >= 2")
    if not all(np.isfinite(value).all() for value in (x, u, ux)):
        raise ValueError("energy inputs must be finite")
    if np.any(np.diff(x) <= 0):
        raise ValueError("energy coordinates must be strictly increasing")
    if not np.isfinite(diffusion) or diffusion <= 0 or not np.isfinite(reaction) or reaction <= 0:
        raise ValueError("diffusion and reaction must be positive and finite")
    density = 0.5 * diffusion * np.square(ux) + 0.25 * reaction * np.square(np.square(u) - 1.0)
    return float(np.trapezoid(density, x))


def _fields(model, points, diffusion, reaction, chunk_size):
    device = next(model.parameters()).device
    outputs = {name: [] for name in ("u", "ux", "ut", "uxx", "residual")}
    with torch.enable_grad():
        for start in range(0, len(points), chunk_size):
            coordinates = torch.tensor(
                points[start : start + chunk_size],
                dtype=torch.float64,
                device=device,
                requires_grad=True,
            )
            u = model(coordinates)
            ux, ut, uxx = derivatives(u, coordinates)
            residual = allen_cahn_residual(u, ut, uxx, diffusion, reaction)
            for name, value in (("u", u), ("ux", ux), ("ut", ut), ("uxx", uxx), ("residual", residual)):
                outputs[name].append(value.detach().cpu().numpy().reshape(-1))
    arrays = {name: np.concatenate(values) for name, values in outputs.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Allen–Cahn held-out evaluation produced non-finite fields")
    return arrays


def _plot_fields(out_dir, xs, ts, prediction, residual):
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Plotly is required for Allen–Cahn exploratory visualizations") from exc

    fig = make_subplots(rows=1, cols=2, subplot_titles=("u(x,t)", "log10 |PDE residual|"))
    fig.add_trace(go.Heatmap(x=ts, y=xs, z=prediction, colorbar={"title": "u"}), row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=ts,
            y=xs,
            z=np.log10(np.maximum(np.abs(residual), 1e-16)),
            colorbar={"title": "log10 |r|"},
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="t", row=1, col=1)
    fig.update_xaxes(title_text="t", row=1, col=2)
    fig.update_yaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="x", row=1, col=2)
    fig.update_layout(title="Allen–Cahn exploratory field and residual", width=1250, height=520)
    fig.write_html(out_dir / "validation.html", include_plotlyjs=True, full_html=True)


def _plot_energy(out_dir, energy_rows):
    import plotly.graph_objects as go

    figure = go.Figure(
        go.Scatter(
            x=[row["time"] for row in energy_rows],
            y=[row["energy"] for row in energy_rows],
            mode="lines+markers",
            name="Allen–Cahn energy",
        )
    )
    figure.update_layout(
        title="Energy along the exploratory trajectory",
        xaxis_title="t",
        yaxis_title="E[u(t)]",
        width=900,
        height=480,
    )
    figure.write_html(out_dir / "energy.html", include_plotlyjs=True, full_html=True)


def evaluate_allen_cahn(model, problem_config, out_dir, pools):
    """Measure held-out residual, hard traces, and descriptive energy signals."""
    config = dict(problem_config or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diffusion = float(config.get("diffusion", 0.001))
    reaction = float(config.get("reaction", 5.0))
    residual_scale = float(config.get("residual_scale", 6.001))
    chunk_size = int(config.get("evaluation_chunk_size", 1024))
    if chunk_size < 1 or residual_scale <= 0:
        raise ValueError("evaluation chunk size and residual scale must be positive")

    heldout = np.asarray(pools["heldout"], dtype=np.float64)
    if heldout.ndim != 2 or heldout.shape[1] != 2 or not len(heldout):
        raise ValueError("heldout must be a nonempty (n,2) point matrix")
    heldout_fields = _fields(model, heldout, diffusion, reaction, chunk_size)
    score = residual_score(
        heldout_fields["residual"], scale=residual_scale, equation="allen_cahn_1d"
    )
    if score.value is None:
        raise ValueError(f"held-out residual grader returned no score: {score.reason}")
    absolute_residual = np.abs(heldout_fields["residual"])
    normalized_absolute = absolute_residual / residual_scale

    boundary = np.asarray(pools["boundary_probe"], dtype=np.float64)
    initial = np.asarray(pools["initial_probe"], dtype=np.float64)
    boundary_fields = _fields(model, boundary, diffusion, reaction, chunk_size)["u"]
    initial_fields = _fields(model, initial, diffusion, reaction, chunk_size)["u"]
    initial_target = np.square(initial[:, 0]) * np.cos(np.pi * initial[:, 0])

    grid = dict(config.get("diagnostic_grid", {}))
    nx, nt = int(grid.get("nx", 257)), int(grid.get("nt", 129))
    if nx < 3 or nt < 2:
        raise ValueError("diagnostic grid must have nx >= 3 and nt >= 2")
    xs = np.linspace(-1.0, 1.0, nx)
    ts = np.linspace(0.0, 1.0, nt)
    xx, tt = np.meshgrid(xs, ts, indexing="ij")
    grid_points = np.column_stack((xx.reshape(-1), tt.reshape(-1)))
    grid_fields = _fields(model, grid_points, diffusion, reaction, chunk_size)
    prediction = grid_fields["u"].reshape(nx, nt)
    grid_residual = grid_fields["residual"].reshape(nx, nt)

    energy_points = int(config.get("energy_points", 513))
    if energy_points < 3 or energy_points % 2 == 0:
        raise ValueError("energy_points must be an odd integer >= 3 to include x=0 symmetrically")
    energy_x = np.linspace(-1.0, 1.0, energy_points)
    configured_times = [float(time) for time in config.get("evaluation_times", [0.25, 0.5, 0.75, 1.0])]
    energy_times = sorted(set([0.0, *configured_times]))
    if any(time < 0.0 or time > 1.0 for time in energy_times):
        raise ValueError("energy evaluation times must lie in [0,1]")
    energy_rows = []
    energy_u = []
    energy_ux = []
    for time in energy_times:
        coordinates = np.column_stack((energy_x, np.full_like(energy_x, time)))
        fields = _fields(model, coordinates, diffusion, reaction, chunk_size)
        energy = allen_cahn_energy(energy_x, fields["u"], fields["ux"], diffusion, reaction)
        energy_u.append(fields["u"])
        energy_ux.append(fields["ux"])
        energy_rows.append({
            "time": time,
            "energy": energy,
            "u_min": float(fields["u"].min()),
            "u_max": float(fields["u"].max()),
        })
    energies = np.asarray([row["energy"] for row in energy_rows], dtype=np.float64)
    initial_energy = max(abs(float(energies[0])), np.finfo(np.float64).tiny)

    metrics = {
        "normalized_residual_mean": float(score.value),
        "heldout_residual_rmse_abs": float(np.sqrt(np.mean(np.square(heldout_fields["residual"])))),
        "heldout_residual_abs_p50": float(np.quantile(absolute_residual, 0.50)),
        "heldout_residual_abs_p95": float(np.quantile(absolute_residual, 0.95)),
        "heldout_residual_abs_max": float(absolute_residual.max()),
        "heldout_normalized_residual_rmse": float(np.sqrt(np.mean(np.square(normalized_absolute)))),
        "heldout_normalized_residual_abs_p95": float(np.quantile(normalized_absolute, 0.95)),
        "heldout_point_count": int(len(heldout)),
        "initial_trace_max_abs_error": float(np.max(np.abs(initial_fields - initial_target))),
        "boundary_trace_max_abs_error": float(np.max(np.abs(boundary_fields + 1.0))),
        "energy_initial": float(energies[0]),
        "energy_final": float(energies[-1]),
        "energy_change": float(energies[-1] - energies[0]),
        "energy_relative_change": float((energies[-1] - energies[0]) / initial_energy),
        "energy_increase_count": int(np.count_nonzero(np.diff(energies) > 0)),
        "energy_interval_count": int(max(0, len(energies) - 1)),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Allen–Cahn exploratory metrics contain non-finite values")

    np.savez_compressed(
        out_dir / "validation.npz",
        heldout=heldout,
        heldout_u=heldout_fields["u"],
        heldout_ux=heldout_fields["ux"],
        heldout_ut=heldout_fields["ut"],
        heldout_uxx=heldout_fields["uxx"],
        heldout_residual=heldout_fields["residual"],
        x=xs,
        t=ts,
        prediction=prediction,
        residual=grid_residual,
        energy_x=energy_x,
        energy_times=np.asarray(energy_times),
        energy=np.asarray(energy_u),
        energy_ux=np.asarray(energy_ux),
        final_test_used=np.asarray(False),
        field_reference_used=np.asarray(False),
        scientific_accepted=np.asarray(False),
    )
    with (out_dir / "energy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time", "energy", "u_min", "u_max"])
        writer.writeheader()
        writer.writerows(energy_rows)
    _plot_fields(out_dir, xs, ts, prediction, grid_residual)
    _plot_energy(out_dir, energy_rows)

    report = {
        "problem": "allen_cahn_1d",
        "phase": "exploratory",
        "heldout_sampling": "independent_scrambled_sobol",
        "heldout_point_count": int(len(heldout)),
        "field_error_status": "locked_pending_reference",
        "field_reference_used": False,
        "final_test_used": False,
        "scientific_accepted": False,
        "residual_score": {
            "name": score.name,
            "value": float(score.value),
            "direction": score.direction,
            "scope": score.scope,
            "n": int(score.n),
            "reason": score.reason,
        },
        "metrics": metrics,
        "energy": energy_rows,
    }
    json_write(out_dir / "validation.json", report)
    return metrics
