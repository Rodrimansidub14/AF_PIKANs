"""Remote DFG campaign runner and quantitative diagnostics for the 2D-1 Re20 case."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc

from src.sciml.dfg import (
    CYLINDER_CENTER,
    CYLINDER_RADIUS,
    DOMAIN_X,
    DOMAIN_Y,
    DFGConfig,
    DFGRBFPIKAN,
    FEATFLOW_REFERENCE,
    FEATFLOW_URL,
    NU,
    PRESSURE_SCALE,
    U_MEAN,
    U_MAX,
    evaluate_dfg_qois,
    generate_dfg_pools,
    resolve_device,
    steady_ns_residual,
    train_dfg,
)


def flatten_finite_metrics(values: Mapping[str, Any], prefix: str = "") -> dict[str, float | int]:
    """Flatten nested results into the strict finite-numeric harness contract."""
    flattened: dict[str, float | int] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_finite_metrics(value, name))
        elif type(value) in (int, float):
            if not math.isfinite(float(value)):
                raise ValueError(f"Metric {name} is non-finite")
            flattened[name] = value
    return flattened


def summarize_residual_regions(
    points: np.ndarray,
    residuals: Mapping[str, np.ndarray],
    *,
    near_distance: float = 0.15,
) -> dict[str, dict[str, float | int | None]]:
    """Quantify normalized PDE residuals globally and in fixed geometric regions."""
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0 or not np.isfinite(xy).all():
        raise ValueError("points must be a non-empty finite array with shape (n,2)")
    if not math.isfinite(near_distance) or near_distance <= CYLINDER_RADIUS:
        raise ValueError("near_distance must be finite and greater than the cylinder radius")
    values: dict[str, np.ndarray] = {}
    for name, array in residuals.items():
        vector = np.asarray(array, dtype=np.float64).reshape(-1)
        if vector.size != xy.shape[0]:
            raise ValueError("Residual arrays must have the same number of rows as points")
        if not np.isfinite(vector).all():
            raise ValueError("Residual arrays must be finite")
        values[str(name)] = vector
    if not values:
        raise ValueError("At least one residual component is required")

    distance = np.hypot(xy[:, 0] - CYLINDER_CENTER[0], xy[:, 1] - CYLINDER_CENTER[1])
    masks = {
        "global": np.ones(xy.shape[0], dtype=bool),
        "near_cylinder": distance <= near_distance,
        "wake": (xy[:, 0] > CYLINDER_CENTER[0] + CYLINDER_RADIUS)
        & (xy[:, 0] <= 0.8)
        & (np.abs(xy[:, 1] - CYLINDER_CENTER[1]) <= 0.05),
        "far_field": xy[:, 0] > 0.8,
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for region, mask in masks.items():
        count = int(np.count_nonzero(mask))
        summary: dict[str, float | int | None] = {"count": count}
        for name, vector in values.items():
            if count:
                selected = vector[mask]
                summary[f"{name}_rms"] = float(np.sqrt(np.mean(np.square(selected))))
                summary[f"{name}_mean_abs"] = float(np.mean(np.abs(selected)))
                summary[f"{name}_p95_abs"] = float(np.quantile(np.abs(selected), 0.95))
            else:
                summary[f"{name}_rms"] = None
                summary[f"{name}_mean_abs"] = None
                summary[f"{name}_p95_abs"] = None
        result[region] = summary
    return result


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_checkpoint(model: DFGRBFPIKAN, checkpoint_path: Path, device: torch.device) -> DFGRBFPIKAN:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _evaluate_heldout(model: DFGRBFPIKAN, device: torch.device, points: np.ndarray, chunk_size: int = 1024):
    residual_arrays: dict[str, list[np.ndarray]] = {"momentum_x": [], "momentum_y": [], "continuity": []}
    elapsed = 0.0
    for start in range(0, len(points), chunk_size):
        xy = torch.as_tensor(points[start : start + chunk_size], dtype=torch.float32, device=device)
        xy.requires_grad_(True)
        tick = time.perf_counter()
        with torch.enable_grad():
            residuals = steady_ns_residual(model(xy), xy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - tick
        for name in residual_arrays:
            residual_arrays[name].append(residuals[name].detach().cpu().numpy().reshape(-1))
        del xy, residuals
    return {name: np.concatenate(chunks) for name, chunks in residual_arrays.items()}, elapsed


def _boundary_signals(model: DFGRBFPIKAN, pools: Mapping[str, np.ndarray], device: torch.device) -> dict[str, float]:
    results: dict[str, float] = {}
    for name, points in pools.items():
        xy = torch.as_tensor(points, dtype=torch.float32, device=device)
        if name == "outlet":
            xy = xy.detach().requires_grad_(True)
            with torch.enable_grad():
                fields = model(xy)
                grad_u = torch.autograd.grad(fields[:, 0].sum(), xy, create_graph=False, retain_graph=True)[0]
                grad_v = torch.autograd.grad(fields[:, 1].sum(), xy, create_graph=False)[0]
            traction_x = (NU * grad_u[:, 0] - fields[:, 2]) / PRESSURE_SCALE
            traction_y = NU * grad_v[:, 0] / PRESSURE_SCALE
            results["boundary/outlet_normal_traction_rms"] = float(
                torch.sqrt(torch.mean(traction_x.square() + traction_y.square())).detach().cpu()
            )
            continue
        predicted = model(xy)[:, :2]
        if name == "inlet":
            y = xy[:, 1:2]
            u = 4.0 * U_MAX * y * (DOMAIN_Y[1] - y) / (DOMAIN_Y[1] - DOMAIN_Y[0]) ** 2
            target = torch.cat((u, torch.zeros_like(u)), dim=1)
        else:
            target = torch.zeros_like(predicted)
        error = (predicted - target) / U_MEAN
        results[f"boundary/{name}_velocity_rms_normalized"] = float(
            torch.sqrt(torch.mean(error.square())).detach().cpu()
        )
    return results


def _basis_signals(model: DFGRBFPIKAN, points: np.ndarray, device: torch.device) -> dict[str, float]:
    captured: list[torch.Tensor] = []
    hooks = [layer.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach()))
             for layer in model.layers]
    sample = torch.as_tensor(points[: min(2048, len(points))], dtype=torch.float32, device=device)
    with torch.no_grad():
        model(sample)
    for hook in hooks:
        hook.remove()
    signals: dict[str, float] = {}
    for index, (layer, inputs) in enumerate(zip(model.layers, captured)):
        activations = torch.exp(-((inputs.unsqueeze(-1) - layer.centers) / layer.width).square())
        mean_activation = activations.mean(dim=(0, 1))
        signals[f"basis/layer_{index}_active_fraction"] = float((mean_activation > 0.05).float().mean().cpu())
        signals[f"basis/layer_{index}_mean_activation"] = float(mean_activation.mean().cpu())
        signals[f"basis/layer_{index}_coefficient_l2"] = float(torch.linalg.vector_norm(layer.coefficients).cpu())
        signals[f"basis/layer_{index}_coefficient_abs_max"] = float(layer.coefficients.detach().abs().max().cpu())
    return signals


def _plot_history(out: Path) -> None:
    import pandas as pd
    import plotly.express as px

    history = pd.read_csv(out / "training_history.csv")
    loss_names = {
        "total_loss", "momentum_x", "momentum_y", "continuity", "inlet",
        "lower_wall", "upper_wall", "cylinder", "outlet",
    }
    losses = [name for name in history.columns if name in loss_names]
    figure = px.line(history, x="step", y=losses, log_y=True, title="DFG Re20 training losses (log scale)")
    figure.update_layout(xaxis_title="Optimizer step", yaxis_title="Normalized loss")
    figure.write_html(out / "dfg_training_losses.html", include_plotlyjs="cdn")


def _save_field_visualizations(model: DFGRBFPIKAN, device: torch.device, out: Path) -> dict[str, Any]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pyvista as pv

    x = np.linspace(DOMAIN_X[0], DOMAIN_X[1], 221)
    y = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], 42)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy_np = np.column_stack((xx.ravel(), yy.ravel()))
    predictions = []
    with torch.no_grad():
        for start in range(0, len(xy_np), 4096):
            xy = torch.as_tensor(xy_np[start : start + 4096], dtype=torch.float32, device=device)
            predictions.append(model(xy).detach().cpu().numpy())
    field = np.concatenate(predictions)
    inside = np.hypot(xy_np[:, 0] - CYLINDER_CENTER[0], xy_np[:, 1] - CYLINDER_CENTER[1]) < CYLINDER_RADIUS
    arrays = {
        "u": field[:, 0].reshape(len(y), len(x)),
        "v": field[:, 1].reshape(len(y), len(x)),
        "p": field[:, 2].reshape(len(y), len(x)),
    }
    mask = inside.reshape(len(y), len(x))
    for array in arrays.values():
        array[mask] = np.nan

    figure = make_subplots(rows=1, cols=3, subplot_titles=("u velocity", "v velocity", "pressure"))
    for index, (name, values) in enumerate(arrays.items(), 1):
        figure.add_trace(
            go.Contour(x=x, y=y, z=values, colorbar={"title": name}, contours={"coloring": "heatmap"}),
            row=1, col=index,
        )
        figure.add_shape(type="circle", x0=0.15, y0=0.15, x1=0.25, y1=0.25,
                         line={"color": "black", "width": 2}, row=1, col=index)
    figure.update_layout(title="RBF-PIKAN DFG Re20 field; cylinder interior masked", height=520)
    figure.write_html(out / "dfg_fields.html", include_plotlyjs="cdn")

    grid = pv.RectilinearGrid(x, y, np.array([0.0]))
    for name, values in arrays.items():
        grid.point_data[name] = values.ravel(order="C")
    grid.point_data["inside_cylinder"] = mask.astype(np.uint8).ravel(order="C")
    grid.set_active_scalars("p")
    grid.save(out / "dfg_fields.vtr")
    return {
        "grid_shape": [len(y), len(x)],
        "grid_points": int(len(xy_np)),
        "masked_inside_cylinder": int(np.count_nonzero(inside)),
        "field_file": "dfg_fields.vtr",
    }


def _log_wandb(
    out: Path,
    flat_metrics: Mapping[str, float | int],
    manifest: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    import wandb

    tracking = manifest.get("tracking", {})
    run = wandb.init(
        entity=tracking.get("entity"),
        project=tracking.get("project", "AF_PIKANS"),
        group=tracking.get("group", "baseline-close-dfg-20260927"),
        name=f"dfg-re20-rbf-sobol-seed-{seed}",
        mode="offline",
        dir=str(out / "wandb"),
        config={"seed": seed, **manifest["training"], "problem_id": "dfg_2d_1_steady_re20"},
    )
    run.log(dict(flat_metrics), step=0)
    artifact = wandb.Artifact(f"dfg-re20-seed-{seed}", type="exploratory-baseline")
    for name in ("metrics.json", "sampling_signals.json", "diagnostic_signals.json", "dfg_training_losses.html"):
        artifact.add_file(str(out / name), name=name)
    run.log_artifact(artifact)
    run.finish()
    return {
        "mode": "offline",
        "run_id": run.id,
        "run_name": run.name,
        "project": tracking.get("project", "AF_PIKANS"),
        "entity": tracking.get("entity"),
        "artifact": artifact.name,
        "local_run_dir": str(out / "wandb"),
    }


def run_campaign(manifest: Mapping[str, Any], out: Path, seed: int) -> dict[str, Any]:
    """Run one seed, held-out measurement, visualizations, and offline tracking."""
    training = manifest["training"]
    problem = manifest["problem"]["dfg"]
    config = DFGConfig(
        seed=seed,
        interior_count=int(training["interior_points"]),
        boundary_count=int(training["boundary_points"]),
        batch_size=int(training["batch_size"]),
        boundary_batch_size=int(training["boundary_batch_size"]),
        learning_rate=float(training["learning_rate"]),
        max_steps=int(training["max_steps"]),
        time_limit_s=float(training["seconds_per_seed"]),
        device="cuda",
    )
    device = resolve_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("DFG remote baseline requires CUDA; CPU fallback is disabled")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = train_dfg(config, out)
    nested_metrics = result["metrics"]
    model = _load_checkpoint(DFGRBFPIKAN(seed=seed).to(device), out / "dfg_rbf_pikan.pt", device)

    heldout_count = int(problem["heldout_points"])
    heldout_seed = (seed + int(problem["heldout_seed_offset"])) % (2**32 - 1)
    heldout_pools, heldout_sampling = generate_dfg_pools(
        heldout_seed,
        interior_count=heldout_count,
        boundary_count=int(problem["heldout_boundary_points"]),
        return_provenance=True,
    )
    heldout_points = heldout_pools["interior"]
    np.savez_compressed(
        out / "dfg_heldout_points.npz",
        interior=heldout_points,
        heldout_seed=np.asarray([heldout_seed], dtype=np.int64),
    )
    residual_arrays, autodiff_seconds = _evaluate_heldout(model, device, heldout_points)
    regional_residuals = summarize_residual_regions(
        heldout_points,
        residual_arrays,
        near_distance=float(problem["near_cylinder_distance"]),
    )
    residual_path = out / "dfg_heldout_residuals.npz"
    np.savez_compressed(residual_path, points=heldout_points, **residual_arrays)

    heldout_unit = heldout_points.copy()
    heldout_unit[:, 0] /= DOMAIN_X[1] - DOMAIN_X[0]
    heldout_unit[:, 1] /= DOMAIN_Y[1] - DOMAIN_Y[0]
    neighbor = cKDTree(heldout_unit).query(heldout_unit, k=2)[0][:, 1]
    sampling = result["run"]["sampling"]
    sampling_signals = {
        "method": "fixed_scrambled_sobol_rejection_interior_independent_sobol_boundaries",
        "seed": seed,
        "interior_count": config.interior_count,
        "boundary_count_each_segment": config.boundary_count,
        "heldout_count": int(len(heldout_points)),
        "heldout_seed": heldout_seed,
        "interior_candidate_count": sampling["interior"]["candidate_count"],
        "interior_rejected_fraction": sampling["interior"]["rejected_count"]
        / sampling["interior"]["candidate_count"],
        "training_pool_sha256": sampling["pool_sha256"],
        "heldout_pool_sha256": heldout_sampling["pool_sha256"]["interior"],
        "heldout_centered_discrepancy": float(qmc.discrepancy(heldout_unit, method="CD")),
        "heldout_nn_min_normalized": float(np.min(neighbor)),
        "heldout_nn_median_normalized": float(np.median(neighbor)),
        "heldout_nn_p95_normalized": float(np.quantile(neighbor, 0.95)),
        "heldout_nn_max_normalized": float(np.max(neighbor)),
    }
    _write_json(out / "sampling_signals.json", sampling_signals)

    boundary = _boundary_signals(model, heldout_pools, device)
    qois = evaluate_dfg_qois(model, device, surface_count=int(problem["qoi_surface_points"]))
    basis = _basis_signals(model, heldout_points, device)
    seconds = float(result["run"]["training_seconds"])
    steps = int(result["run"]["training_steps"])
    autodiff_rate = len(heldout_points) / max(autodiff_seconds, np.finfo(float).tiny)
    signal_data = {
        "problem": "dfg_2d_1_steady_re20",
        "seed": seed,
        "device": str(device),
        "training_steps": steps,
        "training_seconds": seconds,
        "seconds_per_optimizer_step": seconds / max(steps, 1),
        "autodiff_heldout_seconds": autodiff_seconds,
        "autodiff_seconds_per_point": autodiff_seconds / len(heldout_points),
        "autodiff_points_per_second": autodiff_rate,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "residual_evaluation_points": len(heldout_points),
        "regional_normalized_residuals": regional_residuals,
        "boundary": boundary,
        "qoi": qois,
        "qoi_reference": FEATFLOW_REFERENCE,
        "qoi_absolute_error_vs_featflow": nested_metrics["qoi_absolute_error_vs_featflow"],
        "qoi_relative_error_vs_featflow": nested_metrics["qoi_relative_error_vs_featflow"],
        "field_relative_l2": None,
        "field_reference_status": "not_available_in_this_runner",
        "basis": basis,
        "optimizer_final_losses": nested_metrics["last_training_losses"],
        "scientific_accepted": False,
    }
    _write_json(out / "diagnostic_signals.json", signal_data)
    _write_json(out / "dfg_metrics_nested.json", nested_metrics)
    _plot_history(out)
    field_visualization = _save_field_visualizations(model, device, out)

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
    }
    for distribution in ("pyvista", "vtk", "plotly", "pandas", "wandb"):
        versions[distribution] = importlib.metadata.version(distribution)
    provenance = {
        "schema_version": 1,
        "problem_id": "dfg_2d_1_steady_re20",
        "reference_source": FEATFLOW_URL,
        "reference_values": FEATFLOW_REFERENCE,
        "field_reference": None,
        "model": {
            "id": DFGRBFPIKAN.model_id,
            "architecture": list(DFGRBFPIKAN.architecture),
            "basis_count": DFGRBFPIKAN.basis_count,
            "dtype": "float32",
        },
        "sampling_seed_stream": {"training": seed, "heldout": heldout_seed},
        "heldout_residual_archive_sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
        "versions": versions,
        "visualization": field_visualization,
        "total_wall_seconds": time.perf_counter() - started,
        "scientific_accepted": False,
    }
    _write_json(out / "provenance.json", provenance)

    values = {
        "qoi": qois,
        "qoi_abs_error": nested_metrics["qoi_absolute_error_vs_featflow"],
        "qoi_rel_error": nested_metrics["qoi_relative_error_vs_featflow"],
        "heldout": regional_residuals,
        "boundary": boundary,
        "sampling": {key: value for key, value in sampling_signals.items() if type(value) in (int, float)},
        "training": {
            "steps": steps,
            "seconds": seconds,
            "seconds_per_step": seconds / max(steps, 1),
        },
        "autodiff": {
            "seconds": autodiff_seconds,
            "seconds_per_point": autodiff_seconds / len(heldout_points),
            "points_per_second": autodiff_rate,
        },
        "cuda": {
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "basis": basis,
    }
    flat_metrics = flatten_finite_metrics(values)
    if steps < 1:
        raise RuntimeError("DFG baseline executed zero optimizer steps")
    _write_json(out / "metrics.json", flat_metrics)
    _write_json(
        out / "config.json",
        {"training": training, "problem": problem, "seed": seed, "model": manifest["model"]},
    )
    _write_json(
        out / "hardware.json",
        {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    )
    _write_json(out / "versions.json", versions)
    with (out / "quantitative_signals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["signal", "value"])
        writer.writeheader()
        writer.writerows({"signal": name, "value": value} for name, value in sorted(flat_metrics.items()))
    note = (
        "# DFG Re20 fixed-RBF-PIKAN exploratory baseline\n\n"
        f"Seed: {seed}. Training uses fixed scrambled-Sobol/rejection pools and a {seconds:.1f}s cap.\n\n"
        f"FEATFLOW targets are recorded at {FEATFLOW_URL}; this run reports integral QoI errors and independent held-out normalized residuals.\n\n"
        "The cylinder geometry, wall/inlet/no-slip conditions and natural do-nothing outlet follow FEATFLOW. "
        "The runner uses direct PyTorch autodifferentiation for the natural outlet traction and integrated cylinder forces. "
        "Field-relative L2 remains unavailable because no independent converged FEM field is packaged. "
        "W&B is offline. scientific_accepted=false; this is exploratory evidence and makes no SOTA claim.\n"
    )
    (out / "decision_note.md").write_text(note, encoding="utf-8")
    tracking = _log_wandb(out, flat_metrics, manifest, seed)
    _write_json(out / "wandb_status.json", tracking)
    return {"metrics": flat_metrics, "tracking": tracking}


def main() -> int:
    manifest_path = Path(os.environ["HARNESS_MANIFEST"])
    output = Path(os.environ["HARNESS_RUN_DIR"])
    seed = int(os.environ["HARNESS_SEED"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    run_campaign(manifest, output, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
