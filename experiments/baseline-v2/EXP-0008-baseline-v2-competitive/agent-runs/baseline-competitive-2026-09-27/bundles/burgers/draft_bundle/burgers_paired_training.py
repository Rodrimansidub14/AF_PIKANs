"""Torch-only Burgers trainer and common post-training Cole-Hopf scorer."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

POOL_KEYS = ("interior", "diagnostic", "left", "right", "initial")
TIME_MARKS = (120, 300, 600, 900)
BATCH_SIZE = 1024
NU = 0.01 / np.pi
RESIDUAL_SCALE = 2.003183098861838


class VanillaMLP(nn.Module):
    model_id = "vanilla_mlp_tanh_5x100"
    model_family = "mlp"
    problem = "burgers"
    hidden_layers = 5
    width = 100

    def __init__(self):
        super().__init__()
        widths = [2] + [self.width] * self.hidden_layers + [1]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        for layer in self.layers[:-1]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, points):
        if hasattr(points, "tensor"):
            points = points.tensor
        values = torch.cat((points[:, 0:1], 2.0 * points[:, 1:2] - 1.0), dim=1)
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index < len(self.layers) - 1:
                values = torch.tanh(values)
        return values


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    values = np.asarray(array)
    header = json.dumps({"dtype": values.dtype.str, "shape": list(values.shape)}, sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def hashes_for(data: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: array_hash(data[name]) for name in POOL_KEYS}


def _save_checkpoint(model, out: Path, model_id: str, model_family: str, seed: int,
                     step: int, elapsed: float, requested: int | str | None,
                     pool_hash: str, problem_config: dict) -> Path:
    if model_id == "rbf_fixed":
        architecture = {"basis_count": 11, "layers": [2, 16, 16, 1]}
    else:
        architecture = {"hidden_layers": 5, "width": 100}
    payload = {
        "problem": "burgers", "validation_problem": "burgers_pinnacle_1d",
        "model_id": model_id, "model_family": model_family,
        "problem_parameters": problem_config, "architecture": architecture,
        "step": int(step), "seed": int(seed),
        "elapsed_gpu_seconds": float(elapsed), "requested_gpu_seconds": requested,
        "pool_content_sha256": pool_hash,
        "optimizer": {"name": "Adam", "learning_rate": 1e-3},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    target = out / (f"checkpoint-gpu-{int(requested):04d}s.pt" if isinstance(requested, int) else "checkpoint-final.pt")
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def train_one(kind: str, seed: int, data: dict[str, np.ndarray], out: str | Path,
              seconds: int = 900, max_steps: int = 20000,
              pool_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train one arm with common fixed pools, Adam, and one visible CUDA device."""
    if kind not in ("rbf", "vanilla"):
        raise ValueError("kind must be 'rbf' or 'vanilla'")
    if not 0 < seconds <= 900 or not 0 < max_steps <= 20000:
        raise ValueError("caps exceed the proposed per-arm limits")
    if set(data) != set(POOL_KEYS):
        raise ValueError(f"training pools must contain exactly {POOL_KEYS}")
    shapes = {"interior": (8192, 2), "diagnostic": (2048, 2), "left": (512, 2),
              "right": (512, 2), "initial": (1024, 2)}
    for name, shape in shapes.items():
        values = np.asarray(data[name])
        if values.shape != shape or values.dtype != np.dtype("float64") or not np.isfinite(values).all():
            raise ValueError(f"{name} pool must be finite float64 with shape {shape}")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one exclusively assigned CUDA device is required")
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    if kind == "rbf":
        from src.research_pipeline.sciml.model import build_model
        model = build_model("burgers", {"id": "rbf_fixed", "basis_count": 11})
        model_id, family = "rbf_fixed", "rbf"
    else:
        model = VanillaMLP()
        model_id, family = VanillaMLP.model_id, VanillaMLP.model_family
    model = model.to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    problem_config = {"definition": "raissi_pinnacle_1d", "nu": NU,
                      "field_scale": 1.0, "residual_scale": RESIDUAL_SCALE,
                      "evaluation_id": "burgers_pinnacle_1d"}
    pool_hashes = hashes_for(data)
    pool_content_hash = hashlib.sha256(json.dumps(pool_hashes, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    pool_path = output / "pools.npz"
    np.savez_compressed(pool_path, **data)
    sampling_source = Path(__file__).resolve().parents[2] / "sciml" / "sampling.py"
    config = {
        "seed": int(seed), "problem": "burgers", "validation_problem": "burgers_pinnacle_1d",
        "domain": {"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        "initial_condition": "-sin(pi*x)", "dirichlet_boundary": 0.0, "nu": NU,
        "model": {"id": model_id, "family": family, "architecture": [2, 100, 100, 100, 100, 100, 1] if kind == "vanilla" else [2, 16, 16, 1],
                  "activation": "tanh" if kind == "vanilla" else "tanh-hidden/silu-base", "parameters": sum(p.numel() for p in model.parameters())},
        "optimizer": {"name": "Adam", "learning_rate": 1e-3, "betas": [0.9, 0.999]},
        "batch_size_per_condition": BATCH_SIZE, "batch_sampling": "independent seeded sampling with replacement from the fixed pool",
        "max_updates": int(max_steps), "gpu_seconds_cap": int(seconds), "precision": "float64",
        "loss_contract": {"pde": "(u_t + u*u_x - nu*u_xx)/residual_scale", "condition_weights": "sum of four condition mean-square losses",
                          "residual_scale": RESIDUAL_SCALE, "initial": "-sin(pi*x)", "left_right": "u=0"},
        "pool": {"counts": {name: int(len(data[name])) for name in POOL_KEYS}, "array_sha256": pool_hashes,
                 "content_sha256": pool_content_hash, "archive": pool_path.name,
                 "archive_sha256": sha256_file(pool_path), "origin": pool_metadata or {}},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__, "torch": torch.__version__},
        "device": {"name": torch.cuda.get_device_name(device), "visible_device_index": 0,
                   "timer": "synchronized per-update elapsed device wall time"},
        "scoring": "not run by train_one; scoring is a separate post-training command",
    }
    json_write(output / ("run_config.json" if kind == "vanilla" else "arm_config.json"), config)

    tensors = {name: torch.as_tensor(data[name], dtype=torch.float64, device="cpu") for name in POOL_KEYS}
    generators = {}
    for offset, name in enumerate(POOL_KEYS, start=1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(10000 * int(seed) + offset)
        generators[name] = generator
    rows = []
    saved_marks: set[int] = set()
    step = 0
    stop_reason = "step_limit"
    last_losses: dict[str, float] = {}
    started = time.perf_counter()

    def elapsed_now() -> float:
        torch.cuda.synchronize(device)
        return time.perf_counter() - started

    while step < max_steps:
        batch = {}
        for name, tensor in tensors.items():
            indices = torch.randint(len(tensor), (BATCH_SIZE,), generator=generators[name])
            batch[name] = tensor[indices].to(device=device, dtype=torch.float64)
        coordinates = batch["interior"].detach().clone().requires_grad_(True)
        prediction = model(coordinates)
        gradient = torch.autograd.grad(prediction.sum(), coordinates, create_graph=True, retain_graph=True)[0]
        ux, ut = gradient[:, 0:1], gradient[:, 1:2]
        uxx = torch.autograd.grad(ux.sum(), coordinates, create_graph=True, retain_graph=True)[0][:, 0:1]
        physics_loss = ((ut + prediction * ux - NU * uxx) / RESIDUAL_SCALE).square().mean()
        initial_prediction = model(batch["initial"])
        initial_target = -torch.sin(torch.pi * batch["initial"][:, 0:1])
        initial_loss = (initial_prediction - initial_target).square().mean()
        left_loss = model(batch["left"]).square().mean()
        right_loss = model(batch["right"]).square().mean()
        total_loss = physics_loss + initial_loss + left_loss + right_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError("nonfinite gradient")
        optimizer.step()
        step += 1
        elapsed = elapsed_now()
        last_losses = {"train/total_loss": float(total_loss.detach().cpu()),
                       "train/physics_loss": float(physics_loss.detach().cpu()),
                       "train/initial_loss": float(initial_loss.detach().cpu()),
                       "train/left_loss": float(left_loss.detach().cpu()),
                       "train/right_loss": float(right_loss.detach().cpu())}
        for mark in TIME_MARKS:
            if mark <= seconds and mark <= elapsed and mark not in saved_marks:
                _save_checkpoint(model, output, model_id, family, seed, step, elapsed, mark,
                                 pool_content_hash, problem_config)
                saved_marks.add(mark)
        if step % 100 == 0 or elapsed >= seconds or step >= max_steps:
            row = {"step": step, "elapsed_sec": elapsed, **last_losses,
                   "event": "time_limit" if elapsed >= seconds else "step_limit" if step >= max_steps else "periodic"}
            rows.append(row)
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        if elapsed >= seconds:
            stop_reason = "gpu_time_limit"
            break
    elapsed = elapsed_now()
    final = _save_checkpoint(model, output, model_id, family, seed, step, elapsed, None,
                             pool_content_hash, problem_config)
    csv_path = output / "cost.csv"
    keys = sorted({key for row in rows for key in row}) or ["step", "elapsed_sec", "event"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    cost = {
        "seed": int(seed), "model_id": model_id, "completed_updates": int(step),
        "allocated_gpu_seconds": float(elapsed), "gpu_seconds_cap": int(seconds),
        "max_updates_cap": int(max_steps), "stop_reason": stop_reason,
        "cap_overshoot_seconds": max(0.0, elapsed - seconds), "checkpoint": final.name,
        "checkpoint_sha256": sha256_file(final), "available_gpu_time_checkpoints_sec": sorted(saved_marks),
        "pool_content_sha256": pool_content_hash, "pool_archive_sha256": sha256_file(pool_path),
        "device": config["device"], "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_metrics_at_last_record": last_losses, "scoring_performed": False,
    }
    json_write(output / ("cost.json" if kind == "vanilla" else "arm_cost.json"), cost)
    return cost


def score_checkpoint(checkpoint_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Post-training score using the common AF Cole-Hopf grid and metric core."""
    from src.research_pipeline.sciml.model import build_model
    from evals.checkpoint_validation import _autograd_fields, _grid, _trapezoid_tensor_weights
    from evals.problems.burgers_pinnacle_1d import reference

    checkpoint_path = Path(checkpoint_path).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_id = payload.get("model_id")
    if model_id == "rbf_fixed":
        model = build_model("burgers", {"id": "rbf_fixed", "basis_count": 11}).double()
    elif model_id == "vanilla_mlp_tanh_5x100":
        model = VanillaMLP().double()
    else:
        raise ValueError(f"unsupported Burgers checkpoint model: {model_id!r}")
    model.load_state_dict(payload["state_dict"], strict=True)
    xs, ts, points = _grid("burgers_pinnacle_1d")
    prediction, ux, ut, uxx = _autograd_fields(model, points)
    x, t = points[:, 0], points[:, 1]
    truth = reference.exact(x, t, nu=reference.NU_DEFAULT, quadrature_order=128)
    residual = reference.residual(prediction, ut, ux, uxx, nu=reference.NU_DEFAULT)
    weights = _trapezoid_tensor_weights(xs, ts)
    error = prediction - truth
    norm_sq = float(np.sum(weights * truth**2))
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        raise ValueError("common reference norm is invalid")
    rmse = float(np.sqrt(np.sum(weights * error**2) / np.sum(weights)))
    metrics = {
        "field_relative_l2": float(np.sqrt(np.sum(weights * error**2) / norm_sq)),
        "field_rmse_abs": rmse,
        "field_p95_abs": float(np.quantile(np.abs(error), .95)),
        "field_max_abs": float(np.max(np.abs(error))),
        "initial_max_abs_error": float(np.max(np.abs(error[t == 0.0]))),
        "boundary_max_abs_error": float(np.max(np.abs(error[(x == xs[0]) | (x == xs[-1])]))),
        "residual_rmse_abs": float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights))),
        "residual_mean_abs": float(np.sum(weights * np.abs(residual)) / np.sum(weights)),
    }
    front = reference.critical_mask(x, t, nu=reference.NU_DEFAULT)
    for name, mask in (("front", front), ("complement", ~front)):
        metrics[f"{name}_field_rmse_abs"] = float(np.sqrt(np.sum(weights[mask] * error[mask]**2) / np.sum(weights[mask])))
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("common score contains nonfinite metrics")
    nx, nt = len(xs), len(ts)
    np.savez_compressed(out / "validation.npz", x=xs, t=ts,
                        prediction=prediction.reshape(nx, nt), reference=truth.reshape(nx, nt),
                        error=error.reshape(nx, nt), residual=residual.reshape(nx, nt),
                        trapezoid_weights=weights.reshape(nx, nt),
                        scientific_accepted=np.asarray(False))
    report = {
        "problem": "burgers_pinnacle_1d", "grid": {"nx": nx, "nt": nt, "kind": "fixed_uniform_tensor"},
        "reference": "Cole-Hopf/Gauss-Hermite order 128", "step": int(payload["step"]),
        "seed": int(payload["seed"]), "model_id": model_id,
        "checkpoint": checkpoint_path.name, "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics": metrics, "confirmatory_evaluation_invoked": True,
        "scores_used_for_tuning": False, "scientific_accepted": False,
    }
    json_write(out / "score.json", report)
    return report


