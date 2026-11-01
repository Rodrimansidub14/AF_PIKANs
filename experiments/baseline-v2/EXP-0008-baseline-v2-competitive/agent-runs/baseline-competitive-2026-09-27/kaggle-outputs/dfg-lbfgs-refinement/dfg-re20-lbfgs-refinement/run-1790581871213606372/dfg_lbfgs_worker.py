#!/usr/bin/env python3
"""Full-pool L-BFGS continuation of an existing DFG Re=20 checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("dfg_refinement_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load DFG runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("DFG-RBF-PIKAN", "DFG-Tanh-MLP"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=480.0)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    runner = load_runner(args.runner)
    args.output.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = runner.build_model(args.arm, args.seed).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.train()

    with np.load(args.pool, allow_pickle=False) as archive:
        pools = {key: np.asarray(archive[key]) for key in archive.files}
    needed = ("interior", "inlet", "lower_wall", "upper_wall", "cylinder", "outlet")
    missing = [key for key in needed if key not in pools]
    if missing:
        raise RuntimeError(f"Saved DFG training pool is missing {missing}")

    tensors = {
        key: torch.as_tensor(pools[key], dtype=torch.float32, device=device)
        for key in needed
    }
    interior_count = int(tensors["interior"].shape[0])
    boundary_count = int(tensors["inlet"].shape[0])
    started = time.perf_counter()
    deadline = started + max(1.0, float(args.seconds))
    step = 0
    history: list[dict] = []
    last_loss = None
    stop_reason = "time_cap"

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=4,
        tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        nonlocal last_loss
        optimizer.zero_grad(set_to_none=True)
        total_value = 0.0

        for start in range(0, interior_count, args.chunk_size):
            if time.perf_counter() >= deadline:
                raise TimeoutError("per-arm L-BFGS time cap reached inside deterministic closure")
            xy = tensors["interior"][start:start + args.chunk_size].detach().requires_grad_(True)
            residuals = runner.residual_components(model(xy), xy)
            weight = xy.shape[0] / interior_count
            piece = sum(value.square().mean() for value in residuals.values()) * weight
            if not torch.isfinite(piece):
                raise FloatingPointError("non-finite interior residual loss")
            piece.backward()
            total_value += float(piece.detach().cpu())

        inlet = tensors["inlet"]
        y = inlet[:, 1:2]
        inlet_u = 4.0 * runner.U_MAX * y * (runner.H - y) / (runner.H * runner.H)
        inlet_target = torch.cat((inlet_u, torch.zeros_like(inlet_u)), dim=1)
        for name, target in (
            ("inlet", inlet_target),
            ("lower_wall", torch.zeros((boundary_count, 2), device=device)),
            ("upper_wall", torch.zeros((boundary_count, 2), device=device)),
            ("cylinder", torch.zeros((boundary_count, 2), device=device)),
        ):
            value = ((model(tensors[name])[:, :2] - target) / runner.U_MEAN).square().mean()
            if not torch.isfinite(value):
                raise FloatingPointError(f"non-finite {name} loss")
            value.backward()
            total_value += float(value.detach().cpu())

        outlet = tensors["outlet"].detach().requires_grad_(True)
        fields = model(outlet)
        grad_u = runner.grad_scalar(fields[:, 0:1], outlet)
        grad_v = runner.grad_scalar(fields[:, 1:2], outlet)
        traction_x = (runner.NU * grad_u[:, 0:1] - fields[:, 2:3]) / runner.PRESSURE_SCALE
        traction_y = runner.NU * grad_v[:, 0:1] / runner.PRESSURE_SCALE
        outlet_loss = traction_x.square().mean() + traction_y.square().mean()
        if not torch.isfinite(outlet_loss):
            raise FloatingPointError("non-finite outlet traction loss")
        outlet_loss.backward()
        total_value += float(outlet_loss.detach().cpu())
        last_loss = total_value
        return torch.as_tensor(total_value, dtype=torch.float32, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        while time.perf_counter() < deadline:
            value = optimizer.step(closure)
            torch.cuda.synchronize(device)
            step += 1
            loss_value = last_loss if last_loss is not None else float(value.detach().cpu())
            history.append({"step": step, "loss": loss_value, "elapsed_seconds": time.perf_counter() - started})
            if not np.isfinite(loss_value):
                stop_reason = "nonfinite_loss"
                break
        else:
            stop_reason = "time_cap"
    except TimeoutError:
        stop_reason = "time_cap_during_closure"
    except FloatingPointError as exc:
        stop_reason = "nonfinite_loss"
        error_text = str(exc)
    except Exception as exc:
        stop_reason = "optimizer_error"
        error_text = f"{type(exc).__name__}: {exc}"

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output_checkpoint = args.output / "checkpoint-lbfgs.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "arm": args.arm,
            "seed": args.seed,
            "lbfgs_steps": step,
            "source_checkpoint": str(args.checkpoint),
            "stage": "deterministic_full_pool_LBFGS_continuation",
        },
        output_checkpoint,
    )
    with (args.output / "lbfgs_history.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {
        "status": "lbfgs_refined" if step > 0 else stop_reason,
        "arm": args.arm,
        "seed": args.seed,
        "stage": "deterministic_full_pool_LBFGS_continuation",
        "lbfgs_steps": step,
        "elapsed_seconds": elapsed,
        "seconds_cap": args.seconds,
        "stop_reason": stop_reason,
        "device": torch.cuda.get_device_name(device),
        "cuda_index_visible_to_worker": 0,
        "dtype": "float32",
        "interior_points": interior_count,
        "boundary_points_per_set": boundary_count,
        "chunk_size": args.chunk_size,
        "last_loss": last_loss,
        "checkpoint": str(output_checkpoint),
        "source_checkpoint": str(args.checkpoint),
        "pool_file": str(args.pool),
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "scientific_accepted": False,
        "field_relative_l2": None,
        "field_reference_status": "missing_converged_FEM_field",
    }
    if "error_text" in locals():
        summary["error"] = error_text
    write_json(args.output / "lbfgs_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
