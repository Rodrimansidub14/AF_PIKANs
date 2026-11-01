#!/usr/bin/env python3
"""Full-pool L-BFGS continuation for one saved DFG RBF-PIKAN seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def clone_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-kernel", required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--source-pool-sha256", required=True)
    parser.add_argument("--source-run-sha256", required=True)
    parser.add_argument("--source-provenance-sha256", required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--global-device-index", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=540.0)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.module_root.resolve()))
    from src.research_pipeline.sciml.dfg import (  # noqa: E402
        DIAMETER,
        DOMAIN_Y,
        NU,
        PRESSURE_SCALE,
        U_MEAN,
        U_MAX,
        DFGRBFPIKAN,
        steady_ns_residual,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This worker requires exactly one visible CUDA device; CPU fallback is disabled")
    visible_name = torch.cuda.get_device_name(0)
    if "T4" not in visible_name.upper():
        raise RuntimeError(f"Expected a Tesla T4, found {visible_name!r}")
    if visible_name != args.device_name:
        raise RuntimeError(f"Assigned GPU identity changed: expected {args.device_name!r}, found {visible_name!r}")

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    worker_started = time.perf_counter()

    checkpoint_sha = sha256(args.checkpoint)
    pool_sha = sha256(args.pool)
    if checkpoint_sha != args.source_checkpoint_sha256 or pool_sha != args.source_pool_sha256:
        raise RuntimeError("Input checkpoint or pool changed after provenance validation")

    checkpoint_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint_payload, dict) or not isinstance(checkpoint_payload.get("state_dict"), dict):
        raise RuntimeError("RBF checkpoint has no state_dict")
    model = DFGRBFPIKAN(seed=args.seed).to(device)
    model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    model.train()

    with np.load(args.pool, allow_pickle=False) as archive:
        required = ("interior", "inlet", "lower_wall", "upper_wall", "cylinder", "outlet")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise RuntimeError(f"Saved DFG pool is missing arrays: {missing}")
        pool_arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in required}
    for name, values in pool_arrays.items():
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"Saved pool {name!r} must be a non-empty finite (n,2) array")
    tensors = {name: torch.as_tensor(values, dtype=torch.float32, device=device) for name, values in pool_arrays.items()}
    interior_count = int(tensors["interior"].shape[0])
    started = time.perf_counter()
    deadline = started + float(args.seconds)
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
    steps = 0
    last_loss: float | None = None
    last_components: dict[str, float] = {}
    stop_reason = "time_cap"
    error_text: str | None = None
    history: list[dict] = []
    accepted_state = clone_state_to_cpu(model)
    torch.cuda.reset_peak_memory_stats(device)

    def closure() -> torch.Tensor:
        nonlocal last_loss, last_components
        optimizer.zero_grad(set_to_none=True)
        total_value = 0.0
        components = {"pde_momentum_x": 0.0, "pde_momentum_y": 0.0, "pde_continuity": 0.0}

        for start_index in range(0, interior_count, args.chunk_size):
            if time.perf_counter() >= deadline:
                raise TimeoutError("per-seed L-BFGS cap reached inside deterministic full-pool closure")
            end_index = min(start_index + args.chunk_size, interior_count)
            xy = tensors["interior"][start_index:end_index].detach().requires_grad_(True)
            residuals = steady_ns_residual(model(xy), xy)
            weight = (end_index - start_index) / interior_count
            names = (("momentum_x", "pde_momentum_x"), ("momentum_y", "pde_momentum_y"), ("continuity", "pde_continuity"))
            chunk_terms = []
            for residual_name, component_name in names:
                value = residuals[residual_name].square().mean() * weight
                if not torch.isfinite(value):
                    raise FloatingPointError(f"non-finite {component_name} loss")
                chunk_terms.append(value)
                scalar = float(value.detach().cpu())
                components[component_name] += scalar
                total_value += scalar
            chunk_loss = torch.stack(chunk_terms).sum()
            chunk_loss.backward()
            del xy, residuals

        inlet = tensors["inlet"]
        y = inlet[:, 1:2]
        inlet_u = 4.0 * U_MAX * y * (DOMAIN_Y[1] - y) / (DOMAIN_Y[1] - DOMAIN_Y[0]) ** 2
        inlet_target = torch.cat((inlet_u, torch.zeros_like(inlet_u)), dim=1)
        boundary_sets = (
            ("inlet", inlet_target),
            ("lower_wall", torch.zeros_like(tensors["lower_wall"])),
            ("upper_wall", torch.zeros_like(tensors["upper_wall"])),
            ("cylinder", torch.zeros_like(tensors["cylinder"])),
        )
        for name, target in boundary_sets:
            value = ((model(tensors[name])[:, :2] - target) / U_MEAN).square().mean()
            if not torch.isfinite(value):
                raise FloatingPointError(f"non-finite {name} velocity loss")
            value.backward()
            scalar = float(value.detach().cpu())
            components[f"boundary_{name}"] = scalar
            total_value += scalar

        outlet = tensors["outlet"].detach().requires_grad_(True)
        fields = model(outlet)
        grad_u = torch.autograd.grad(
            fields[:, 0].sum(), outlet, create_graph=True, retain_graph=True
        )[0]
        grad_v = torch.autograd.grad(fields[:, 1].sum(), outlet, create_graph=True, retain_graph=True)[0]
        traction_x = (NU * grad_u[:, 0:1] - fields[:, 2:3]) / PRESSURE_SCALE
        traction_y = NU * grad_v[:, 0:1] / PRESSURE_SCALE
        outlet_loss = traction_x.square().mean() + traction_y.square().mean()
        if not torch.isfinite(outlet_loss):
            raise FloatingPointError("non-finite outlet traction loss")
        outlet_loss.backward()
        components["boundary_outlet_traction"] = float(outlet_loss.detach().cpu())
        total_value += components["boundary_outlet_traction"]
        last_components = components
        last_loss = total_value
        return torch.as_tensor(total_value, dtype=torch.float32, device=device)

    try:
        while time.perf_counter() < deadline:
            prior_accepted = accepted_state
            try:
                optimizer.step(closure)
                torch.cuda.synchronize(device)
            except TimeoutError:
                model.load_state_dict(prior_accepted, strict=True)
                stop_reason = "time_cap_during_closure"
                break
            except FloatingPointError as exc:
                model.load_state_dict(prior_accepted, strict=True)
                stop_reason = "nonfinite_loss"
                error_text = str(exc)
                break
            accepted_state = clone_state_to_cpu(model)
            steps += 1
            history.append(
                {
                    "step": steps,
                    "loss": last_loss,
                    "components": dict(last_components),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if last_loss is None or not math.isfinite(last_loss):
                model.load_state_dict(prior_accepted, strict=True)
                accepted_state = prior_accepted
                stop_reason = "nonfinite_loss"
                break
        else:
            stop_reason = "time_cap"
    except Exception as exc:
        model.load_state_dict(accepted_state, strict=True)
        stop_reason = "optimizer_error"
        error_text = f"{type(exc).__name__}: {exc}"

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output_checkpoint = args.output / "checkpoint-lbfgs.pt"
    torch.save(
        {
            "state_dict": accepted_state,
            "seed": args.seed,
            "stage": "deterministic_full_pool_LBFGS_continuation",
            "source_checkpoint_sha256": checkpoint_sha,
            "source_pool_sha256": pool_sha,
            "lbfgs_steps": steps,
            "loss_components": last_components,
        },
        output_checkpoint,
    )
    with (args.output / "lbfgs_history.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {
        "status": "lbfgs_refined" if steps > 0 else stop_reason,
        "seed": args.seed,
        "source_kernel": args.source_kernel,
        "stage": "deterministic_full_pool_LBFGS_continuation",
        "optimizer": "torch.optim.LBFGS",
        "pool_policy": "saved full pool, fixed order, deterministic chunked closure",
        "lbfgs_steps": steps,
        "elapsed_seconds": elapsed,
        "worker_wall_seconds": time.perf_counter() - worker_started,
        "seconds_cap": args.seconds,
        "stop_reason": stop_reason,
        "last_loss": last_loss,
        "loss_components": last_components,
        "device": visible_name,
        "cuda_index_visible_to_worker": 0,
        "global_device_index": args.global_device_index,
        "interior_points": interior_count,
        "boundary_points_per_set": {name: int(tensors[name].shape[0]) for name in ("inlet", "lower_wall", "upper_wall", "cylinder", "outlet")},
        "chunk_size": args.chunk_size,
        "input_checkpoint_sha256": checkpoint_sha,
        "input_pool_sha256": pool_sha,
        "source_run_sha256": args.source_run_sha256,
        "source_provenance_sha256": args.source_provenance_sha256,
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": sha256(output_checkpoint),
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "field_relative_l2": None,
        "scientific_accepted": False,
    }
    if error_text is not None:
        summary["error"] = error_text
    write_json(args.output / "lbfgs_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
