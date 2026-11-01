"""Kaggle-only step-matched Heat RBF/B-spline runner; outputs remain unaccepted."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from af_sciml_model import build_model

ALPHA = 0.1
RESIDUAL_SCALE = 1.1
POOL_N = 4096
BOUNDARY_N = 512
INITIAL_N = 1024
ADAM_UPDATES = 20_000
LR = 1.0e-3
TIME_CHECKPOINTS = (120.0, 300.0, 500.0)
ARMS = {
    "rbf": {"model_id": "rbf_fixed", "label": "H-RBF", "parameters_expected": 3648},
    "bspline": {"model_id": "bspline_cubic", "label": "H-BS", "parameters_expected": 3648},
}


def json_write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sync_cuda() -> None:
    torch.cuda.synchronize()


def pool_hash(pool: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(pool):
        array = np.ascontiguousarray(pool[name], dtype="<f8")
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_pool(path: Path, seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        pool = {name: np.array(saved[name], dtype=np.float64, copy=True) for name in ("interior", "left", "right", "initial")}
        saved_seed = int(saved["seed"])
        declared_hash = str(saved["pool_sha256"])
    if saved_seed != int(seed):
        raise ValueError(f"Pool seed {saved_seed} does not match run seed {seed}.")
    actual_hash = pool_hash(pool)
    if actual_hash != declared_hash:
        raise ValueError("Persisted shared pool hash does not match its arrays.")
    return pool


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float64)


def device_profile() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each run must see exactly one assigned CUDA GPU.")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if "t4" not in name.lower():
        raise RuntimeError(f"An assigned T4 is required; found {name!r}.")
    props = torch.cuda.get_device_properties(0)
    return {
        "device": name,
        "visible_device_count": 1,
        "active_device_index": 0,
        "total_memory_bytes": int(props.total_memory),
        "capability": list(torch.cuda.get_device_capability(0)),
        "allocation_clock": "synchronized wall seconds on one assigned T4",
    }


def heat_loss(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    output = model(points)
    gradient = torch.autograd.grad(
        output, points, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True
    )[0]
    ux = gradient[:, 0:1]
    ut = gradient[:, 1:2]
    uxx = torch.autograd.grad(
        ux, points, grad_outputs=torch.ones_like(ux), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    residual = (ut - ALPHA * uxx) / RESIDUAL_SCALE
    return torch.mean(residual.square())


def save_checkpoint(
    model: torch.nn.Module, path: Path, *, arm: str, seed: int, adam_step: int,
    lbfgs_step: int, stage: str, elapsed_seconds: float, pool_digest: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "status": "PENDING_HUMAN_SIGNATURE",
            "arm": arm,
            "model_id": ARMS[arm]["model_id"],
            "seed": int(seed),
            "step": int(adam_step + lbfgs_step),
            "adam_step": int(adam_step),
            "lbfgs_step": int(lbfgs_step),
            "stage": stage,
            "pool_sha256": pool_digest,
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError("Only the predeclared RBF and B-spline arms are enabled.")
    set_reproducible_seed(args.seed)
    profile = device_profile()
    pool = read_pool(args.pool_file, args.seed)
    pool_digest = pool_hash(pool)
    device = torch.device("cuda:0")
    model = build_model("heat", {"id": ARMS[args.arm]["model_id"]}).to(device=device, dtype=torch.float64)
    parameter_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if parameter_count != ARMS[args.arm]["parameters_expected"]:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}")
    points = torch.as_tensor(pool["interior"], dtype=torch.float64, device=device)
    points.requires_grad_(True)
    adam = torch.optim.Adam(model.parameters(), lr=LR)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    adam_updates = 0
    lbfgs_updates = 0
    lbfgs_evaluations = 0
    lbfgs_stop_reason = "adam_target_not_reached"
    lbfgs_error = None
    last_loss = None
    checkpoints_written: list[float] = []

    def elapsed() -> float:
        sync_cuda()
        return max(0.0, time.perf_counter() - started)

    def checkpoint(path: Path, stage: str) -> None:
        save_checkpoint(
            model, path, arm=args.arm, seed=args.seed, adam_step=adam_updates,
            lbfgs_step=lbfgs_updates, stage=stage, elapsed_seconds=elapsed(), pool_digest=pool_digest,
        )

    def write_progress(stage: str, run_status: str) -> None:
        json_write_atomic(
            args.out / "progress.json",
            {
                "status": run_status, "stage": stage, "arm": args.arm, "seed": args.seed,
                "adam_updates": adam_updates, "adam_target_updates": ADAM_UPDATES,
                "adam_target_reached": adam_updates >= ADAM_UPDATES,
                "lbfgs_updates": lbfgs_updates, "lbfgs_maxiter": args.lbfgs_maxiter,
                "lbfgs_function_evaluations": lbfgs_evaluations, "lbfgs_maxfun": args.lbfgs_maxfun,
                "lbfgs_stop_reason": lbfgs_stop_reason, "elapsed_gpu_seconds_estimate": elapsed(),
                "pool_sha256": pool_digest, "scientific_accepted": False,
            },
        )

    def write_cost(run_status: str, stage: str) -> dict[str, Any]:
        seconds = elapsed()
        result = {
            "draft_status": "PENDING_HUMAN_SIGNATURE",
            "arm": args.arm, "label": ARMS[args.arm]["label"], "model_id": ARMS[args.arm]["model_id"],
            "seed": int(args.seed), "run_kind": "tier2_step_matched",
            "run_status": run_status, "stage": stage,
            "adam_updates": int(adam_updates), "adam_target_updates": ADAM_UPDATES,
            "adam_target_reached": bool(adam_updates >= ADAM_UPDATES),
            "lbfgs_updates": int(lbfgs_updates), "lbfgs_max_iter": int(args.lbfgs_maxiter),
            "lbfgs_function_evaluations": int(lbfgs_evaluations), "lbfgs_max_eval": int(args.lbfgs_maxfun),
            "lbfgs_stop_reason": lbfgs_stop_reason, "lbfgs_error": lbfgs_error,
            "lbfgs_config": {
                "history_size": 100, "tolerance_grad": 1e-8, "tolerance_change": 0.0,
                "line_search_fn": "strong_wolfe", "chunk_iter": args.lbfgs_chunk_iter,
                "chunk_max_eval": args.lbfgs_chunk_max_eval,
            },
            "optimizer_schedule": "Adam lr=1e-3 for 20,000 full-batch updates, then PyTorch L-BFGS using DeepXDE 1.12.2 stopping parameters",
            "training_gpu_seconds_estimate": seconds, "evaluation_gpu_seconds_estimate": 0.0,
            "allocated_gpu_seconds_estimate": seconds, "gpu_seconds_cap": float(args.gpu_seconds),
            "trainable_parameters": parameter_count, "last_training_loss_scaled_mse": last_loss,
            "pool_sha256": pool_digest, "pool_shapes": {k: list(v.shape) for k, v in pool.items()},
            "pool_generator": "exact saved Heat v4 Torch scrambled Sobol pool; streams 10*seed+1/+3/+4",
            "time_checkpoints_written": sorted(checkpoints_written), "device": profile,
            "trace_treatment": "hard: sin(pi*x)+t*x*(1-x)*raw(x,t)",
            "residual": "(u_t-0.1*u_xx)/1.1; full-batch mean-squared residual",
            "evaluation": "deferred; frozen checkpoint scored on common 512x256 CUDA grid after all training",
            "protected_512x256_grid_loaded": False, "deepxde_library_used": False,
            "scientific_accepted": False, "reference_accepted": False,
        }
        json_write_atomic(args.out / "cost.json", result)
        json_write_atomic(args.out / "run_metadata.json", result)
        write_progress(stage, run_status)
        return result

    write_progress("starting", "running")
    while adam_updates < ADAM_UPDATES:
        if elapsed() >= args.gpu_seconds:
            break
        adam.zero_grad(set_to_none=True)
        loss = heat_loss(model, points)
        if not torch.isfinite(loss):
            last_loss = None
            break
        loss.backward()
        adam.step()
        adam_updates += 1
        last_loss = float(loss.detach().cpu().item())
        current = elapsed()
        if adam_updates % 100 == 0:
            append_history(args.out / "history.csv", {
                "stage": "adam", "step": adam_updates, "loss": last_loss,
                "elapsed_gpu_seconds_estimate": current, "pool_sha256": pool_digest,
            })
        for mark in TIME_CHECKPOINTS:
            if mark not in checkpoints_written and current >= mark:
                checkpoint(args.out / f"checkpoint-{int(mark)}s.pt", "adam")
                checkpoints_written.append(mark)
        if adam_updates == 1 or adam_updates % 1000 == 0:
            checkpoint(args.out / "checkpoint-latest.pt", "adam")
            write_progress("adam", "running_adam")
    checkpoint(args.out / "checkpoint-adam-final.pt", "adam_complete" if adam_updates == ADAM_UPDATES else "adam_partial")
    checkpoint(args.out / "checkpoint-final.pt", "adam_complete" if adam_updates == ADAM_UPDATES else "adam_partial")
    if adam_updates < ADAM_UPDATES:
        lbfgs_stop_reason = "adam_target_not_reached"
        return write_cost("incomplete_adam_gpu_time_cap", "adam_partial")

    write_cost("adam_target_complete_lbfgs_running", "lbfgs_start")
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=args.lbfgs_chunk_iter,
        max_eval=args.lbfgs_chunk_max_eval, tolerance_grad=1e-8,
        tolerance_change=0.0, history_size=100, line_search_fn="strong_wolfe",
    )
    parameter0 = next(model.parameters())

    def closure() -> torch.Tensor:
        nonlocal lbfgs_evaluations, last_loss
        optimizer.zero_grad(set_to_none=True)
        value = heat_loss(model, points)
        if not torch.isfinite(value):
            raise FloatingPointError("nonfinite L-BFGS loss")
        value.backward()
        lbfgs_evaluations += 1
        last_loss = float(value.detach().cpu().item())
        return value

    while lbfgs_updates < args.lbfgs_maxiter and lbfgs_evaluations < args.lbfgs_maxfun:
        if elapsed() >= args.gpu_seconds:
            lbfgs_stop_reason = "gpu_time_cap"
            break
        state = optimizer.state[parameter0]
        before = int(state.get("n_iter", 0))
        try:
            optimizer.step(closure)
        except Exception as exc:
            lbfgs_error = f"{type(exc).__name__}: {exc}"
            lbfgs_stop_reason = "optimizer_error"
            break
        state = optimizer.state[parameter0]
        after = int(state.get("n_iter", before))
        delta = max(0, after - before)
        lbfgs_updates += delta
        if delta == 0:
            lbfgs_stop_reason = "optimizer_stopped_without_iteration"
        elif delta < args.lbfgs_chunk_iter:
            lbfgs_stop_reason = "optimizer_stopped_early"
        elif lbfgs_updates >= args.lbfgs_maxiter:
            lbfgs_stop_reason = "max_iterations"
        elif lbfgs_evaluations >= args.lbfgs_maxfun:
            lbfgs_stop_reason = "max_function_evaluations"
        else:
            lbfgs_stop_reason = "chunk_complete"
        checkpoint(args.out / "checkpoint-final.pt", "lbfgs")
        checkpoint(args.out / "checkpoint-lbfgs-latest.pt", "lbfgs")
        append_history(args.out / "history.csv", {
            "stage": "lbfgs", "step": lbfgs_updates, "loss": last_loss,
            "elapsed_gpu_seconds_estimate": elapsed(), "pool_sha256": pool_digest,
        })
        write_cost("adam_target_complete_lbfgs_running", "lbfgs")
        if lbfgs_stop_reason in ("optimizer_stopped_without_iteration", "optimizer_stopped_early",
                                 "max_iterations", "max_function_evaluations"):
            break

    if lbfgs_stop_reason == "chunk_complete":
        lbfgs_stop_reason = "gpu_time_cap" if elapsed() >= args.gpu_seconds else "lbfgs_iteration_cap"
    if lbfgs_stop_reason == "gpu_time_cap":
        final_status = "adam_target_complete_lbfgs_time_cap"
    elif lbfgs_stop_reason in ("optimizer_stopped_early", "max_iterations", "max_function_evaluations"):
        final_status = "adam_target_complete_lbfgs_stopped"
    elif lbfgs_stop_reason == "optimizer_stopped_without_iteration":
        final_status = "adam_target_complete_lbfgs_no_progress"
    else:
        final_status = "adam_target_complete_lbfgs_error_or_partial"
    checkpoint(args.out / "checkpoint-final.pt", "lbfgs_final")
    return write_cost(final_status, "lbfgs_final")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-pool", action="store_true")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu-seconds", type=float, default=600.0)
    parser.add_argument("--step-cap", type=int, default=ADAM_UPDATES)
    parser.add_argument("--lbfgs-maxiter", type=int, default=15_000)
    parser.add_argument("--lbfgs-maxfun", type=int, default=18_750)
    parser.add_argument("--lbfgs-chunk-iter", type=int, default=25)
    parser.add_argument("--lbfgs-chunk-max-eval", type=int, default=32)
    args = parser.parse_args()
    if args.verify_pool:
        pool = read_pool(args.pool_file, args.seed)
        print(json.dumps({"seed": args.seed, "pool_sha256": pool_hash(pool),
                          "pool_shapes": {k: list(v.shape) for k, v in pool.items()}}, sort_keys=True))
        return
    if args.arm is None or args.out is None:
        raise ValueError("Training requires --arm and --out.")
    if args.gpu_seconds <= 0 or args.step_cap != ADAM_UPDATES:
        raise ValueError("S2 requires a positive time cap and exactly 20,000 Adam updates.")
    if args.lbfgs_maxiter <= 0 or args.lbfgs_maxfun <= 0:
        raise ValueError("L-BFGS iteration/evaluation caps must be positive.")
    result = run_arm(args)
    print(json.dumps({key: result.get(key) for key in (
        "arm", "seed", "run_status", "adam_updates", "lbfgs_updates", "lbfgs_stop_reason",
        "training_gpu_seconds_estimate", "pool_sha256",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
