"""Self-contained runner for the matched Heat arms in the Kaggle notebook.

DRAFT_UNVERIFIED. Run only inside the reviewed remote notebook. This is a new
matched full-batch protocol; it does not claim to reproduce prior PINA runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from af_sciml_model import build_model
from heat_common_eval import finite_smoke, score_torch_model


ALPHA = 0.1
RESIDUAL_SCALE = 1.1
POOL_N = 4096
BOUNDARY_N = 512
INITIAL_N = 1024
STEP_CAP = 1_000_000
LR = 1.0e-3
TIME_CHECKPOINTS = (120.0, 300.0, 500.0)
ARMS = {
    "rbf": {"model_id": "rbf_fixed", "label": "H-RBF", "parameters_expected": 3648},
    "bspline": {"model_id": "bspline_cubic", "label": "H-BS", "parameters_expected": 3648},
    "fnn": {"model_id": "torch_tanh_fnn_20x3_hard_trace", "label": "H-FNN (DeepXDE tutorial architecture port)", "parameters_expected": 921},
}


def json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pool_arrays(seed: int) -> dict[str, np.ndarray]:
    def sobol(count: int, dimension: int, stream_seed: int) -> np.ndarray:
        engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=stream_seed)
        return engine.draw(count, dtype=torch.float64).cpu().numpy()

    interior = sobol(POOL_N, 2, 10 * int(seed) + 1)
    boundary_t = sobol(BOUNDARY_N, 1, 10 * int(seed) + 3)[:, 0]
    initial_x = sobol(INITIAL_N, 1, 10 * int(seed) + 4)[:, 0]
    core = {
        "interior": interior,
        "left": np.column_stack((np.zeros(BOUNDARY_N), boundary_t)),
        "right": np.column_stack((np.ones(BOUNDARY_N), boundary_t)),
        "initial": np.column_stack((initial_x, np.zeros(INITIAL_N))),
    }
    expected_shapes = {"interior": (POOL_N, 2), "left": (BOUNDARY_N, 2), "right": (BOUNDARY_N, 2), "initial": (INITIAL_N, 2)}
    for name, shape in expected_shapes.items():
        if core[name].shape != shape:
            raise ValueError(f"Unexpected {name} pool shape {core[name].shape}; expected {shape}.")
    return core


def pool_hash(pool: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(pool):
        array = np.ascontiguousarray(pool[name], dtype="<f8")
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_pool(path: Path, seed: int) -> None:
    pool = pool_arrays(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **pool,
        seed=np.asarray(seed, dtype=np.int64),
        pool_sha256=np.asarray(pool_hash(pool)),
        generator=np.asarray("torch.quasirandom.SobolEngine scrambled; seeds 10*seed+1/+3/+4"),
        contract=np.asarray("heat-1d-alpha-0.1;float64;4096-interior;1024-IC;512-per-boundary;torch-sobol-v1"),
    )


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


def device_profile() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each isolated run must see exactly one assigned CUDA GPU; no training started.")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if "t4" not in name.lower():
        raise RuntimeError(f"This campaign requires an assigned T4; found {name!r}.")
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": name,
        "visible_device_count": int(torch.cuda.device_count()),
        "active_device_index": 0,
        "total_memory_bytes": int(properties.total_memory),
        "capability": list(torch.cuda.get_device_capability(0)),
        "allocation_clock": "synchronized wall seconds on one exclusively assigned visible GPU",
    }


def append_cost_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


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


class TutorialFNN(torch.nn.Module):
    """Torch-only port of the tutorial's [2,20,20,20,1] tanh network."""

    def __init__(self) -> None:
        super().__init__()
        widths = (2, 20, 20, 20, 1)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(left, right) for left, right in zip(widths[:-1], widths[1:])]
        )
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        values = coordinates
        for layer in self.layers[:-1]:
            values = torch.tanh(layer(values))
        raw = self.layers[-1](values)
        x = coordinates[:, 0:1]
        t = coordinates[:, 1:2]
        return torch.sin(torch.pi * x) + t * x * (1.0 - x) * raw


def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    arm: str,
    seed: int,
    step: int,
    requested_sec: float | None,
    actual_sec: float,
    pool_digest: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "status": "DRAFT_UNVERIFIED",
            "arm": arm,
            "model_id": ARMS[arm]["model_id"],
            "seed": int(seed),
            "step": int(step),
            "requested_gpu_seconds": requested_sec,
            "actual_gpu_seconds_estimate": float(actual_sec),
            "pool_sha256": pool_digest,
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        },
        path,
    )


def heat_loss(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    output = model(points)
    gradient = torch.autograd.grad(
        output,
        points,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]
    ux = gradient[:, 0:1]
    ut = gradient[:, 1:2]
    uxx = torch.autograd.grad(
        ux,
        points,
        grad_outputs=torch.ones_like(ux),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    residual = (ut - ALPHA * uxx) / RESIDUAL_SCALE
    return torch.mean(residual**2)


def run_torch_arm(args: argparse.Namespace, pool: dict[str, np.ndarray], profile: dict[str, Any]) -> dict[str, Any]:
    set_reproducible_seed(args.seed)
    device = torch.device("cuda:0")
    if args.arm == "fnn":
        model = TutorialFNN()
    else:
        model = build_model("heat", {"id": ARMS[args.arm]["model_id"]})
    model = model.to(device=device, dtype=torch.float64)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    if parameter_count != ARMS[args.arm]["parameters_expected"]:
        raise RuntimeError(f"{args.arm} parameter count {parameter_count} did not match declared count.")
    points = torch.as_tensor(pool["interior"], dtype=torch.float64, device=device)
    points.requires_grad_(True)
    adam = torch.optim.Adam(model.parameters(), lr=LR)
    pool_digest = pool_hash(pool)
    checkpoints_saved: list[float] = []
    history_path = args.out / "history.csv"
    step = 0
    status = "adam_step_cap_before_gpu_seconds_cap"
    last_loss: float | None = None
    checkpoints = tuple(float(value) for value in args.checkpoint_times)
    sync_cuda()
    started = time.perf_counter()

    def elapsed() -> float:
        sync_cuda()
        return max(0.0, time.perf_counter() - started)

    def save_due(stage: str, stage_step: int, loss_value: float | None) -> float:
        actual = elapsed()
        if stage_step % args.log_every == 0 or stage_step == 1:
            append_cost_row(
                history_path,
                {"stage": stage, "step": stage_step, "loss": loss_value, "elapsed_gpu_seconds_estimate": actual, "pool_sha256": pool_digest},
            )
        for milestone in checkpoints:
            if milestone not in checkpoints_saved and actual >= milestone:
                save_checkpoint(
                    model,
                    args.out / f"checkpoint-{int(milestone)}s.pt",
                    arm=args.arm,
                    seed=args.seed,
                    step=step,
                    requested_sec=milestone,
                    actual_sec=actual,
                    pool_digest=pool_digest,
                )
                checkpoints_saved.append(milestone)
        return actual

    while step < args.step_cap:
        if elapsed() >= args.gpu_seconds:
            status = "gpu_seconds_cap_during_adam"
            break
        adam.zero_grad(set_to_none=True)
        loss = heat_loss(model, points)
        if not torch.isfinite(loss):
            status = "nonfinite_loss_adam"
            break
        loss.backward()
        adam.step()
        step += 1
        last_loss = float(loss.detach().cpu().item())
        actual = save_due("adam", step, last_loss)
        if actual >= args.gpu_seconds:
            status = "gpu_seconds_cap_during_adam"
            break
    if step >= args.step_cap and status != "gpu_seconds_cap_during_adam":
        status = "adam_step_cap_before_gpu_seconds_cap"

    checkpoint_start_elapsed = elapsed()
    save_checkpoint(
        model,
        args.out / "checkpoint-final.pt",
        arm=args.arm,
        seed=args.seed,
        step=step,
        requested_sec=args.gpu_seconds,
        actual_sec=checkpoint_start_elapsed,
        pool_digest=pool_digest,
    )
    allocated = elapsed()

    eval_wall_seconds = 0.0
    evaluation_gpu_seconds = 0.0
    if args.smoke:
        smoke_result = finite_smoke(model, device="cuda:0")
        json_write(args.out / "smoke_check.json", smoke_result)
        evaluation_gpu_seconds = float(smoke_result["evaluation_gpu_seconds_estimate"])
        if not smoke_result["finite_output_check"]:
            status = "nonfinite_output"
    else:
        eval_started = time.perf_counter()
        metrics = score_torch_model(model, args.out, device="cuda:0")
        eval_wall_seconds = max(0.0, time.perf_counter() - eval_started)
        validation = json.loads((args.out / "validation.json").read_text(encoding="utf-8"))
        evaluation_gpu_seconds = float(validation["evaluation_gpu_seconds_estimate"])
        (args.out / "decision_note.md").write_text(
            "Development-only score from heat-dev-257x129-v1. The protected AF "
            "512x256 grid was not loaded or scored. H-FNN is a Torch-only port "
            "of the published tutorial architecture and optimizer recipe, not "
            "DeepXDE execution or a literal reproduction.\n",
            encoding="utf-8",
        )

    summary = {
        "draft_status": "DRAFT_UNVERIFIED",
        "arm": args.arm,
        "label": ARMS[args.arm]["label"],
        "model_id": ARMS[args.arm]["model_id"],
        "seed": int(args.seed),
        "run_kind": "smoke" if args.smoke else "tier2",
        "run_status": status,
        "optimizer_schedule": f"Adam lr=1e-3 until the shared {args.gpu_seconds:g}-second device-time cap; no L-BFGS",
        "adam_updates": step,
        "lbfgs_updates": 0,
        "adam_target_updates": int(args.step_cap),
        "lbfgs_max_iter": 0,
        "final_checkpoint_requested_gpu_seconds": float(args.gpu_seconds),
        "gpu_seconds_cap": float(args.gpu_seconds),
        "training_gpu_seconds_estimate": allocated,
        "evaluation_gpu_seconds_estimate": evaluation_gpu_seconds,
        "allocated_gpu_seconds_estimate": allocated + evaluation_gpu_seconds,
        "training_cap_overshoot_seconds": max(0.0, allocated - args.gpu_seconds),
        "trainable_parameters": parameter_count,
        "last_training_loss_scaled_mse": last_loss,
        "pool_sha256": pool_digest,
        "pool_shapes": {name: list(value.shape) for name, value in pool.items()},
        "pool_generator": "torch.quasirandom.SobolEngine scrambled streams 10*seed+1/+3/+4",
        "time_checkpoints_written": sorted(checkpoints_saved),
        "device": profile,
        "trace_treatment": "hard: sin(pi*x)+t*x*(1-x)*raw(x,t)",
        "residual": "(u_t-0.1*u_xx)/1.1; full-batch mean-squared residual",
        "evaluation": "finite-only CUDA smoke" if args.smoke else "common heat-dev-257x129 CUDA score",
        "evaluation_wall_seconds_including_artifact_io": eval_wall_seconds,
        "cpu_role": "artifact serialization only",
        "common_development_metrics": None if args.smoke else metrics,
        "protected_512x256_grid_loaded": False,
        "deepxde_library_used": False,
        "h_fnn_method_note": "PyTorch port of the tutorial architecture with the shared Adam-only comparison recipe; not a DeepXDE run or faithful paper reproduction" if args.arm == "fnn" else None,
    }
    json_write(args.out / "cost.json", summary)
    json_write(args.out / "run_metadata.json", summary)
    return summary


def prepare_pool(args: argparse.Namespace) -> None:
    write_pool(args.pool_out, args.seed)
    with np.load(args.pool_out, allow_pickle=False) as pool_file:
        print(json.dumps({"seed": int(pool_file["seed"]), "pool_sha256": str(pool_file["pool_sha256"]), "path": str(args.pool_out)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRAFT_UNVERIFIED common-contract Heat pilot runner")
    parser.add_argument("--prepare-pool", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool-out", type=Path)
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu-seconds", type=float, default=500.0)
    parser.add_argument("--step-cap", type=int, default=STEP_CAP)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-times", type=float, nargs="*", default=list(TIME_CHECKPOINTS))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_pool:
        if args.pool_out is None:
            raise ValueError("--pool-out is required with --prepare-pool.")
        prepare_pool(args)
        return
    if args.arm is None or args.out is None or args.pool_file is None:
        raise ValueError("Training requires --arm, --out, and --pool-file.")
    if args.gpu_seconds <= 0 or args.step_cap <= 0:
        raise ValueError("The time and step caps must be positive.")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Run output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    pool = read_pool(args.pool_file, args.seed)
    profile = device_profile()
    try:
        result = run_torch_arm(args, pool, profile)
        print(json.dumps({key: result.get(key) for key in ("arm", "seed", "run_status", "allocated_gpu_seconds_estimate", "time_checkpoints_written")}, sort_keys=True))
    finally:
        sync_cuda()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
