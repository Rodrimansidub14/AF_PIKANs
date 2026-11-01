"""Kaggle T4 worker for fixed-pool Allen-Cahn L-BFGS continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PAYLOAD = Path(__file__).resolve().parent
CONFIG = json.loads((PAYLOAD / "s3-source-config.json").read_text(encoding="utf-8"))
AC_ROOT = Path(CONFIG["input_roots"]["allen_cahn"])
S1_ROOT = Path(CONFIG["input_roots"]["s1"])
sys.path.insert(0, str(AC_ROOT))
import run_campaign as base  # noqa: E402


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_input_paths(arm: str, seed: int) -> tuple[Path, Path]:
    checkpoint = AC_ROOT / "campaign" / arm / f"seed-{seed}" / "checkpoint-final.pt"
    pool = AC_ROOT / "campaign" / "pools" / f"seed-{seed}.npz"
    return checkpoint, pool


def verify_input_hashes(arm: str, seed: int) -> tuple[Path, Path, dict[str, str]]:
    checkpoint, pool = expected_input_paths(arm, seed)
    expected_checkpoint = CONFIG["checkpoint_hashes"][arm][str(seed)]
    expected_pool = CONFIG["pool_hashes"][str(seed)]
    actual_checkpoint = sha256(checkpoint)
    actual_pool = sha256(pool)
    if actual_checkpoint != expected_checkpoint:
        raise ValueError(f"checkpoint hash mismatch for {arm} seed {seed}")
    if actual_pool != expected_pool:
        raise ValueError(f"pool hash mismatch for seed {seed}")
    return checkpoint, pool, {"checkpoint_sha256": actual_checkpoint, "pool_file_sha256": actual_pool}


def load_pools(path: Path, seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        pools = {name: np.array(stored[name], dtype=np.float64, copy=True) for name in ("interior", "heldout")}
    if pools["interior"].shape != (8192, 2) or pools["heldout"].shape != (16384, 2):
        raise ValueError(f"unexpected saved pool shapes for seed {seed}")
    for name, points in pools.items():
        if not np.isfinite(points).all():
            raise ValueError(f"saved {name} pool contains non-finite coordinates for seed {seed}")
        if np.any(points[:, 0] < -1.0) or np.any(points[:, 0] > 1.0):
            raise ValueError(f"saved {name} pool has x coordinates outside [-1, 1] for seed {seed}")
        if np.any(points[:, 1] < 0.0) or np.any(points[:, 1] > 1.0):
            raise ValueError(f"saved {name} pool has t coordinates outside [0, 1] for seed {seed}")
    return pools


def write_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                     arm: str, seed: int, updates: int, elapsed: float, source: dict[str, str],
                     status: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    state = {
        "status": "PENDING_HUMAN_SIGNATURE",
        "arm": arm,
        "seed": int(seed),
        "lbfgs_updates": int(updates),
        "elapsed_gpu_seconds_estimate": float(elapsed),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "pool_file_sha256": source["pool_file_sha256"],
        "run_status": status,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, temporary)
    temporary.replace(path)


def configure_torch() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one assigned CUDA device; CPU fallback is disabled.")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if "t4" not in name.lower():
        raise RuntimeError(f"Worker requires an assigned T4, found {name!r}")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda:0")


def train(arm: str, seed: int, output: Path, cap: float) -> int:
    device = configure_torch()
    checkpoint_path, pool_path, source_hashes = verify_input_hashes(arm, seed)
    pools = load_pools(pool_path, seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != arm or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError(f"source checkpoint identity mismatch for {arm} seed {seed}")
    model = base.build_model(arm, seed).to(device=device, dtype=torch.float64)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.train()
    points = torch.as_tensor(pools["interior"], dtype=torch.float64, device=device)
    output.mkdir(parents=True, exist_ok=False)
    parameters = list(model.parameters())
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=1,
        max_eval=8,
        tolerance_grad=1.0e-8,
        tolerance_change=0.0,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    started = time.perf_counter()
    updates = 0
    closure_evaluations = 0
    last_loss: float | None = None
    run_status = "running"
    error: str | None = None
    final_path = output / "checkpoint-final.pt"
    progress_path = output / "progress.json"

    def elapsed() -> float:
        torch.cuda.synchronize(device)
        return max(0.0, time.perf_counter() - started)

    def save_progress(status: str) -> None:
        seconds = elapsed()
        write_checkpoint(final_path, model, optimizer, arm, seed, updates, seconds, source_hashes, status)
        write_json(
            progress_path,
            {
                "status": status,
                "arm": arm,
                "seed": seed,
                "lbfgs_updates": updates,
                "closure_evaluations": closure_evaluations,
                "last_loss_scaled_mse": last_loss,
                "elapsed_gpu_seconds_estimate": seconds,
                "gpu_seconds_cap": cap,
                "source_checkpoint_sha256": source_hashes["checkpoint_sha256"],
                "pool_file_sha256": source_hashes["pool_file_sha256"],
                "scientific_accepted": False,
            },
        )

    save_progress("loaded_checkpoint")
    try:
        def closure() -> torch.Tensor:
            nonlocal closure_evaluations, last_loss
            optimizer.zero_grad(set_to_none=True)
            residual = base.pde_residual(model, points)
            loss = residual.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Allen-Cahn L-BFGS closure loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("non-finite Allen-Cahn L-BFGS gradient")
            closure_evaluations += 1
            last_loss = float(loss.detach().cpu().item())
            return loss

        while elapsed() < cap:
            optimizer.step(closure)
            updates += 1
            if updates % 5 == 0:
                save_progress("running_lbfgs")
        run_status = "time_cap_reached"
    except Exception as exc:
        run_status = "lbfgs_exception"
        error = f"{type(exc).__name__}: {exc}"
    try:
        save_progress(run_status)
    except Exception as exc:
        error = (error + "; " if error else "") + f"final_checkpoint_error={type(exc).__name__}: {exc}"
        run_status = "checkpoint_write_failed"
    write_json(
        output / "cost.json",
        {
            "status": run_status,
            "arm": arm,
            "seed": seed,
            "lbfgs_updates": updates,
            "closure_evaluations": closure_evaluations,
            "last_loss_scaled_mse": last_loss,
            "lbfgs_gpu_seconds_cap": cap,
            "training_gpu_seconds_estimate": elapsed(),
            "history_size": 100,
            "max_eval_per_step": 8,
            "line_search_fn": "strong_wolfe",
            "source_checkpoint_sha256": source_hashes["checkpoint_sha256"],
            "pool_file_sha256": source_hashes["pool_file_sha256"],
            "error": error,
            "scientific_accepted": False,
            "reference_accepted": False,
        },
    )
    return 1 if run_status in {"lbfgs_exception", "checkpoint_write_failed"} else 0


def load_reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        x_source = np.array(archive["x"], dtype=np.float64, copy=True)
        t_source = np.array(archive["t"], dtype=np.float64, copy=True)
        field = np.array(archive["u"], dtype=np.float64, copy=True)
    if field.shape == (len(x_source), len(t_source)):
        field = field.T
    if field.shape != (len(t_source), len(x_source)):
        raise ValueError(f"reference field shape {field.shape} does not match its x/t grids")
    x_eval = np.linspace(-1.0, 1.0, 512, dtype=np.float64)
    t_eval = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    along_x = np.stack([np.interp(x_eval, x_source, row) for row in field], axis=0)
    evaluated = np.stack([np.interp(t_eval, t_source, along_x[:, index]) for index in range(len(x_eval))], axis=1)
    return {"x": x_eval, "t": t_eval, "u": evaluated}


def score(arm: str, seed: int, run_dir: Path, output: Path) -> int:
    device = configure_torch()
    checkpoint_path, pool_path, source_hashes = verify_input_hashes(arm, seed)
    del checkpoint_path
    pools = load_pools(pool_path, seed)
    s3_checkpoint = run_dir / "checkpoint-final.pt"
    if not s3_checkpoint.is_file():
        raise FileNotFoundError(f"No S3 continuation checkpoint for {arm} seed {seed}")
    model_data = torch.load(s3_checkpoint, map_location="cpu", weights_only=False)
    model = base.build_model(arm, seed).to(device=device, dtype=torch.float64)
    model.load_state_dict(model_data["state_dict"], strict=True)
    model.eval()
    started = time.perf_counter()
    score_results: dict[str, Any] = {}
    for ref_name in ("FD8192", "ETDRK4"):
        ref_path = (AC_ROOT / CONFIG["reference_paths"][ref_name]) if ref_name == "FD8192" else (S1_ROOT / CONFIG["reference_paths"][ref_name])
        expected = CONFIG["reference_hashes"][ref_name]
        actual = sha256(ref_path)
        if actual != expected:
            raise ValueError(f"{ref_name} reference hash mismatch")
        reference = load_reference(ref_path)
        score_results[ref_name] = base.evaluate_field(model, reference, device)
    residual = base.heldout_residual(model, pools["heldout"], device)
    seconds = time.perf_counter() - started
    result = {
        "status": "scored",
        "arm": arm,
        "seed": seed,
        "source_checkpoint_sha256": source_hashes["checkpoint_sha256"],
        "pool_file_sha256": source_hashes["pool_file_sha256"],
        "continuation_checkpoint_sha256": sha256(s3_checkpoint),
        "relative_weighted_space_time_l2": score_results,
        "heldout_pde_residual_rms_normalized": residual,
        "scoring_wall_seconds": seconds,
        "score_grid": "512x256 for both frozen references",
        "scientific_accepted": False,
        "reference_accepted": False,
    }
    write_json(output, result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "score"), required=True)
    parser.add_argument("--arm", choices=tuple(base.ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=tuple(CONFIG["seeds"]), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cap", type=float, default=240.0)
    args = parser.parse_args()
    try:
        if args.mode == "train":
            return train(args.arm, args.seed, args.out, args.cap)
        if args.run_dir is None:
            parser.error("--run-dir is required for scoring")
        return score(args.arm, args.seed, args.run_dir, args.out)
    except Exception as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            args.out.parent / ("worker-failure.json" if args.mode == "train" else "score-failure.json"),
            {"status": "failed", "mode": args.mode, "arm": args.arm, "seed": args.seed,
             "error": f"{type(exc).__name__}: {exc}", "scientific_accepted": False},
        )
        print(f"worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
