#!/usr/bin/env python3
"""DRAFT_UNVERIFIED CUDA-T4 Allen–Cahn paired campaign.

This file is intended to be copied into a Kaggle T4 session with NumPy, SciPy
and PyTorch installed. It has deliberately not been executed or validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn
from torch.nn import functional as F

STATUS = "DRAFT_UNVERIFIED"
SEEDS = [401, 402, 403, 404, 405]
ARMS = ("AC-RBF-Sobol", "AC-DeepXDE-MLP")
DOMAIN_X = (-1.0, 1.0)
DOMAIN_T = (0.0, 1.0)
DIFFUSION = 0.001
REACTION = 5.0
RESIDUAL_SCALE = 6.001
TRAIN_POOL = 8192
HELDOUT_POOL = 16384
RUN_CAP_SECONDS = 600.0
REFERENCE_CAP_SECONDS = 2700.0
SESSION_QUOTA_CAP_SECONDS = 7200.0
SCRIPT_WALL_CAP_SECONDS = 6900.0
SCORING_CAP_SECONDS = 600.0
CHECKPOINTS = (120.0, 300.0, 480.0, 600.0)
REFERENCE_LEVELS = (2048, 4096, 8192)
REFERENCE_DT = 1.0e-5
SNAPSHOT_DT = 0.005
SPATIAL_GATE = 5.0e-5
TEMPORAL_GATE = 1.0e-7


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_t4(*, worker: bool = False) -> torch.device:
    required = 1 if worker else 2
    if not torch.cuda.is_available() or torch.cuda.device_count() < required:
        raise RuntimeError(f"At least {required} visible CUDA T4 GPU(s) required; CPU fallback is disabled.")
    names = [torch.cuda.get_device_name(index) for index in range(required)]
    if any("T4" not in name.upper() for name in names):
        raise RuntimeError(f"Expected Tesla T4 device(s), found {names!r}.")
    return torch.device("cuda:0")


def launch_pair(seed: int, output: Path, deadline: float) -> dict[str, bool]:
    """Run both arms for one seed concurrently, one isolated process per T4."""
    logs = output / "worker-logs"
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    for device_index, arm in enumerate(ARMS):
        run_dir = output / arm / f"seed-{seed}"
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(run_dir),
                   "--worker-arm", arm, "--worker-seed", str(seed),
                   "--worker-cap", str(RUN_CAP_SECONDS)]
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(device_index)
        log_stream = (logs / f"{arm}-seed-{seed}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=child_env, stdout=log_stream,
                                   stderr=subprocess.STDOUT)
        children.append((arm, run_dir, process, log_stream))

    pair_deadline = min(deadline, time.perf_counter() + RUN_CAP_SECONDS + 30.0)
    while any(process.poll() is None for _, _, process, _ in children):
        if time.perf_counter() >= pair_deadline:
            for _, _, process, _ in children:
                if process.poll() is None:
                    process.terminate()
            for _, _, process, _ in children:
                if process.poll() is None:
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
            break
        time.sleep(1.0)

    outcome: dict[str, bool] = {}
    for arm, run_dir, process, log_stream in children:
        log_stream.close()
        ok = process.returncode == 0 and (run_dir / "checkpoint-final.pt").exists()
        outcome[arm] = ok
        if not ok:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "failure.json", {
                "arm": arm, "seed": seed, "status": "failed_or_pair_deadline",
                "returncode": process.returncode,
                "log": str(logs / f"{arm}-seed-{seed}.log"),
            })
    return outcome


def sobol_points(count: int, dimension: int, seed: int) -> np.ndarray:
    exponent = int(round(math.log2(count)))
    if 2**exponent != count:
        raise ValueError("Sobol pool sizes must be powers of two.")
    return qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(exponent)


def make_pools(seed: int) -> dict[str, np.ndarray]:
    xy = sobol_points(TRAIN_POOL, 2, 10 * seed + 1)
    xy[:, 0] = 2.0 * xy[:, 0] - 1.0
    heldout = sobol_points(HELDOUT_POOL, 2, 10 * seed + 5)
    heldout[:, 0] = 2.0 * heldout[:, 0] - 1.0
    return {"interior": xy, "heldout": heldout}


class RBFEdge(nn.Module):
    """Same fixed Gaussian edge basis and SiLU residual path as the incumbent."""

    def __init__(self, inputs: int, outputs: int, basis_count: int = 11):
        super().__init__()
        self.register_buffer("centers", torch.linspace(-1.05, 1.05, basis_count))
        self.register_buffer("width", torch.tensor(0.21))
        self.coefficients = nn.Parameter(torch.empty(outputs, inputs, basis_count))
        self.base = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.normal_(self.coefficients, std=0.1 / math.sqrt(inputs))
        nn.init.normal_(self.base, std=0.1 / math.sqrt(inputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())
        return torch.einsum("big,oig->bo", basis, self.coefficients) + F.linear(F.silu(x), self.base)


class HardConstraintAC(nn.Module):
    model_id = "abstract"

    def normalize(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.stack((coordinates[:, 0], 2.0 * coordinates[:, 1] - 1.0), dim=1)

    def constrain(self, coordinates: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        x = coordinates[:, 0:1]
        t = coordinates[:, 1:2]
        initial = x.square() * torch.cos(torch.pi * x)
        return initial + t * (1.0 - x.square()) * raw


class RBFAC(HardConstraintAC):
    model_id = "rbf_fixed_2_16_16_1"

    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layers = nn.ModuleList((RBFEdge(2, 16), RBFEdge(16, 16), RBFEdge(16, 1)))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        hidden = self.normalize(coordinates)
        for layer in self.layers[:-1]:
            hidden = torch.tanh(layer(hidden))
        return self.constrain(coordinates, self.layers[-1](hidden))


class DeepXDEMLPAC(HardConstraintAC):
    """Four hidden tanh layers of width 20, matching the cited DeepXDE demo."""

    model_id = "deepxde_tanh_4x20_hard_transform"

    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            widths = (2, 20, 20, 20, 20, 1)
            self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
            for layer in self.layers[:-1]:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        hidden = self.normalize(coordinates)
        for layer in self.layers[:-1]:
            hidden = torch.tanh(layer(hidden))
        return self.constrain(coordinates, self.layers[-1](hidden))


def build_model(arm: str, seed: int) -> nn.Module:
    if arm == "AC-RBF-Sobol":
        return RBFAC(seed)
    if arm == "AC-DeepXDE-MLP":
        return DeepXDEMLPAC(seed)
    raise ValueError(f"Unknown arm: {arm}")


def pde_residual(model: nn.Module, coordinates: torch.Tensor) -> torch.Tensor:
    xy = coordinates.detach().requires_grad_(True)
    u = model(xy)
    gradient = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    ux, ut = gradient[:, 0:1], gradient[:, 1:2]
    uxx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True, retain_graph=True)[0][:, 0:1]
    return (ut - DIFFUSION * uxx - REACTION * (u - u.pow(3))) / RESIDUAL_SCALE


def sync_elapsed(started: float, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def train_one(arm: str, seed: int, pools: dict[str, np.ndarray], output: Path,
              device: torch.device, cap: float, *, smoke: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(arm, seed).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    generator = torch.Generator(device=device).manual_seed(seed + 100000)
    train_tensor = torch.as_tensor(pools["interior"], dtype=torch.float64, device=device)
    history: list[dict] = []
    started = time.perf_counter()
    next_checkpoint = 0
    step = 0

    def save_checkpoint(elapsed: float, label: str) -> None:
        torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                    "step": step, "elapsed_gpu_wall_seconds": elapsed, "label": label},
                   output / f"checkpoint-{label}.pt")

    while True:
        elapsed = sync_elapsed(started, device)
        if elapsed >= cap:
            break
        index = torch.randint(train_tensor.shape[0], (1024,), generator=generator, device=device)
        batch = train_tensor.index_select(0, index)
        optimizer.zero_grad(set_to_none=True)
        loss = pde_residual(model, batch).square().mean()
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            save_checkpoint(sync_elapsed(started, device), "nonfinite-gradient")
            raise FloatingPointError(f"non-finite gradient in {arm}, seed {seed}, step {step + 1}")
        optimizer.step()
        scalar_loss = float(loss.detach().cpu())
        step += 1
        if not math.isfinite(scalar_loss):
            save_checkpoint(sync_elapsed(started, device), "nonfinite")
            raise FloatingPointError(f"non-finite loss in {arm}, seed {seed}, step {step}")
        if step % 100 == 0:
            history.append({"step": step, "loss": scalar_loss,
                            "elapsed_gpu_wall_seconds": sync_elapsed(started, device),
                            "optimizer": "adam"})
        elapsed = sync_elapsed(started, device)
        while next_checkpoint < len(CHECKPOINTS) and elapsed >= CHECKPOINTS[next_checkpoint]:
            save_checkpoint(elapsed, f"t{int(CHECKPOINTS[next_checkpoint])}")
            next_checkpoint += 1

    elapsed = sync_elapsed(started, device)
    save_checkpoint(elapsed, "final")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": STATUS,
        "arm": arm,
        "seed": seed,
        "device": torch.cuda.get_device_name(device),
        "dtype": "float64",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": step,
        "elapsed_gpu_wall_seconds": elapsed,
        "cap_overrun_seconds": max(0.0, elapsed - cap),
        "optimizer": "adam",
        "smoke_only": smoke,
        "checkpoint_selection": "final state; no reference-based selection",
    }
    write_json(output / "training_summary.json", summary)
    return summary


def laplacian_dirichlet(interior: torch.Tensor, h: float) -> torch.Tensor:
    minus_one = torch.full((1,), -1.0, dtype=interior.dtype, device=interior.device)
    left = torch.cat((minus_one, interior[:-1]))
    right = torch.cat((interior[1:], minus_one))
    return (left - 2.0 * interior + right) / (h * h)


def reference_rhs(interior: torch.Tensor, h: float) -> torch.Tensor:
    return DIFFUSION * laplacian_dirichlet(interior, h) + REACTION * (interior - interior.pow(3))


def reference_grid(n_intervals: int, dt: float, device: torch.device,
                   started: float, deadline: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps_per_snapshot = int(round(SNAPSHOT_DT / dt))
    if not math.isclose(steps_per_snapshot * dt, SNAPSHOT_DT, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("dt must divide the 0.005 snapshot interval exactly")
    total_steps = int(round(1.0 / dt))
    if not math.isclose(total_steps * dt, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("dt must divide the unit time interval exactly")
    x = torch.linspace(-1.0, 1.0, n_intervals + 1, dtype=torch.float64, device=device)
    full_initial = x.square() * torch.cos(torch.pi * x)
    interior = full_initial[1:-1].clone()
    h = 2.0 / n_intervals
    saved = [torch.cat((full_initial[:1] * 0.0 - 1.0, interior, full_initial[-1:] * 0.0 - 1.0))]
    for step in range(1, total_steps + 1):
        k1 = reference_rhs(interior, h)
        k2 = reference_rhs(interior + (0.5 * dt) * k1, h)
        k3 = reference_rhs(interior + (0.5 * dt) * k2, h)
        k4 = reference_rhs(interior + dt * k3, h)
        interior = interior + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if step % steps_per_snapshot == 0:
            saved.append(torch.cat((saved[0][:1], interior, saved[0][-1:])))
        if step % 1000 == 0:
            elapsed = sync_elapsed(started, device)
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"GPU reference cap exceeded at N={n_intervals}, step={step}, elapsed={elapsed:.1f}s")
    values = torch.stack(saved).detach().cpu().numpy()
    return x.detach().cpu().numpy(), np.linspace(0.0, 1.0, 201), values


def weighted_error(coarse: np.ndarray, fine: np.ndarray, dx: float, dt: float) -> tuple[float, float]:
    if coarse.shape != fine.shape:
        raise ValueError("reference arrays must have the same nested-grid shape")
    wx = np.ones(coarse.shape[1], dtype=np.float64)
    wt = np.ones(coarse.shape[0], dtype=np.float64)
    wx[[0, -1]] = 0.5
    wt[[0, -1]] = 0.5
    weights = wt[:, None] * wx[None, :]
    numerator = np.sum(weights * (fine - coarse) ** 2) * dx * dt
    denominator = np.sum(weights * fine**2) * dx * dt
    space_time = float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))
    per_time_num = np.sum(wx[None, :] * (fine - coarse) ** 2, axis=1) * dx
    per_time_den = np.sum(wx[None, :] * fine**2, axis=1) * dx
    per_time = np.sqrt(per_time_num / np.maximum(per_time_den, np.finfo(float).tiny))
    return space_time, float(np.max(per_time))


def generate_reference(output: Path, device: torch.device,
                       cap_seconds: float = REFERENCE_CAP_SECONDS) -> tuple[dict | None, dict]:
    reference_dir = output / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + min(REFERENCE_CAP_SECONDS, max(0.0, cap_seconds))
    levels = []
    previous = None
    try:
        for n_intervals in REFERENCE_LEVELS:
            x, times, values = reference_grid(n_intervals, REFERENCE_DT, device, started, deadline)
            level_path = reference_dir / f"fd_rk4_N{n_intervals}.npz"
            np.savez_compressed(level_path, x=x, t=times, u=values, dt=REFERENCE_DT)
            row = {"n_intervals": n_intervals, "dt": REFERENCE_DT,
                   "sha256": sha256(level_path), "spatial_gate_pass": None}
            if previous is not None:
                old_n, old_values = previous
                if n_intervals != 2 * old_n:
                    raise ValueError("spatial levels must be nested by a factor of two")
                fine_on_coarse = values[:, ::2]
                space_time, max_snapshot = weighted_error(old_values, fine_on_coarse,
                                                          2.0 / old_n, SNAPSHOT_DT)
                row.update({"coarse_n_intervals": old_n,
                            "relative_space_time_l2": space_time,
                            "max_snapshot_relative_l2": max_snapshot,
                            "spatial_gate_pass": space_time <= SPATIAL_GATE and max_snapshot <= SPATIAL_GATE})
            levels.append(row)
            previous = (n_intervals, values)

        pair_results = [row.get("spatial_gate_pass") is True for row in levels]
        spatial_ok = any(pair_results[index] and pair_results[index + 1]
                         for index in range(len(pair_results) - 1))
        final_n = REFERENCE_LEVELS[-1]
        x, times, base = reference_grid(final_n, REFERENCE_DT, device, started, deadline)
        x2, times2, half = reference_grid(final_n, REFERENCE_DT / 2.0, device, started, deadline)
        if not np.array_equal(x, x2) or not np.array_equal(times, times2):
            raise RuntimeError("temporal grids are inconsistent")
        temporal_space_time, temporal_snapshot = weighted_error(base, half, 2.0 / final_n, SNAPSHOT_DT)
        temporal_ok = temporal_space_time <= TEMPORAL_GATE and temporal_snapshot <= TEMPORAL_GATE
        status = {
            "status": STATUS,
            "method": "CUDA float64 second-order FD method-of-lines + explicit classical RK4",
            "reference_ready_for_review": bool(spatial_ok and temporal_ok),
            "reference_accepted": False,
            "external_review_required": True,
            "reference_gpu_wall_seconds": sync_elapsed(started, device),
            "spatial_levels": levels,
            "two_consecutive_spatial_pairs_pass": spatial_ok,
            "temporal_check": {"dt": REFERENCE_DT, "dt_half": REFERENCE_DT / 2.0,
                               "relative_space_time_l2": temporal_space_time,
                               "max_snapshot_relative_l2": temporal_snapshot,
                               "passed": temporal_ok},
            "field_score_status": "candidate_only" if spatial_ok and temporal_ok else "blocked_reference_gates_failed",
            "comparison_to_archived_bdf": "not performed by this script; reviewer must check provenance and common-grid agreement",
        }
        write_json(reference_dir / "reference_status.json", status)
        if not (spatial_ok and temporal_ok):
            return None, status
        final_path = reference_dir / "ac_gpu_reference_candidate.npz"
        np.savez_compressed(final_path, x=x, t=times, u=half, n_intervals=final_n,
                            dt=REFERENCE_DT / 2.0, method="FD-MOL-RK4-CUDA-float64")
        status["candidate_sha256"] = sha256(final_path)
        write_json(reference_dir / "reference_status.json", status)
        return {"x": x, "t": times, "u": half}, status
    except Exception as exc:
        status = {"status": STATUS, "reference_ready_for_review": False,
                  "reference_accepted": False, "field_score_status": "blocked_reference_generation_failed",
                  "reference_gpu_wall_seconds": sync_elapsed(started, device),
                  "exception": f"{type(exc).__name__}: {exc}", "completed_spatial_levels": levels}
        write_json(reference_dir / "reference_status.json", status)
        return None, status


def evaluate_field(model: nn.Module, reference: dict, device: torch.device,
                   deadline: float | None = None) -> dict:
    x = reference["x"]
    times = reference["t"]
    target = reference["u"]
    flat = np.stack(np.meshgrid(x, times, indexing="xy"), axis=-1).reshape(-1, 2)
    predicted = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat), 32768):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("campaign scoring allowance exhausted during field evaluation")
            batch = torch.as_tensor(flat[start:start + 32768], dtype=torch.float64, device=device)
            predicted.append(model(batch).squeeze(1).cpu().numpy())
    pred = np.concatenate(predicted).reshape(len(times), len(x))
    if pred.shape != target.shape:
        raise ValueError(f"reference shape {target.shape} != prediction shape {pred.shape}")
    weights_x = np.ones(len(x), dtype=np.float64)
    weights_t = np.ones(len(times), dtype=np.float64)
    weights_x[[0, -1]] = 0.5
    weights_t[[0, -1]] = 0.5
    dx = 2.0 / (len(x) - 1)
    dt = 1.0 / (len(times) - 1)
    weights = weights_t[:, None] * weights_x[None, :]
    rel = float(np.sqrt(np.sum(weights * (pred - target) ** 2) /
                        max(np.sum(weights * target**2), np.finfo(float).tiny)))
    max_abs = float(np.max(np.abs(pred - target)))
    slice_num = np.sum(weights_x[None, :] * (pred - target) ** 2, axis=1) * dx
    slice_den = np.sum(weights_x[None, :] * target**2, axis=1) * dx
    slice_rel = np.sqrt(slice_num / np.maximum(slice_den, np.finfo(float).tiny))
    return {"relative_weighted_space_time_l2": rel,
            "max_snapshot_relative_l2": float(np.max(slice_rel)),
            "max_absolute_error": max_abs}


def heldout_residual(model: nn.Module, points: np.ndarray, device: torch.device,
                     deadline: float | None = None) -> float:
    values = []
    model.eval()
    for start in range(0, len(points), 1024):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("campaign scoring allowance exhausted during held-out residual")
        chunk = torch.as_tensor(points[start:start + 1024], dtype=torch.float64, device=device)
        values.append(pde_residual(model, chunk).detach().square().sum().item())
    return float(math.sqrt(sum(values) / len(points)))


def trace_errors(model: nn.Module, device: torch.device) -> dict:
    x = torch.linspace(-1.0, 1.0, 1025, dtype=torch.float64, device=device)
    t = torch.linspace(0.0, 1.0, 257, dtype=torch.float64, device=device)
    initial_xy = torch.stack((x, torch.zeros_like(x)), dim=1)
    boundary_xy = torch.cat((torch.stack((-torch.ones_like(t), t), dim=1),
                             torch.stack((torch.ones_like(t), t), dim=1)))
    with torch.no_grad():
        initial_target = x.square() * torch.cos(torch.pi * x)
        initial_error = torch.max(torch.abs(model(initial_xy).squeeze(1) - initial_target)).item()
        boundary_error = torch.max(torch.abs(model(boundary_xy).squeeze(1) + 1.0)).item()
    return {"initial_max_abs": float(initial_error), "boundary_max_abs": float(boundary_error)}


def score_run(arm: str, seed: int, run_dir: Path, reference: dict,
              pools: dict[str, np.ndarray], device: torch.device,
              deadline: float) -> dict:
    checkpoint = torch.load(run_dir / "checkpoint-final.pt", map_location=device, weights_only=False)
    model = build_model(arm, seed).to(device=device, dtype=torch.float64)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    result = {"arm": arm, "seed": seed, "status": STATUS,
              **evaluate_field(model, reference, device, deadline),
              "heldout_pde_residual_rms_normalized": heldout_residual(model, pools["heldout"], device, deadline),
              **trace_errors(model, device)}
    write_json(run_dir / "score.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="run only seed 400 for transport/logging; never score")
    parser.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--worker-cap", type=float, default=RUN_CAP_SECONDS)
    args = parser.parse_args()
    if args.worker_arm is not None:
        if args.worker_seed is None:
            parser.error("--worker-seed is required with --worker-arm")
        worker_device = require_t4(worker=True)
        worker_pools = make_pools(args.worker_seed)
        train_one(args.worker_arm, args.worker_seed, worker_pools, args.output,
                  worker_device, args.worker_cap)
        return

    campaign_started = time.perf_counter()
    campaign_deadline = campaign_started + SCRIPT_WALL_CAP_SECONDS
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = require_t4()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    device_info = {"device": torch.cuda.get_device_name(device),
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "python": platform.python_version(), "status": STATUS}
    write_json(output / "hardware.json", device_info)

    if args.smoke:
        smoke_pools = make_pools(400)
        for arm in ARMS:
            pool_dir = output / "smoke-seed-400"
            pool_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pool_dir / "pools.npz", **smoke_pools)
            train_one(arm, 400, smoke_pools, pool_dir / arm, device, 120.0, smoke=True)
        write_json(output / "smoke_status.json", {"status": STATUS, "scores": "not computed",
                                                    "seed": 400, "purpose": "transport and finite-output only"})
        return

    reference, ref_status = generate_reference(
        output, device, min(REFERENCE_CAP_SECONDS, max(0.0, campaign_deadline - time.perf_counter())))
    if reference is None:
        write_json(output / "campaign_status.json", {"status": STATUS, "execution": "stopped_before_training",
                                                       "reason": ref_status["field_score_status"]})
        return

    all_scores = []
    scoring_started = time.perf_counter()
    scoring_deadline = min(campaign_deadline, scoring_started + SCORING_CAP_SECONDS)
    for seed in SEEDS:
        if time.perf_counter() + RUN_CAP_SECONDS + 5.0 >= campaign_deadline:
            break
        pools = make_pools(seed)
        pool_dir = output / "pools"
        pool_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pool_dir / f"seed-{seed}.npz", **pools)
        pair_status = launch_pair(seed, output, campaign_deadline)
        for arm in ARMS:
            run_dir = output / arm / f"seed-{seed}"
            if not pair_status[arm]:
                all_scores.append({"arm": arm, "seed": seed, "status": "training_failed"})
                continue
            if time.perf_counter() >= scoring_deadline:
                all_scores.append({"arm": arm, "seed": seed, "status": "score_budget_exhausted"})
                continue
            try:
                all_scores.append(score_run(arm, seed, run_dir, reference, pools, device, scoring_deadline))
            except Exception as exc:
                failure = {"arm": arm, "seed": seed, "status": "score_failed_preserved",
                           "exception": f"{type(exc).__name__}: {exc}"}
                write_json(run_dir / "score_failure.json", failure)
                all_scores.append(failure)
    source_path = Path(__file__).resolve()
    result = {"status": STATUS, "campaign_id": "allen-cahn-t4-competitive-2026-09-27",
              "reference_status": ref_status, "scores": all_scores,
              "campaign_wall_seconds": time.perf_counter() - campaign_started,
              "session_quota_cap_seconds": SESSION_QUOTA_CAP_SECONDS,
              "script_wall_cap_seconds": SCRIPT_WALL_CAP_SECONDS,
              "source_sha256": sha256(source_path), "scientific_accepted": False,
              "field_reference_accepted": False}
    write_json(output / "campaign_results.json", result)


if __name__ == "__main__":
    main()
