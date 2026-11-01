#!/usr/bin/env python3
"""DRAFT_UNVERIFIED CUDA-T4 paired DFG Re=20 PINN campaign.

Self-contained draft using only NumPy, SciPy and PyTorch. It has not been run
or validated. FEATFLOW QoIs are the score anchors; no field FEM is bundled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import os
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
CAMPAIGN_ID = "dfg-re20-t4-competitive-2026-09-27"
SEEDS = [501, 502, 503, 504, 505]
ARMS = ("DFG-RBF-PIKAN", "DFG-Tanh-MLP")
L, H = 2.2, 0.41
CX, CY, RADIUS = 0.2, 0.2, 0.05
RHO, NU, U_MAX, U_MEAN = 1.0, 0.001, 0.3, 0.2
DIAMETER = 2.0 * RADIUS
MOMENTUM_SCALE = U_MEAN**2 / DIAMETER
CONTINUITY_SCALE = U_MEAN / DIAMETER
PRESSURE_SCALE = RHO * U_MEAN**2
FEATFLOW = {"drag_coefficient": 5.57953523384,
            "lift_coefficient": 0.010618948146,
            "pressure_difference": 0.11752016697}
FEATFLOW_URL = "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html"
INTERIOR_COUNT, BOUNDARY_COUNT = 16384, 512
INTERIOR_BATCH, BOUNDARY_BATCH = 1024, 256
RUN_CAP_SECONDS = 1200.0
SESSION_QUOTA_CAP_SECONDS = 7200.0
SCRIPT_WALL_CAP_SECONDS = 6900.0
SCORING_CAP_SECONDS = 600.0
CHECKPOINTS = (120.0, 300.0, 600.0, 900.0, 1200.0)


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
        process.poll()
        log_stream.close()
        ok = process.returncode == 0 and (run_dir / "checkpoint-final.pt").exists()
        outcome[arm] = ok
        if not ok:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "failure.json", {
                "arm": arm, "seed": seed, "status": "failed_or_pair_deadline",
                "returncode": process.returncode,
                "log": str(logs / f"{arm}-seed-{seed}.log"),
                "field_relative_l2": None,
            })
    return outcome


def sobol_1d(count: int, seed: int) -> np.ndarray:
    exponent = int(math.ceil(math.log2(count)))
    values = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(exponent)[:count, 0]
    return np.clip(values, 1.0e-12, 1.0 - 1.0e-12)


def interior_sobol(count: int, seed: int) -> tuple[np.ndarray, dict]:
    exponent = max(1, int(math.ceil(math.log2(count))))
    max_candidates = max(1024, count * 8)
    while 2**exponent <= max_candidates:
        candidates = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(exponent)
        x = L * candidates[:, 0]
        y = H * candidates[:, 1]
        radius_squared = (x - CX) ** 2 + (y - CY) ** 2
        keep = ((x > 0.0) & (x < L) & (y > 0.0) & (y < H) & (radius_squared > RADIUS**2))
        accepted = np.column_stack((x[keep], y[keep]))
        if len(accepted) >= count:
            return accepted[:count].astype(np.float64), {
                "seed": seed, "candidate_count": 2**exponent,
                "accepted_before_truncation": int(len(accepted)),
                "sampling": "scrambled Sobol with rejection of closed cylinder",
            }
        exponent += 1
    raise RuntimeError("Sobol rejection pool could not fill the requested interior count")


def make_pools(seed: int) -> tuple[dict[str, np.ndarray], dict]:
    interior, provenance = interior_sobol(INTERIOR_COUNT, seed)
    u_inlet = sobol_1d(BOUNDARY_COUNT, seed + 101)
    u_lower = sobol_1d(BOUNDARY_COUNT, seed + 211)
    u_upper = sobol_1d(BOUNDARY_COUNT, seed + 307)
    u_cylinder = sobol_1d(BOUNDARY_COUNT, seed + 401)
    u_outlet = sobol_1d(BOUNDARY_COUNT, seed + 503)
    pools = {
        "interior": interior,
        "inlet": np.column_stack((np.zeros(BOUNDARY_COUNT), H * u_inlet)),
        "lower_wall": np.column_stack((L * u_lower, np.zeros(BOUNDARY_COUNT))),
        "upper_wall": np.column_stack((L * u_upper, np.full(BOUNDARY_COUNT, H))),
        "cylinder": np.column_stack((CX + RADIUS * np.cos(2 * np.pi * u_cylinder),
                                      CY + RADIUS * np.sin(2 * np.pi * u_cylinder))),
        "outlet": np.column_stack((np.full(BOUNDARY_COUNT, L), H * u_outlet)),
    }
    heldout, heldout_provenance = interior_sobol(8192, seed + 900001)
    pools["heldout"] = heldout
    validation_unit = {
        "inlet": sobol_1d(BOUNDARY_COUNT, seed + 100101),
        "lower_wall": sobol_1d(BOUNDARY_COUNT, seed + 100211),
        "upper_wall": sobol_1d(BOUNDARY_COUNT, seed + 100307),
        "cylinder": sobol_1d(BOUNDARY_COUNT, seed + 100401),
    }
    pools.update({
        "validation_inlet": np.column_stack((np.zeros(BOUNDARY_COUNT), H * validation_unit["inlet"])),
        "validation_lower_wall": np.column_stack((L * validation_unit["lower_wall"], np.zeros(BOUNDARY_COUNT))),
        "validation_upper_wall": np.column_stack((L * validation_unit["upper_wall"], np.full(BOUNDARY_COUNT, H))),
        "validation_cylinder": np.column_stack((CX + RADIUS * np.cos(2 * np.pi * validation_unit["cylinder"]),
                                                  CY + RADIUS * np.sin(2 * np.pi * validation_unit["cylinder"]))),
    })
    return pools, {"interior": provenance, "heldout": heldout_provenance,
                   "boundary_sampling": "per-boundary independent scrambled Sobol arclength/angle; no duplicated corners"}


class RBFEdge(nn.Module):
    """Fixed Gaussian edge bases with trainable coefficients and SiLU path."""

    def __init__(self, inputs: int, outputs: int, basis_count: int = 11):
        super().__init__()
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, basis_count, dtype=torch.float32))
        self.register_buffer("width", torch.tensor(0.35, dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.empty(outputs, inputs, basis_count, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(outputs, inputs, dtype=torch.float32))
        nn.init.normal_(self.coefficients, mean=0.0, std=0.1 / math.sqrt(inputs))
        nn.init.normal_(self.base, mean=0.0, std=0.1 / math.sqrt(inputs))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((values.unsqueeze(-1) - self.centers) / self.width).square())
        return torch.einsum("big,oig->bo", basis, self.coefficients) + F.linear(F.silu(values), self.base)


class DFGModelBase(nn.Module):
    output_scales = (U_MEAN, U_MEAN, PRESSURE_SCALE)

    def normalize(self, xy: torch.Tensor) -> torch.Tensor:
        x = 2.0 * xy[:, 0:1] / L - 1.0
        y = 2.0 * xy[:, 1:2] / H - 1.0
        return torch.cat((x, y), dim=1)

    def scale_output(self, raw: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(self.output_scales, dtype=raw.dtype, device=raw.device)
        return raw * scale


class DFGRBFPIKAN(DFGModelBase):
    model_id = "dfg_rbf_fixed_2_16_16_3"

    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layers = nn.ModuleList((RBFEdge(2, 16), RBFEdge(16, 16), RBFEdge(16, 3)))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        hidden = self.normalize(xy)
        for layer in self.layers[:-1]:
            hidden = torch.tanh(layer(hidden))
        return self.scale_output(self.layers[-1](hidden))


class DFGMLP(DFGModelBase):
    """Five hidden layers of width 64; same outputs, coordinates and physics loss."""

    model_id = "dfg_tanh_mlp_5x64"

    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            widths = (2, 64, 64, 64, 64, 64, 3)
            self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
            for layer in self.layers[:-1]:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        hidden = self.normalize(xy)
        for layer in self.layers[:-1]:
            hidden = torch.tanh(layer(hidden))
        return self.scale_output(self.layers[-1](hidden))


def build_model(arm: str, seed: int) -> nn.Module:
    if arm == "DFG-RBF-PIKAN":
        return DFGRBFPIKAN(seed)
    if arm == "DFG-Tanh-MLP":
        return DFGMLP(seed)
    raise ValueError(f"Unknown arm: {arm}")


def grad_scalar(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(value, xy, torch.ones_like(value), create_graph=True, retain_graph=True)[0]


def residual_components(fields: torch.Tensor, xy: torch.Tensor) -> dict[str, torch.Tensor]:
    u, v, p = (fields[:, i:i + 1] for i in range(3))
    gu, gv, gp = grad_scalar(u, xy), grad_scalar(v, xy), grad_scalar(p, xy)
    ux, uy = gu[:, 0:1], gu[:, 1:2]
    vx, vy = gv[:, 0:1], gv[:, 1:2]
    px, py = gp[:, 0:1], gp[:, 1:2]
    uxx = grad_scalar(ux, xy)[:, 0:1]
    uyy = grad_scalar(uy, xy)[:, 1:2]
    vxx = grad_scalar(vx, xy)[:, 0:1]
    vyy = grad_scalar(vy, xy)[:, 1:2]
    return {
        "momentum_x": (u * ux + v * uy + px - NU * (uxx + uyy)) / MOMENTUM_SCALE,
        "momentum_y": (u * vx + v * vy + py - NU * (vxx + vyy)) / MOMENTUM_SCALE,
        "continuity": (ux + vy) / CONTINUITY_SCALE,
    }


def sync_elapsed(started: float, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def train_one(arm: str, seed: int, pools: dict[str, np.ndarray], provenance: dict,
              output: Path, device: torch.device, cap: float, *, smoke: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(arm, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    generator = torch.Generator(device=device).manual_seed(seed + 100000)
    tensors = {name: torch.as_tensor(value, dtype=torch.float32, device=device)
               for name, value in pools.items()}
    started = time.perf_counter()
    step = 0
    history: list[dict] = []
    next_checkpoint = 0

    def batch(name: str, requested: int) -> torch.Tensor:
        source = tensors[name]
        indices = torch.randint(source.shape[0], (min(requested, source.shape[0]),),
                                generator=generator, device=device)
        return source.index_select(0, indices)

    def save_checkpoint(elapsed: float, label: str) -> None:
        torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                    "step": step, "elapsed_gpu_wall_seconds": elapsed, "label": label},
                   output / f"checkpoint-{label}.pt")

    while True:
        if sync_elapsed(started, device) >= cap:
            break
        optimizer.zero_grad(set_to_none=True)
        interior = batch("interior", INTERIOR_BATCH).detach().requires_grad_(True)
        residuals = residual_components(model(interior), interior)
        losses = {name: value.square().mean() for name, value in residuals.items()}

        inlet = batch("inlet", BOUNDARY_BATCH)
        y = inlet[:, 1:2]
        inlet_u = 4.0 * U_MAX * y * (H - y) / (H * H)
        inlet_target = torch.cat((inlet_u, torch.zeros_like(inlet_u)), dim=1)
        losses["inlet"] = ((model(inlet)[:, :2] - inlet_target) / U_MEAN).square().mean()
        zero_target = torch.zeros((BOUNDARY_BATCH, 2), dtype=torch.float32, device=device)
        for name in ("lower_wall", "upper_wall", "cylinder"):
            boundary = batch(name, BOUNDARY_BATCH)
            losses[name] = ((model(boundary)[:, :2] - zero_target) / U_MEAN).square().mean()

        outlet = batch("outlet", BOUNDARY_BATCH).detach().requires_grad_(True)
        outlet_fields = model(outlet)
        outlet_gu = grad_scalar(outlet_fields[:, 0:1], outlet)
        outlet_gv = grad_scalar(outlet_fields[:, 1:2], outlet)
        outlet_traction_x = (NU * outlet_gu[:, 0:1] - outlet_fields[:, 2:3]) / PRESSURE_SCALE
        outlet_traction_y = NU * outlet_gv[:, 0:1] / PRESSURE_SCALE
        losses["outlet"] = outlet_traction_x.square().mean() + outlet_traction_y.square().mean()

        total = torch.stack(list(losses.values())).sum()
        if not torch.isfinite(total):
            save_checkpoint(sync_elapsed(started, device), "nonfinite")
            raise FloatingPointError(f"non-finite loss in {arm}, seed {seed}, step {step + 1}")
        total.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            save_checkpoint(sync_elapsed(started, device), "nonfinite-gradient")
            raise FloatingPointError(f"non-finite gradient in {arm}, seed {seed}, step {step + 1}")
        optimizer.step()
        step += 1
        elapsed = sync_elapsed(started, device)
        if step % 100 == 0:
            history.append({"step": step, "elapsed_gpu_wall_seconds": elapsed,
                            "total_loss": float(total.detach().cpu()),
                            **{name: float(value.detach().cpu()) for name, value in losses.items()}})
        while next_checkpoint < len(CHECKPOINTS) and elapsed >= CHECKPOINTS[next_checkpoint]:
            save_checkpoint(elapsed, f"t{int(CHECKPOINTS[next_checkpoint])}")
            next_checkpoint += 1

    elapsed = sync_elapsed(started, device)
    save_checkpoint(elapsed, "final")
    with (output / "training_history.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {
        "status": STATUS, "arm": arm, "seed": seed,
        "device": torch.cuda.get_device_name(device), "dtype": "float32",
        "model_id": model.model_id,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": step, "elapsed_gpu_wall_seconds": elapsed,
        "cap_overrun_seconds": max(0.0, elapsed - cap),
        "pool_provenance": provenance,
        "scientific_accepted": False,
        "field_relative_l2": None,
        "field_reference_status": "missing_converged_FEM_field",
        "smoke_only": smoke,
    }
    write_json(output / "training_summary.json", summary)
    return summary


def qoi_for_model(model: nn.Module, device: torch.device, surface_count: int = 512,
                  deadline: float | None = None) -> dict:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("campaign scoring allowance exhausted before QoI quadrature")
    theta = 2.0 * math.pi * np.arange(surface_count, dtype=np.float64) / surface_count
    surface_np = np.column_stack((CX + RADIUS * np.cos(theta), CY + RADIUS * np.sin(theta)))
    surface = torch.as_tensor(surface_np, dtype=torch.float32, device=device).requires_grad_(True)
    fields = model(surface)
    gu = grad_scalar(fields[:, 0:1], surface)
    gv = grad_scalar(fields[:, 1:2], surface)
    du_dx, du_dy = gu[:, 0].detach().cpu().numpy(), gu[:, 1].detach().cpu().numpy()
    dv_dx, dv_dy = gv[:, 0].detach().cpu().numpy(), gv[:, 1].detach().cpu().numpy()
    pressure = fields[:, 2].detach().cpu().numpy()
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("campaign scoring allowance exhausted during QoI quadrature")
    nx, ny = np.cos(theta), np.sin(theta)
    tx, ty = -ny, nx
    normal_du = du_dx * nx + du_dy * ny
    normal_dv = dv_dx * nx + dv_dy * ny
    du_tau_dn = tx * normal_du + ty * normal_dv
    ds = 2.0 * math.pi * RADIUS / surface_count
    drag_force = float(np.sum(NU * du_tau_dn * ny - pressure * nx) * ds)
    lift_force = float(-np.sum(NU * du_tau_dn * nx + pressure * ny) * ds)
    scale = 2.0 / (U_MEAN**2 * DIAMETER)
    front = torch.tensor([[CX - RADIUS, CY]], dtype=torch.float32, device=device)
    rear = torch.tensor([[CX + RADIUS, CY]], dtype=torch.float32, device=device)
    front_pressure = float(model(front)[0, 2].detach().cpu())
    rear_pressure = float(model(rear)[0, 2].detach().cpu())
    return {"drag_force": drag_force, "lift_force": lift_force,
            "drag_coefficient": scale * drag_force,
            "lift_coefficient": scale * lift_force,
            "pressure_front": front_pressure, "pressure_rear": rear_pressure,
            "pressure_difference": front_pressure - rear_pressure,
            "surface_count": surface_count}


def heldout_diagnostics(model: nn.Module, points: np.ndarray, pools: dict[str, np.ndarray],
                        device: torch.device, deadline: float | None = None) -> dict:
    residual_sq = {"momentum_x": 0.0, "momentum_y": 0.0, "continuity": 0.0}
    count = 0
    for start in range(0, len(points), 256):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("campaign scoring allowance exhausted during held-out diagnostics")
        xy = torch.as_tensor(points[start:start + 256], dtype=torch.float32, device=device).requires_grad_(True)
        residuals = residual_components(model(xy), xy)
        n = xy.shape[0]
        for name, value in residuals.items():
            residual_sq[name] += float(value.detach().square().sum().cpu())
        count += n
    result = {f"heldout_{name}_rms": math.sqrt(value / count) for name, value in residual_sq.items()}
    # Boundary diagnostics use a disjoint deterministic set from the training pools.
    for name in ("inlet", "lower_wall", "upper_wall", "cylinder"):
        points_b = torch.as_tensor(pools[f"validation_{name}"], dtype=torch.float32, device=device)
        velocity = model(points_b)[:, :2]
        if name == "inlet":
            target_u = 4.0 * U_MAX * points_b[:, 1:2] * (H - points_b[:, 1:2]) / (H * H)
            target = torch.cat((target_u, torch.zeros_like(target_u)), dim=1)
        else:
            target = torch.zeros_like(velocity)
        result[f"{name}_velocity_rmse_over_umean"] = float(torch.sqrt(((velocity - target) / U_MEAN).square().mean()).cpu())
    return result


def score_run(arm: str, seed: int, run_dir: Path, pools: dict[str, np.ndarray],
              device: torch.device, deadline: float) -> dict:
    checkpoint = torch.load(run_dir / "checkpoint-final.pt", map_location=device, weights_only=False)
    model = build_model(arm, seed).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    qoi512 = qoi_for_model(model, device, 512, deadline)
    qoi1024 = qoi_for_model(model, device, 1024, deadline)
    errors = {
        "drag_relative_error_vs_featflow": abs(qoi512["drag_coefficient"] - FEATFLOW["drag_coefficient"]) / FEATFLOW["drag_coefficient"],
        "lift_absolute_error_vs_featflow": abs(qoi512["lift_coefficient"] - FEATFLOW["lift_coefficient"]),
        "pressure_difference_relative_error_vs_featflow": abs(qoi512["pressure_difference"] - FEATFLOW["pressure_difference"]) / FEATFLOW["pressure_difference"],
    }
    sensitivity = {name: abs(qoi1024[name] - qoi512[name]) for name in
                   ("drag_coefficient", "lift_coefficient", "pressure_difference")}
    result = {"status": STATUS, "arm": arm, "seed": seed,
              "qois": qoi512, "featflow_reference": FEATFLOW,
              "featflow_source": FEATFLOW_URL, "qoi_errors": errors,
              "quadrature_512_to_1024_absolute_change": sensitivity,
              "diagnostics": heldout_diagnostics(model, pools["heldout"], pools, device, deadline),
              "field_relative_l2": None,
              "field_reference_status": "missing_converged_FEM_field",
              "scientific_accepted": False}
    write_json(run_dir / "score.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="seed 500, 120 seconds per arm, no QoI score")
    parser.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--worker-cap", type=float, default=RUN_CAP_SECONDS)
    args = parser.parse_args()
    if args.worker_arm is not None:
        if args.worker_seed is None:
            parser.error("--worker-seed is required with --worker-arm")
        worker_device = require_t4(worker=True)
        worker_pools, worker_provenance = make_pools(args.worker_seed)
        train_one(args.worker_arm, args.worker_seed, worker_pools, worker_provenance,
                  args.output, worker_device, args.worker_cap)
        return

    campaign_started = time.perf_counter()
    campaign_deadline = campaign_started + SCRIPT_WALL_CAP_SECONDS
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; use a new run path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = require_t4()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(output / "hardware.json", {"device": torch.cuda.get_device_name(device),
                                           "torch": torch.__version__, "cuda": torch.version.cuda,
                                           "python": platform.python_version(), "status": STATUS})
    run_seeds = [500] if args.smoke else SEEDS
    cap = 120.0 if args.smoke else RUN_CAP_SECONDS
    results = []
    scoring_started = time.perf_counter()
    scoring_deadline = min(campaign_deadline, scoring_started + SCORING_CAP_SECONDS)
    for seed in run_seeds:
        pools, provenance = make_pools(seed)
        pool_path = output / "pools" / f"seed-{seed}.npz"
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pool_path, **pools)
        if args.smoke:
            for arm in ARMS:
                run_dir = output / arm / f"seed-{seed}"
                train_one(arm, seed, pools, provenance, run_dir, device, cap, smoke=True)
                results.append({"arm": arm, "seed": seed, "status": "smoke_complete", "qois_scored": False})
            continue
        if time.perf_counter() + RUN_CAP_SECONDS + 5.0 >= campaign_deadline:
            break
        pair_status = launch_pair(seed, output, campaign_deadline)
        for arm in ARMS:
            run_dir = output / arm / f"seed-{seed}"
            if not pair_status[arm]:
                results.append({"arm": arm, "seed": seed, "status": "training_failed",
                                "field_relative_l2": None})
                continue
            if time.perf_counter() >= scoring_deadline:
                results.append({"arm": arm, "seed": seed, "status": "score_budget_exhausted",
                                "field_relative_l2": None})
                continue
            try:
                results.append(score_run(arm, seed, run_dir, pools, device, scoring_deadline))
            except Exception as exc:
                run_dir.mkdir(parents=True, exist_ok=True)
                failure = {"arm": arm, "seed": seed, "status": "score_failed_preserved",
                           "exception": f"{type(exc).__name__}: {exc}",
                           "field_relative_l2": None}
                write_json(run_dir / "failure.json", failure)
                results.append(failure)
    source_path = Path(__file__).resolve()
    summary = {"status": STATUS, "campaign_id": CAMPAIGN_ID,
               "seeds": run_seeds, "smoke_only": args.smoke,
               "campaign_wall_seconds": time.perf_counter() - campaign_started,
               "session_quota_cap_seconds": SESSION_QUOTA_CAP_SECONDS,
               "script_wall_cap_seconds": SCRIPT_WALL_CAP_SECONDS,
               "results": results, "source_sha256": sha256(source_path),
               "featflow_qois": FEATFLOW, "field_reference_status": "missing_converged_FEM_field",
               "scientific_accepted": False}
    write_json(output / ("smoke_status.json" if args.smoke else "campaign_results.json"), summary)


if __name__ == "__main__":
    main()
