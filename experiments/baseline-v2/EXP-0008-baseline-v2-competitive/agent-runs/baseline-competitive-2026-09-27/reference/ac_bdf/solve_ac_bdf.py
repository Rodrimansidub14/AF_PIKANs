#!/usr/bin/env python3
"""Candidate Allen–Cahn reference: finite differences in space, BDF in time.

This program writes a reproducible candidate and convergence evidence. A passing
run is not, by itself, approval to use the data in an evaluation leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.sparse import diags


D = 0.001
X_LEFT = -1.0
X_RIGHT = 1.0
T_FINAL = 1.0
REACTION_COEFFICIENT = 5.0
N_INTERVALS = (
    ("AC-L0", 256),
    ("AC-L1", 512),
    ("AC-L2", 1024),
    ("AC-L3", 2048),
    ("AC-L4", 4096),
    ("AC-L5", 8192),
)
T_OUTPUT = np.linspace(0.0, T_FINAL, 201, dtype=np.float64)
SPACE_TIME_TOL = 5e-5
SNAPSHOT_TOL = 5e-5
TIME_TOL = 1e-7
TIME_REFINEMENT_LIMIT = 8
NESTED_PAIR_MIN_FINE_N = 4096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}Z-{uuid.uuid4().hex[:8]}"


def initial_condition(x: np.ndarray) -> np.ndarray:
    return np.asarray(x * x * np.cos(np.pi * x), dtype=np.float64)


def grid_for_intervals(n_intervals: int) -> tuple[np.ndarray, float]:
    h = (X_RIGHT - X_LEFT) / n_intervals
    x = X_LEFT + h * np.arange(n_intervals + 1, dtype=np.float64)
    x[0], x[-1] = X_LEFT, X_RIGHT
    return x, h


def solve_bdf(
    n_intervals: int,
    *,
    rtol: float,
    atol: float,
    max_step: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve the method-of-lines system with a sparse analytic Jacobian."""
    x, h = grid_for_intervals(n_intervals)
    n_unknowns = n_intervals - 1
    inverse_h2 = 1.0 / (h * h)

    laplacian = diags(
        (
            np.full(n_unknowns - 1, inverse_h2, dtype=np.float64),
            np.full(n_unknowns, -2.0 * inverse_h2, dtype=np.float64),
            np.full(n_unknowns - 1, inverse_h2, dtype=np.float64),
        ),
        offsets=(-1, 0, 1),
        shape=(n_unknowns, n_unknowns),
        format="csc",
        dtype=np.float64,
    )
    boundary_contribution = np.zeros(n_unknowns, dtype=np.float64)
    boundary_contribution[0] -= inverse_h2
    boundary_contribution[-1] -= inverse_h2

    def rhs(_time: float, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return D * (laplacian @ values + boundary_contribution) + (
            REACTION_COEFFICIENT * (values - values**3)
        )

    def jacobian(_time: float, values: np.ndarray):
        reaction_diagonal = REACTION_COEFFICIENT * (1.0 - 3.0 * values**2)
        return D * laplacian + diags(reaction_diagonal, offsets=0, format="csc")

    y0 = initial_condition(x[1:-1])
    solution = solve_ivp(
        rhs,
        (0.0, T_FINAL),
        y0,
        method="BDF",
        t_eval=T_OUTPUT,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        jac=jacobian,
    )
    if not solution.success:
        raise RuntimeError(f"BDF failed for N={n_intervals}: {solution.message}")
    if solution.y.shape != (n_unknowns, T_OUTPUT.size):
        raise RuntimeError(
            f"BDF returned an unexpected shape for N={n_intervals}: "
            f"{solution.y.shape}"
        )

    full_solution = np.full(
        (n_intervals + 1, T_OUTPUT.size), -1.0, dtype=np.float64
    )
    full_solution[1:-1, :] = np.asarray(solution.y, dtype=np.float64)
    stats: dict[str, Any] = {
        "n_intervals": n_intervals,
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "nfev": int(solution.nfev),
        "njev": int(solution.njev),
        "nlu": int(solution.nlu),
        "status": int(solution.status),
        "message": str(solution.message),
        "returned_time_count": int(solution.t.size),
        "returned_final_time": float(solution.t[-1]),
    }
    return x, full_solution, stats


def _quadrature_weights(size: int, spacing: float) -> np.ndarray:
    weights = np.ones(size, dtype=np.float64)
    weights[0] = 0.5
    weights[-1] = 0.5
    return weights * spacing


def relative_l2_metrics(
    reference: np.ndarray,
    approximation: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> dict[str, float]:
    """Return trapezoidal space-time and worst-snapshot relative L2 errors."""
    reference = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(approximation, dtype=np.float64)
    if reference.shape != approximation.shape:
        raise ValueError("Reference and approximation shapes must match")
    if reference.shape != (x.size, t.size):
        raise ValueError("Solution array must have shape (len(x), len(t))")

    wx = _quadrature_weights(x.size, float(x[1] - x[0]))
    wt = _quadrature_weights(t.size, float(t[1] - t[0]))
    weights = wx[:, None] * wt[None, :]
    numerator = float(np.sum((reference - approximation) ** 2 * weights))
    denominator = float(np.sum(reference**2 * weights))
    if denominator <= 0.0:
        raise ValueError("Relative L2 denominator is zero")

    snapshot_numerator = np.sum((reference - approximation) ** 2 * wx[:, None], axis=0)
    snapshot_denominator = np.sum(reference**2 * wx[:, None], axis=0)
    if np.any(snapshot_denominator <= 0.0):
        raise ValueError("At least one snapshot has a zero relative L2 denominator")
    return {
        "space_time_relative_l2": float(np.sqrt(numerator / denominator)),
        "max_snapshot_relative_l2": float(
            np.max(np.sqrt(snapshot_numerator / snapshot_denominator))
        ),
    }


def compare_nested_levels(
    coarse_x: np.ndarray,
    coarse_u: np.ndarray,
    fine_x: np.ndarray,
    fine_u: np.ndarray,
) -> dict[str, float]:
    if coarse_x.size * 2 - 1 != fine_x.size:
        raise ValueError("Spatial levels are not nested by a factor of two")
    # N is a power of two, so every coarse node is a fine node at an even index.
    common_fine_u = fine_u[::2, :]
    return relative_l2_metrics(common_fine_u, coarse_u, coarse_x, T_OUTPUT)


def write_level(path: Path, x: np.ndarray, u: np.ndarray, level: str) -> None:
    np.savez_compressed(
        path,
        x=np.asarray(x, dtype=np.float64),
        t=T_OUTPUT,
        u=np.asarray(u, dtype=np.float64),
        level=np.asarray(level),
    )


def write_convergence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "level",
        "n_intervals",
        "h",
        "space_time_relative_l2_vs_coarse",
        "max_snapshot_relative_l2_vs_coarse",
        "pair_pass",
        "passing_pair_streak",
        "nfev",
        "njev",
        "nlu",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_hash_manifest(run_dir: Path, script_path: Path, readme_path: Path) -> None:
    hash_path = run_dir / "SHA256SUMS.txt"
    files: list[tuple[str, Path]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path != hash_path:
            files.append((path.relative_to(run_dir).as_posix(), path))
    files.extend(
        [
            ("source/solve_ac_bdf.py", script_path),
            ("source/README.md", readme_path),
        ]
    )
    lines = [f"{sha256_file(path)}  {label}" for label, path in files]
    hash_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deepxde_grid(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_common = np.linspace(X_LEFT, X_RIGHT, 201, dtype=np.float64)
    t_common = T_OUTPUT[::2]
    u_common = np.empty((x_common.size, t_common.size), dtype=np.float64)
    for j, time_index in enumerate(range(0, T_OUTPUT.size, 2)):
        u_common[:, j] = np.interp(x_common, x, u[:, time_index])
    return x_common, t_common, u_common


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Directory under which a unique run directory is created (default: runs).",
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique directory name; omit to generate a UTC timestamp ID.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or utc_run_id()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise SystemExit("--run-id must contain only letters, digits, '.', '_' or '-'")

    output_root = args.output_root.expanduser().resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    script_path = Path(__file__).resolve()
    readme_path = script_path.with_name("README.md")

    level_stats: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    time_stats: list[dict[str, Any]] = []
    pair_passes: list[bool] = []
    final_level: dict[str, Any] | None = None
    tight_solution: np.ndarray | None = None
    spatial_converged = False
    temporal_converged = False
    error_message: str | None = None
    return_code = 1

    try:
        for level, n_intervals in N_INTERVALS:
            x, u, stats = solve_bdf(
                n_intervals,
                rtol=1e-10,
                atol=1e-12,
                max_step=0.005,
            )
            level_stats.append({"level": level, **stats})
            write_level(run_dir / f"{level.lower()}.npz", x, u, level)
            current: dict[str, Any] = {
                "level": level,
                "n_intervals": n_intervals,
                "h": (X_RIGHT - X_LEFT) / n_intervals,
                "space_time_relative_l2_vs_coarse": "",
                "max_snapshot_relative_l2_vs_coarse": "",
                "pair_pass": "",
                "passing_pair_streak": 0,
                "nfev": stats["nfev"],
                "njev": stats["njev"],
                "nlu": stats["nlu"],
            }
            if final_level is not None:
                metrics = compare_nested_levels(
                    final_level["x"], final_level["u"], x, u
                )
                pair_pass = (
                    metrics["space_time_relative_l2"] <= SPACE_TIME_TOL
                    and metrics["max_snapshot_relative_l2"] <= SNAPSHOT_TOL
                )
                pair_passes.append(pair_pass)
                streak = 0
                for passed in reversed(pair_passes):
                    if not passed:
                        break
                    streak += 1
                current.update(
                    {
                        "space_time_relative_l2_vs_coarse": metrics[
                            "space_time_relative_l2"
                        ],
                        "max_snapshot_relative_l2_vs_coarse": metrics[
                            "max_snapshot_relative_l2"
                        ],
                        "pair_pass": pair_pass,
                        "passing_pair_streak": streak,
                    }
                )
            convergence_rows.append(current)
            final_level = {"level": level, "n_intervals": n_intervals, "x": x, "u": u}

            if (
                n_intervals >= NESTED_PAIR_MIN_FINE_N
                and len(pair_passes) >= 2
                and pair_passes[-1]
                and pair_passes[-2]
            ):
                spatial_converged = True
                break

        if final_level is None:
            raise RuntimeError("No spatial level completed")

        base_x = final_level["x"]
        base_u = final_level["u"]
        for refinement in range(1, TIME_REFINEMENT_LIMIT + 1):
            max_step = 0.005 / (2**refinement)
            tight_x, tight_u, stats = solve_bdf(
                final_level["n_intervals"],
                rtol=1e-12,
                atol=1e-14,
                max_step=max_step,
            )
            tight_metrics = relative_l2_metrics(base_u, tight_u, base_x, T_OUTPUT)
            time_stats.append(
                {
                    "refinement": refinement,
                    "max_step": max_step,
                    **stats,
                    **tight_metrics,
                }
            )
            tight_solution = tight_u
            temporal_converged = (
                tight_metrics["space_time_relative_l2"] <= TIME_TOL
                and tight_metrics["max_snapshot_relative_l2"] <= TIME_TOL
            )
            if temporal_converged:
                break

        candidate_passed = spatial_converged and temporal_converged
        write_level(
            run_dir / "ac_candidate_finest.npz",
            final_level["x"],
            final_level["u"],
            final_level["level"],
        )
        if tight_solution is not None:
            np.savez_compressed(
                run_dir / "ac_time_check.npz",
                x=np.asarray(final_level["x"], dtype=np.float64),
                t=T_OUTPUT,
                u_tight=np.asarray(tight_solution, dtype=np.float64),
            )
        if candidate_passed:
            write_level(
                run_dir / "ac_reference.npz",
                final_level["x"],
                final_level["u"],
                final_level["level"],
            )
            x_dx, t_dx, u_dx = deepxde_grid(final_level["x"], final_level["u"])
            np.savez_compressed(
                run_dir / "ac_reference_deepxde_grid.npz",
                x=x_dx,
                t=t_dx,
                usol=u_dx,
            )
            return_code = 0
        else:
            return_code = 2

    except Exception as exc:  # Preserve partial logs and hashes for diagnosis.
        error_message = f"{type(exc).__name__}: {exc}"
        return_code = 1

    write_convergence_csv(run_dir / "convergence.csv", convergence_rows)
    (run_dir / "solver_stats.json").write_text(
        json.dumps(
            {"spatial_levels": level_stats, "time_refinements": time_stats},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    environment_lock = (
        f"python=={platform.python_version()}\n"
        f"numpy=={np.__version__}\n"
        f"scipy=={scipy.__version__}\n"
        f"implementation={platform.python_implementation()}\n"
    )
    (run_dir / "environment.lock").write_text(environment_lock, encoding="utf-8")

    candidate_passed = spatial_converged and temporal_converged and error_message is None
    output_names = [
        path.relative_to(run_dir).as_posix()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    output_names.extend(["manifest.json", "SHA256SUMS.txt"])
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "numerically_converged_candidate"
            if candidate_passed
            else ("solver_error" if error_message else "candidate_did_not_meet_thresholds")
        ),
        "evaluation_ready": False,
        "promotion_requires_external_review": True,
        "problem": {
            "equation": "u_t = 0.001 u_xx + 5 (u - u^3)",
            "x_domain": [X_LEFT, X_RIGHT],
            "t_domain": [0.0, T_FINAL],
            "initial_condition": "u(x,0) = x^2 cos(pi x)",
            "dirichlet_boundary": "u(-1,t) = u(1,t) = -1",
        },
        "method": {
            "space": "uniform-grid second-order centered finite differences",
            "time": "scipy.integrate.solve_ivp(method='BDF')",
            "jacobian": "analytic sparse tridiagonal",
            "dtype": "float64",
            "output_times": "0.005*k, k=0..200",
        },
        "thresholds": {
            "space_time_relative_l2": SPACE_TIME_TOL,
            "max_snapshot_relative_l2": SNAPSHOT_TOL,
            "required_consecutive_refinement_pairs": 2,
            "minimum_fine_intervals_before_spatial_stop": NESTED_PAIR_MIN_FINE_N,
            "time_check_relative_l2": TIME_TOL,
            "time_refinement_limit": TIME_REFINEMENT_LIMIT,
        },
        "levels": [
            {"level": level, "n_intervals": n} for level, n in N_INTERVALS
        ],
        "completed_levels": [row["level"] for row in convergence_rows],
        "final_level": final_level["level"] if final_level else None,
        "spatial_converged": spatial_converged,
        "temporal_converged": temporal_converged,
        "error": error_message,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "command": sys.argv,
        "working_directory": str(Path.cwd()),
        "script_sha256": sha256_file(script_path),
        "readme_sha256": sha256_file(readme_path),
        "outputs": output_names,
        "reference_note": (
            "Candidate only. This script never marks the result evaluation-ready."
        ),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_hash_manifest(run_dir, script_path, readme_path)

    print(f"Run directory: {run_dir}")
    print(f"Candidate status: {manifest['status']}")
    print("Evaluation ready: False")
    if error_message:
        print(f"Error: {error_message}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
