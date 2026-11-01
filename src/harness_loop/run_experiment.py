"""Lightweight harness loop for protocol-aligned PIKAN experiments.

This file is intentionally conservative: it records the experiment contract,
creates a durable filesystem memory entry, and logs to MLflow if available.
Connect `execute_training` to an existing notebook/script when ready.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: PyYAML. Install with `python -m pip install pyyaml`."
    ) from exc


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child_prefix, nested))
        return result
    if isinstance(value, list):
        return {prefix: ",".join(str(item) for item in value)}
    return {prefix: value}


def execute_training(
    manifest: dict[str, Any],
    manifest_path: Path,
    run_dir: Path,
    command: list[str] | None,
) -> dict[str, Any]:
    """Run the configured training command or emit a dry-run summary.

    The command should write real metrics/artifacts later. For now, this
    harness records execution status and elapsed time.
    """

    started = time.perf_counter()
    if not command:
        return {
            "status": "dry_run",
            "message": "No training command provided.",
            "elapsed_sec": 0.0,
            "metrics": {},
        }

    env = os.environ.copy()
    env["HARNESS_RUN_DIR"] = str(run_dir)
    env["HARNESS_MANIFEST"] = str(manifest_path)

    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - started
    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "status": "completed" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
        "metrics": metrics,
    }


def create_run_dir(
    manifest_path: Path,
    manifest: dict[str, Any],
    memory_root: Path,
) -> Path:
    iteration_id = str(manifest["iteration"]["id"])
    run_dir = memory_root / iteration_id / utc_now().replace(":", "-")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest_snapshot.yaml").write_text(
        manifest_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return run_dir


def write_summary(
    manifest_path: Path,
    manifest: dict[str, Any],
    result: dict[str, Any],
    run_dir: Path,
) -> None:
    summary = {
        "created_at": utc_now(),
        "manifest_path": str(manifest_path),
        "iteration": manifest.get("iteration", {}),
        "hypothesis": manifest.get("hypothesis", {}),
        "result": result,
    }

    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def log_mlflow(manifest: dict[str, Any], result: dict[str, Any], run_dir: Path) -> str:
    try:
        import mlflow
    except ImportError:
        return "mlflow_not_installed"

    mlflow_config = manifest.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mlflow_config.get("experiment_name", "pikan_harness")
    mlflow.set_experiment(experiment_name)

    run_name = str(manifest.get("iteration", {}).get("id", "unnamed_iteration"))
    with mlflow.start_run(run_name=run_name):
        for key, value in mlflow_config.get("tags", {}).items():
            mlflow.set_tag(key, value)
        for key, value in flatten("", manifest).items():
            if value is not None and len(str(value)) <= 250:
                mlflow.log_param(key, value)
        mlflow.log_metric("harness_elapsed_sec", float(result.get("elapsed_sec", 0.0)))
        for key, value in result.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
        mlflow.set_tag("harness_status", result.get("status", "unknown"))
        mlflow.log_artifacts(str(run_dir))

    return "logged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="src/harness_loop/experiment_manifest.yaml",
        help="Path to the experiment manifest.",
    )
    parser.add_argument(
        "--memory-root",
        default="tmp/harness_runs",
        help="Filesystem memory root for harness run records.",
    )
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="Optional training command, for example: --command python train.py --config x.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    memory_root = Path(args.memory_root).resolve()

    manifest = load_manifest(manifest_path)
    run_dir = create_run_dir(manifest_path, manifest, memory_root)
    result = execute_training(manifest, manifest_path, run_dir, args.command)
    write_summary(manifest_path, manifest, result, run_dir)
    mlflow_status = log_mlflow(manifest, result, run_dir)

    print(
        json.dumps(
            {
                "status": result["status"],
                "run_dir": str(run_dir),
                "mlflow": mlflow_status,
            },
            indent=2,
        )
    )
    return 0 if result["status"] in {"completed", "dry_run"} else 1


if __name__ == "__main__":
    sys.exit(main())
