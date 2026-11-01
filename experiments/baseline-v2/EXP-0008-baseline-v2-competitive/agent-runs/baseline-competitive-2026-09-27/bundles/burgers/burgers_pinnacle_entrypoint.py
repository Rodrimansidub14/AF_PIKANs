"""One paired seed: fixed Sobol RBF arm and PINNacle Vanilla, no scoring."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np

POOL_KEYS = ("interior", "diagnostic", "left", "right", "initial")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.asarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    manifest_path = Path(os.environ["HARNESS_MANIFEST"]).resolve()
    run_dir = Path(os.environ["HARNESS_RUN_DIR"]).resolve()
    seed = int(os.environ["HARNESS_SEED"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("budget", {}).get("approved") is not True:
        raise RuntimeError("proposed budget is not approved; no training may start")
    if manifest.get("research_status", {}).get("execution_authorized") is not True:
        raise RuntimeError("execution is not authorized in the reviewed manifest")
    training = manifest["training"]
    if seed not in training["seeds"]:
        raise ValueError(f"unexpected paired seed {seed}")
    seconds = int(training.get("seconds_per_arm", 900))
    steps = int(training.get("steps_per_arm", 20000))
    if not 0 < seconds <= 900 or not 0 < steps <= 20000:
        raise ValueError("arm limits exceed 900 seconds or 20000 Adam updates")
    run_dir.mkdir(parents=True, exist_ok=True)  # runtime.run_bundle creates the seed directory.

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one assigned CUDA device per seed worker")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name.upper():
        raise RuntimeError(f"expected a Tesla T4 worker, received {device_name}")

    bundle_root = Path(__file__).resolve().parents[3]
    if str(bundle_root) not in sys.path:
        sys.path.insert(0, str(bundle_root))
    from src.research_pipeline.sciml.sampling import pools
    from src.research_pipeline.remote.burgers_paired_training import train_one
    import scipy

    # Generate the five loss pools once, then pass those same arrays to both arms.
    generated = pools("burgers", seed, n=8192, boundary=512, heldout=16384)
    shared = {name: np.asarray(generated[name], dtype=np.float64) for name in POOL_KEYS}
    shared_path = run_dir / "paired_pools.npz"
    np.savez_compressed(shared_path, **shared)
    hashes = {name: array_hash(shared[name]) for name in POOL_KEYS}
    content_hash = sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("ascii"))
    pool_metadata = {
        "origin": "generated once by src.research_pipeline.sciml.sampling.pools",
        "seed": seed,
        "scipy_version": scipy.__version__,
        "seed_streams": {"interior": 10 * seed + 1, "diagnostic": 10 * seed + 2,
                         "left_right_time": 10 * seed + 3, "initial_space": 10 * seed + 4},
        "array_sha256": hashes, "content_sha256": content_hash,
        "counts": {name: int(len(shared[name])) for name in POOL_KEYS},
    }
    write_json(run_dir / "paired_pool_manifest.json", {
        "seed": seed, "source": pool_metadata["origin"], "scipy_version": scipy.__version__,
        "arrays": {name: {"shape": list(shared[name].shape), "dtype": shared[name].dtype.str,
                           "sha256": hashes[name]} for name in POOL_KEYS},
        "exactly_shared": True, "archive": shared_path.name,
        "archive_sha256": sha256_file(shared_path), "content_sha256": content_hash,
    })
    problem_config = {"definition": "raissi_pinnacle_1d", "nu": 0.01 / np.pi,
                      "field_scale": 1.0, "residual_scale": 2.003183098861838,
                      "evaluation_id": "burgers_pinnacle_1d"}
    write_json(run_dir / "config.json", {
        "manifest": manifest, "seed": seed, "pool": pool_metadata,
        "problem_config": problem_config,
        "training_limits_per_arm": {"seconds": seconds, "updates": steps},
    })
    write_json(run_dir / "versions.json", {
        "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "cuda_runtime": torch.version.cuda,
    })
    write_json(run_dir / "hardware.json", {
        "gpu_name": device_name, "visible_device_count": int(torch.cuda.device_count()),
        "cuda_runtime": torch.version.cuda, "gpu_slot": os.environ.get("HARNESS_GPU_SLOT"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing": "one assigned CUDA device; synchronized time includes training and logging",
    })

    rbf_dir = run_dir / "burgers-rbf"
    vanilla_dir = run_dir / "burgers-vanilla"
    rbf_cost = train_one("rbf", seed, shared, rbf_dir, seconds, steps, pool_metadata)
    vanilla_cost = train_one("vanilla", seed, shared, vanilla_dir, seconds, steps, pool_metadata)
    if rbf_cost["pool_content_sha256"] != vanilla_cost["pool_content_sha256"] or rbf_cost["pool_content_sha256"] != content_hash:
        raise ValueError("paired pool hash mismatch: both arms must receive identical arrays")
    if rbf_cost.get("scoring_performed") or vanilla_cost.get("scoring_performed"):
        raise RuntimeError("training entrypoint must never invoke scoring")

    write_json(rbf_dir / "training_summary.json", {
        "steps": rbf_cost["completed_updates"],
        "training_and_diagnostics_sec": rbf_cost["allocated_gpu_seconds"],
        "parameters": int(rbf_cost["parameter_count"]),
        "model_id": "rbf_fixed", "model_family": "rbf", "stop_reason": rbf_cost["stop_reason"],
    })
    metrics = {
        "rbf/completed_updates": int(rbf_cost["completed_updates"]),
        "rbf/allocated_gpu_seconds": float(rbf_cost["allocated_gpu_seconds"]),
        "vanilla/completed_updates": int(vanilla_cost["completed_updates"]),
        "vanilla/allocated_gpu_seconds": float(vanilla_cost["allocated_gpu_seconds"]),
        "paired/shared_pool_exact": 1,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "problem_status.json", {
        "seed": seed, "status": "completed", "shared_pool_exact": True,
        "arms": {"rbf": rbf_cost["stop_reason"], "vanilla": vanilla_cost["stop_reason"]},
    })
    adapter = bundle_root / "var/agent_runs/baseline-competitive-2026-09-27/adapters/burgers_pinnacle.py"
    bundle_index = bundle_root / "bundle.json"
    bundle_id = json.loads(bundle_index.read_text(encoding="utf-8")).get("bundle_id") if bundle_index.is_file() else None
    write_json(run_dir / "provenance.json", {
        "bundle_id": bundle_id, "seed": seed,
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(adapter), "paired_pool_sha256": hashes,
        "training_scoring_calls": 0,
    })
    (run_dir / "decision_note.md").write_text(
        "Paired training artifacts only. RBF and Vanilla received identical per-seed Sobol arrays. "
        "No protected evaluator or field score was called during training. Vanilla used Adam only; "
        "no L-BFGS stage was appended. Any later score is confirmatory only, never for tuning. "
        "Scientific acceptance is false.\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


