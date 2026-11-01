from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "kaggle-outputs"
AC_OUTPUT = OUTPUTS / "allen-cahn-t4x2-v1-20260928" / "allen-cahn-t4x2"
S1_OUTPUT = OUTPUTS / "closure-s1-20260928" / "closure-s1"
SEEDS = [401, 402, 403, 404, 405]
ARMS = ["AC-RBF-Sobol", "AC-DeepXDE-MLP"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_repo(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    source = AC_OUTPUT / "run_campaign.py"
    s1a_path = S1_OUTPUT / "allen_cahn" / "s1a_summary.json"
    s1a_summary = json.loads(s1a_path.read_text(encoding="utf-8"))
    if s1a_summary.get("status") != "candidate_complete":
        raise RuntimeError("S3 is allowed only after S1a candidate_complete.")

    checkpoint_hashes = {}
    for arm in ARMS:
        checkpoint_hashes[arm] = {}
        for seed in SEEDS:
            checkpoint = AC_OUTPUT / "campaign" / arm / f"seed-{seed}" / "checkpoint-final.pt"
            checkpoint_hashes[arm][str(seed)] = sha256(checkpoint)
    pool_hashes = {
        str(seed): sha256(AC_OUTPUT / "campaign" / "pools" / f"seed-{seed}.npz")
        for seed in SEEDS
    }
    reference_files = {
        "FD8192": AC_OUTPUT / "campaign" / "reference" / "ac_gpu_reference_candidate.npz",
        "ETDRK4": S1_OUTPUT / "allen_cahn" / "etdrk4_reference.npz",
    }
    reference_hashes = {name: sha256(path) for name, path in reference_files.items()}
    config = {
        "session": "S3",
        "seeds": SEEDS,
        "arms": ARMS,
        "s1a_status": s1a_summary["status"],
        "input_roots": {
            "allen_cahn": "/kaggle/input/notebooks/rodrimandub/af-pikans-allen-cahn-t4x2-20260927/allen-cahn-t4x2",
            "s1": "/kaggle/input/notebooks/rodrimandub/af-pikans-competitive-baseline-s1-20260928/closure-s1",
        },
        "reference_paths": {
            "FD8192": "campaign/reference/ac_gpu_reference_candidate.npz",
            "ETDRK4": "allen_cahn/etdrk4_reference.npz",
        },
        "model_source_sha256": sha256(source),
        "checkpoint_hashes": checkpoint_hashes,
        "pool_hashes": pool_hashes,
        "reference_hashes": reference_hashes,
        "checkpoint_local_paths": {
            arm: {
                str(seed): relative_to_repo(AC_OUTPUT / "campaign" / arm / f"seed-{seed}" / "checkpoint-final.pt")
                for seed in SEEDS
            }
            for arm in ARMS
        },
        "pool_local_paths": {
            str(seed): relative_to_repo(AC_OUTPUT / "campaign" / "pools" / f"seed-{seed}.npz")
            for seed in SEEDS
        },
        "reference_local_paths": {name: relative_to_repo(path) for name, path in reference_files.items()},
    }
    (OUT / "s3-source-config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    files = {
        "ac_lbfgs_worker.py": (OUT / "ac_lbfgs_worker.py").read_bytes(),
        "s3-source-config.json": (OUT / "s3-source-config.json").read_bytes(),
    }
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    payload = packed.getvalue()
    template = (OUT / "ac_lbfgs_campaign.py").read_text(encoding="utf-8")
    if template.count("__S3_PAYLOAD_B64__") != 1:
        raise RuntimeError("S3 payload marker is missing or duplicated.")
    entrypoint = template.replace("__S3_PAYLOAD_B64__", base64.b64encode(payload).decode("ascii"))
    entrypoint_path = OUT / "ac-lbfgs-continuation-s3.py"
    entrypoint_path.write_text(entrypoint, encoding="utf-8", newline="\n")

    kernel_id = "rodrimandub/af-pikans-ac-lbfgs-continuation-s3-retry-20260928"
    kernel_sources = [
        "rodrimandub/af-pikans-allen-cahn-t4x2-20260927",
        "rodrimandub/af-pikans-competitive-baseline-s1-20260928",
    ]
    metadata = {
        "id": kernel_id,
        "title": "AF PIKANs AC LBFGS Continuation S3 Retry 20260928",
        "code_file": entrypoint_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "session": "S3",
        "status": "prepared_within_remaining_s3_stage_cap",
        "kernel_id": kernel_id,
        "kernel_url": "https://www.kaggle.com/code/rodrimandub/af-pikans-ac-lbfgs-continuation-s3-retry-20260928",
        "session_cap_hours": (2700.0 - 25.358649441) / 3600.0,
        "session_cap_seconds": 2674.641350559,
        "s3_stage_total_cap_seconds": 2700.0,
        "previous_s3_attempt_seconds": 25.358649441,
        "per_arm_seed_cap_seconds": 240,
        "accelerator": "private Kaggle T4x2",
        "arms": ARMS,
        "seeds": SEEDS,
        "sources": kernel_sources,
        "source_hashes": {
            "model_source_sha256": config["model_source_sha256"],
            "checkpoint_hashes": checkpoint_hashes,
            "pool_hashes": pool_hashes,
            "reference_hashes": reference_hashes,
            "worker_sha256": sha256(OUT / "ac_lbfgs_worker.py"),
            "campaign_template_sha256": sha256(OUT / "ac_lbfgs_campaign.py"),
            "entrypoint_sha256": sha256(entrypoint_path),
            "payload_zip_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "optimizer": {
            "name": "PyTorch L-BFGS",
            "history_size": 100,
            "tolerance_grad": 1.0e-8,
            "tolerance_change": 0.0,
            "line_search_fn": "strong_wolfe",
            "per_step_max_iter": 1,
            "per_step_max_eval": 8,
        },
        "scoring": "frozen 512x256 common grid against FD8192 and independent ETDRK4 candidates",
        "scientific_accepted": False,
        "reference_accepted": False,
        "s1a_status": s1a_summary["status"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
