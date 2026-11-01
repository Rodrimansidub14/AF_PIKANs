"""Single-file Kaggle T4x2 launcher; worker and hash manifest are embedded."""
import base64
import datetime
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile

PAYLOAD_B64 = "__S3_PAYLOAD_B64__"
KERNEL_SLUG = "rodrimandub/af-pikans-ac-lbfgs-continuation-s3-retry-20260928"
SESSION_START = time.monotonic()
SESSION_STARTED_UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()
SESSION_CAP_SECONDS = 2674.641350559
S3_STAGE_TOTAL_CAP_SECONDS = 2700.0
PREVIOUS_S3_ATTEMPT_SECONDS = 25.358649441
SESSION_CLOSE_MARGIN_SECONDS = 60.0
RUN_CAP_SECONDS = 240.0
TRAIN_WAVE_WALL_SECONDS = 270.0
SCORE_WAVE_WALL_SECONDS = 240.0
SEEDS = [401, 402, 403, 404, 405]
ARMS = ["AC-RBF-Sobol", "AC-DeepXDE-MLP"]
KERNEL_SOURCES = [
    "rodrimandub/af-pikans-allen-cahn-t4x2-20260927",
    "rodrimandub/af-pikans-competitive-baseline-s1-20260928",
]

WORK = pathlib.Path("/kaggle/working/closure-s3-20260928")
WORK.mkdir(parents=True, exist_ok=True)
PAYLOAD_DIR = WORK / "payload"
PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64))) as bundle:
    bundle.extractall(PAYLOAD_DIR)
RUNNER = PAYLOAD_DIR / "ac_lbfgs_worker.py"
CONFIG_PATH = PAYLOAD_DIR / "s3-source-config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
AC_ROOT = pathlib.Path(CONFIG["input_roots"]["allen_cahn"])
S1_ROOT = pathlib.Path(CONFIG["input_roots"]["s1"])
LEDGER_PATH = WORK / "campaign-ledger.json"
LOGS = WORK / "worker-logs"
LOGS.mkdir(parents=True, exist_ok=True)
LEDGER = {
    "campaign": "allen-cahn-lbfgs-continuation-s3-retry-2026-09-28",
    "status": "starting",
    "kernel_slug": KERNEL_SLUG,
    "session_cap_seconds": SESSION_CAP_SECONDS,
    "per_arm_seed_cap_seconds": RUN_CAP_SECONDS,
    "accelerator": "NvidiaTeslaT4 (GPU T4 x2)",
    "seeds": SEEDS,
    "arms": ARMS,
    "scientific_accepted": False,
    "reference_accepted": False,
    "source_preflight": {},
    "runs": [],
}


def elapsed():
    return max(0.0, time.monotonic() - SESSION_START)


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_ledger(status=None):
    if status:
        LEDGER["status"] = status
    LEDGER["session_wall_seconds"] = elapsed()
    write_json_atomic(LEDGER_PATH, LEDGER)
    manifest = {
        "schema_version": 1,
        "session": "S3",
        "kernel_slug": KERNEL_SLUG,
        "status": LEDGER["status"],
        "started_at_utc": SESSION_STARTED_UTC,
        "session_wall_seconds": elapsed(),
        "session_cap_seconds": SESSION_CAP_SECONDS,
        "session_quota_cap_hours": SESSION_CAP_SECONDS / 3600.0,
        "s3_stage_total_cap_seconds": S3_STAGE_TOTAL_CAP_SECONDS,
        "previous_s3_attempt_seconds": PREVIOUS_S3_ATTEMPT_SECONDS,
        "per_arm_seed_cap_seconds": RUN_CAP_SECONDS,
        "runtime": LEDGER.get("runtime"),
        "source_preflight": LEDGER.get("source_preflight"),
        "runs": LEDGER["runs"],
        "scientific_accepted": False,
        "reference_accepted": False,
    }
    write_json_atomic(WORK / "manifest.json", manifest)


def require_t4x2():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S3 requires exactly two visible T4 devices; no fallback is allowed.")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("t4" not in name.lower() for name in names):
        raise RuntimeError(f"S3 requires T4x2; visible devices={names}")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu_names": names, "visible_gpu_count": torch.cuda.device_count()}


def source_path(kind, arm=None, seed=None):
    if kind == "checkpoint":
        return AC_ROOT / "campaign" / arm / f"seed-{seed}" / "checkpoint-final.pt"
    if kind == "pool":
        return AC_ROOT / "campaign" / "pools" / f"seed-{seed}.npz"
    if kind == "FD8192":
        return AC_ROOT / CONFIG["reference_paths"][kind]
    if kind == "ETDRK4":
        return S1_ROOT / CONFIG["reference_paths"][kind]
    if kind == "model_source":
        return AC_ROOT / "run_campaign.py"
    raise ValueError(f"Unknown preflight artifact {kind}")


def verify_sources():
    observed = {"model_source": sha256(source_path("model_source")), "checkpoints": {}, "pools": {}, "references": {}}
    if observed["model_source"] != CONFIG["model_source_sha256"]:
        raise ValueError("Allen-Cahn model source hash mismatch")
    for seed in SEEDS:
        pool_file = source_path("pool", seed=seed)
        actual_pool = sha256(pool_file)
        if actual_pool != CONFIG["pool_hashes"][str(seed)]:
            raise ValueError(f"saved pool hash mismatch for seed {seed}")
        observed["pools"][str(seed)] = actual_pool
        observed["checkpoints"][str(seed)] = {}
        for arm in ARMS:
            checkpoint_file = source_path("checkpoint", arm=arm, seed=seed)
            actual_checkpoint = sha256(checkpoint_file)
            expected = CONFIG["checkpoint_hashes"][arm][str(seed)]
            if actual_checkpoint != expected:
                raise ValueError(f"checkpoint hash mismatch for {arm} seed {seed}")
            observed["checkpoints"][str(seed)][arm] = actual_checkpoint
    for ref in ("FD8192", "ETDRK4"):
        actual = sha256(source_path(ref))
        if actual != CONFIG["reference_hashes"][ref]:
            raise ValueError(f"{ref} reference hash mismatch")
        observed["references"][ref] = actual
    observed["s1a_status"] = CONFIG["s1a_status"]
    observed["all_hashes_verified"] = True
    return observed


def launch_pair(mode, seed):
    if mode not in {"train", "score"}:
        raise ValueError(mode)
    cap_wall = TRAIN_WAVE_WALL_SECONDS if mode == "train" else SCORE_WAVE_WALL_SECONDS
    if elapsed() + cap_wall + SESSION_CLOSE_MARGIN_SECONDS > SESSION_CAP_SECONDS:
        save_ledger(f"stopped_session_guard_before_{mode}_seed_{seed}")
        return False
    processes = []
    wave_start = time.monotonic()
    for device_index, arm in enumerate(ARMS):
        run_dir = WORK / "runs" / arm / f"seed-{seed}"
        if mode == "train":
            if run_dir.exists():
                raise FileExistsError(f"Refusing to reuse S3 run directory: {run_dir}")
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            out_path = run_dir
            args = [sys.executable, str(RUNNER), "--mode", "train", "--arm", arm,
                    "--seed", str(seed), "--out", str(out_path), "--cap", str(RUN_CAP_SECONDS)]
        else:
            score_path = run_dir / "score.json"
            if not (run_dir / "checkpoint-final.pt").is_file():
                LEDGER["runs"].append({"mode": "score", "arm": arm, "seed": seed,
                                       "status": "missing_continuation_checkpoint"})
                continue
            args = [sys.executable, str(RUNNER), "--mode", "score", "--arm", arm,
                    "--seed", str(seed), "--run-dir", str(run_dir), "--out", str(score_path)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
        log_path = LOGS / f"{mode}-seed-{seed}-{arm}.log"
        stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(args, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((arm, seed, run_dir, process, stream, log_path))

    while any(item[3].poll() is None for item in processes):
        if elapsed() >= SESSION_CAP_SECONDS - SESSION_CLOSE_MARGIN_SECONDS or time.monotonic() - wave_start >= cap_wall:
            for item in processes:
                if item[3].poll() is None:
                    item[3].terminate()
            for item in processes:
                try:
                    item[3].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    item[3].kill()
                    item[3].wait(timeout=10)
            save_ledger(f"stopped_{mode}_wave_guard_partial")
            break
        time.sleep(1.0)

    for arm, seed, run_dir, process, stream, log_path in processes:
        return_code = process.poll()
        if return_code is None:
            return_code = process.wait()
        stream.close()
        row = {"mode": mode, "arm": arm, "seed": seed, "return_code": return_code,
               "log": str(log_path.relative_to(WORK))}
        if mode == "train":
            cost_path = run_dir / "cost.json"
            progress_path = run_dir / "progress.json"
            if cost_path.is_file():
                cost = json.loads(cost_path.read_text(encoding="utf-8"))
                row.update({"status": cost.get("status"), "lbfgs_updates": cost.get("lbfgs_updates"),
                            "closure_evaluations": cost.get("closure_evaluations"),
                            "training_gpu_seconds": cost.get("training_gpu_seconds_estimate"),
                            "continuation_checkpoint": str((run_dir / "checkpoint-final.pt").relative_to(WORK))
                            if (run_dir / "checkpoint-final.pt").is_file() else None})
            elif progress_path.is_file():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                row.update({"status": progress.get("status"), "lbfgs_updates": progress.get("lbfgs_updates"),
                            "partial_progress": str(progress_path.relative_to(WORK))})
            else:
                row["status"] = "worker_failed_before_progress"
        else:
            score_path = run_dir / "score.json"
            if score_path.is_file():
                score = json.loads(score_path.read_text(encoding="utf-8"))
                row.update({"status": score.get("status"), "scoring_wall_seconds": score.get("scoring_wall_seconds"),
                            "score": str(score_path.relative_to(WORK))})
            else:
                row["status"] = "scorer_failed"
        LEDGER["runs"].append(row)
        save_ledger()
    return True


def main():
    save_ledger("starting_preflight")
    try:
        LEDGER["runtime"] = require_t4x2()
        LEDGER["source_preflight"] = verify_sources()
        save_ledger("sources_verified")
    except Exception as exc:
        LEDGER["preflight_error"] = f"{type(exc).__name__}: {exc}"
        save_ledger("preflight_failed")
        raise

    for seed in SEEDS:
        if not launch_pair("train", seed):
            break
    trained = [row for row in LEDGER["runs"] if row.get("mode") == "train"]
    save_ledger("training_stage_finished" if len(trained) == len(SEEDS) * len(ARMS) else "training_partial")

    if elapsed() < SESSION_CAP_SECONDS - SESSION_CLOSE_MARGIN_SECONDS:
        for seed in SEEDS:
            if not launch_pair("score", seed):
                break

    scored = [row for row in LEDGER["runs"] if row.get("mode") == "score" and row.get("status") == "scored"]
    training_complete = len(trained) == 10 and all(row.get("status") == "time_cap_reached" for row in trained)
    status = "complete" if training_complete and len(scored) == 10 else "partial"
    LEDGER["completion"] = {
        "training_runs": len(trained),
        "training_reaching_time_cap": sum(row.get("status") == "time_cap_reached" for row in trained),
        "scored_runs": len(scored),
        "scientific_accepted": False,
        "reference_accepted": False,
    }
    LEDGER["ended_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_ledger(status)
    artifacts = []
    for path in sorted(WORK.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({"path": str(path.relative_to(WORK)), "size_bytes": path.stat().st_size,
                              "sha256": sha256(path)})
    final_manifest = {
        "schema_version": 1,
        "session": "S3",
        "kernel_slug": KERNEL_SLUG,
        "status": status,
        "started_at_utc": SESSION_STARTED_UTC,
        "ended_at_utc": LEDGER["ended_at_utc"],
        "actual_session_seconds": elapsed(),
        "push_cap_seconds": SESSION_CAP_SECONDS,
        "session_quota_cap_hours": SESSION_CAP_SECONDS / 3600.0,
        "s3_stage_total_cap_seconds": S3_STAGE_TOTAL_CAP_SECONDS,
        "previous_s3_attempt_seconds": PREVIOUS_S3_ATTEMPT_SECONDS,
        "per_arm_seed_cap_seconds": RUN_CAP_SECONDS,
        "runtime": LEDGER.get("runtime"),
        "source_preflight": LEDGER.get("source_preflight"),
        "arms": ARMS,
        "seeds": SEEDS,
        "optimizer": {"name": "PyTorch LBFGS", "history_size": 100,
                      "tolerance_grad": 1.0e-8, "tolerance_change": 0.0,
                      "line_search_fn": "strong_wolfe", "per_step_max_iter": 1,
                      "per_step_max_eval": 8},
        "references": ["FD8192 candidate", "independent ETDRK4 candidate"],
        "score_grid": "512x256",
        "kernel_sources": KERNEL_SOURCES,
        "artifacts": artifacts,
        "scientific_accepted": False,
        "reference_accepted": False,
    }
    write_json_atomic(WORK / "manifest.json", final_manifest)
    print(json.dumps({"status": status, "session_seconds": elapsed(),
                      "training_runs": len(trained), "scored_runs": len(scored),
                      "scientific_accepted": False}, indent=2))


main()
