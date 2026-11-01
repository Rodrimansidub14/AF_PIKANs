"""Single-file Kaggle T4x2 entrypoint; payload marker is filled before upload."""
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

PAYLOAD_B64 = "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"
KERNEL_SLUG = "rodrimandub/af-pikans-heat-step-matched-s2-20260928"
SESSION_START = time.monotonic()
SESSION_STARTED_UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()
SESSION_CAP = 4500.0
SESSION_CLOSE_MARGIN = 60.0
RUN_CAP = 680.0
WAVE_OVERHEAD = 20.0
CONFIRMATION_WAVE_CAP = 120.0
DEVICE_CAP = 9000.0
DEVICE_SPENT = 0.0
POOL_EXPECTED = {
    901: "318cdd951dc76a19ce0be65dd170beaffdf40741d4e11301abda5a8b2fe18f51",
    902: "c89cd6ae60703dcb8cdf04448e2d50f8cdc02324446c1807dcfb3a2145b926d6",
    903: "0e3557a470006da65b348dd066ba73ccad768c6682e3c337c5392c1e8bc6d4e7",
    904: "f3067f1997a1b17947fc3910fd4f70db0e5fded96cb87ef3f87a12f43ea05825",
    905: "dc567fc082828a68ac5483975659f155644d9040cf01916147758e1b5e6736b0",
}
SEEDS = [901, 902, 903, 904, 905]
ARMS = ["rbf", "bspline"]
KERNEL_SOURCES = ["rodrimandub/af-pikans-heat-tier2-torch-pilot-t4x2"]

WORK = pathlib.Path("/kaggle/working/heat-stepmatched-s2")
WORK.mkdir(parents=True, exist_ok=True)
PAYLOAD_DIR = WORK / "payload"
PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64))) as bundle:
    bundle.extractall(PAYLOAD_DIR)
RUNNER = PAYLOAD_DIR / "heat_stepmatched_runner.py"
LEDGER_PATH = WORK / "campaign-ledger.json"
RUNTIME = {}
ledger = {
    "campaign": "heat-stepmatched-s2-2026-09-28",
    "status": "starting",
    "session_cap_seconds": SESSION_CAP,
    "accelerator": "NvidiaTeslaT4 (GPU T4 x2)",
    "scientific_accepted": False,
    "reference_accepted": False,
    "runs": [],
    "pool_files": [],
}


def elapsed():
    return max(0.0, time.monotonic() - SESSION_START)


def session_elapsed():
    return elapsed()


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_ledger(status=None):
    global DEVICE_SPENT
    if status:
        ledger["status"] = status
    ledger["session_wall_seconds"] = elapsed()
    ledger["kaggle_quota_hours_estimate"] = elapsed() / 3600.0
    ledger["aggregate_device_seconds_estimate"] = DEVICE_SPENT
    ledger["aggregate_gpu_hours_estimate"] = DEVICE_SPENT / 3600.0
    write_json_atomic(LEDGER_PATH, ledger)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_t4x2():
    import torch
    import numpy as np

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S2 requires two visible T4 GPUs; no run started.")
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if any("t4" not in name.lower() for name in names):
        raise RuntimeError(f"S2 requires T4x2; visible devices={names}")
    torch.set_num_threads(2)
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gpu_names": names,
        "visible_gpu_count": torch.cuda.device_count(),
    }


RUNTIME = require_t4x2()
V4_POOL_ROOT = pathlib.Path(
    "/kaggle/input/notebooks/rodrimandub/af-pikans-heat-tier2-torch-pilot-t4x2/heat-tier2/pools"
)
POOL_DIR = WORK / "pools"
POOL_DIR.mkdir(parents=True, exist_ok=True)
for seed, expected in POOL_EXPECTED.items():
    candidate = V4_POOL_ROOT / f"heat-seed-{seed}.npz"
    if not candidate.is_file():
        matches = [
            path
            for path in pathlib.Path("/kaggle/input").rglob(f"heat-seed-{seed}.npz")
            if "af-pikans-heat-tier2-torch-pilot-t4x2" in str(path)
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one saved v4 Sobol pool for seed {seed}; found {len(matches)}")
        candidate = matches[0]
    local_pool = POOL_DIR / candidate.name
    shutil.copy2(candidate, local_pool)
    checked = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--verify-pool",
            "--seed",
            str(seed),
            "--pool-file",
            str(local_pool),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pool_info = json.loads(checked.stdout.strip().splitlines()[-1])
    if pool_info["pool_sha256"] != expected:
        raise ValueError(f"v4 pool hash mismatch for seed {seed}")
    ledger["pool_files"].append(
        {
            "seed": seed,
            "source": str(candidate),
            "local_copy": str(local_pool),
            "pool_sha256": pool_info["pool_sha256"],
            "verified_by_runner": True,
        }
    )
save_ledger("pools_verified")
if elapsed() > 300.0:
    save_ledger("setup_cap_partial")
    raise RuntimeError("Pool-copy/setup allowance exceeded.")

RUNS = WORK / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
TRAINING_WAVE_WALL = RUN_CAP + WAVE_OVERHEAD


def run_training_wave(tasks, future_reserve_seconds=0.0):
    global DEVICE_SPENT
    if not 1 <= len(tasks) <= 2:
        raise ValueError("Each wave contains one or two runs.")
    wave_wall_cap = max(task[2] for task in tasks) + WAVE_OVERHEAD
    if session_elapsed() + wave_wall_cap + future_reserve_seconds > SESSION_CAP - SESSION_CLOSE_MARGIN:
        save_ledger("stopped_session_guard_before_wave")
        raise RuntimeError("Insufficient S2 session cap for this wave and reserved confirmations.")
    processes = []
    wave_started = time.monotonic()
    for gpu, (seed, arm, cap) in enumerate(tasks):
        output = RUNS / "tier2" / f"seed-{seed}" / arm
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to reuse run directory: {output}")
        command = [
            sys.executable,
            str(RUNNER),
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--pool-file",
            str(POOL_DIR / f"heat-seed-{seed}.npz"),
            "--out",
            str(output),
            "--gpu-seconds",
            str(cap),
            "--step-cap",
            "20000",
            "--lbfgs-maxiter",
            "15000",
            "--lbfgs-maxfun",
            "18750",
            "--lbfgs-chunk-iter",
            "25",
            "--lbfgs-chunk-max-eval",
            "32",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        stdout_path = output.parent / f"{arm}.stdout.log"
        stderr_path = output.parent / f"{arm}.stderr.log"
        stdout = stdout_path.open("w")
        stderr = stderr_path.open("w")
        process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
        processes.append((seed, arm, cap, output, process, stdout, stderr))
    while any(item[4].poll() is None for item in processes):
        if elapsed() >= SESSION_CAP - SESSION_CLOSE_MARGIN or time.monotonic() - wave_started > wave_wall_cap:
            for item in processes:
                if item[4].poll() is None:
                    item[4].terminate()
            for item in processes:
                try:
                    item[4].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    item[4].kill()
                    item[4].wait(timeout=10)
            save_ledger("stopped_wave_guard_partial")
            raise RuntimeError("Training wave stopped at its wall/session guard.")
        time.sleep(1.0)
    for seed, arm, cap, output, process, stdout, stderr in processes:
        return_code = process.wait()
        stdout.close()
        stderr.close()
        for name, source in (("stdout.log", output.parent / f"{arm}.stdout.log"), ("stderr.log", output.parent / f"{arm}.stderr.log")):
            if source.is_file() and output.is_dir():
                source.replace(output / name)
        cost_path = output / "cost.json"
        if not cost_path.is_file():
            ledger["runs"].append(
                {"kind": "tier2", "seed": seed, "arm": arm, "return_code": return_code, "status": "cost_unknown"}
            )
            save_ledger("stopped_cost_unknown")
            raise RuntimeError(f"Missing partial cost for {arm} seed {seed}; no replacement run.")
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        actual = float(cost.get("allocated_gpu_seconds_estimate", 0.0))
        DEVICE_SPENT += actual
        ledger["runs"].append(
            {
                "kind": "tier2",
                "seed": seed,
                "arm": arm,
                "return_code": return_code,
                "run_status": cost.get("run_status"),
                "adam_updates": cost.get("adam_updates"),
                "adam_target_reached": cost.get("adam_target_reached"),
                "lbfgs_updates": cost.get("lbfgs_updates"),
                "lbfgs_stop_reason": cost.get("lbfgs_stop_reason"),
                "training_device_seconds": cost.get("training_gpu_seconds_estimate"),
                "total_device_seconds": actual,
                "pool_sha256": cost.get("pool_sha256"),
                "cost_path": str(cost_path),
                "checkpoint": str(output / "checkpoint-final.pt") if (output / "checkpoint-final.pt").is_file() else None,
            }
        )
        if DEVICE_SPENT > DEVICE_CAP:
            save_ledger("device_cap_exceeded")
            raise RuntimeError("Aggregate device-time cap exceeded.")
        save_ledger()
        if return_code:
            print(f"Recorded partial run; no replacement: {arm} seed {seed}, rc={return_code}")


for index, seed in enumerate(SEEDS):
    remaining_waves = len(SEEDS) - index - 1
    reserve = remaining_waves * TRAINING_WAVE_WALL + 5 * (CONFIRMATION_WAVE_CAP + WAVE_OVERHEAD) + 40.0
    run_training_wave([(seed, "rbf", RUN_CAP), (seed, "bspline", RUN_CAP)], future_reserve_seconds=reserve)
save_ledger("training_stage_finished")

FINAL = WORK / "confirmation"
FINAL.mkdir(parents=True, exist_ok=True)


def score_wave(tasks):
    global DEVICE_SPENT
    wall_cap = CONFIRMATION_WAVE_CAP + WAVE_OVERHEAD
    if elapsed() + wall_cap > SESSION_CAP - SESSION_CLOSE_MARGIN:
        save_ledger("incomplete_confirmation_budget")
        return False
    processes = []
    wave_started = time.monotonic()
    for gpu, (seed, arm) in enumerate(tasks):
        checkpoint = RUNS / "tier2" / f"seed-{seed}" / arm / "checkpoint-final.pt"
        if not checkpoint.is_file():
            ledger["runs"].append({"kind": "confirmation", "seed": seed, "arm": arm, "status": "missing_final_checkpoint"})
            save_ledger()
            continue
        output = FINAL / f"seed-{seed}" / arm
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(PAYLOAD_DIR / "heat_final_eval.py"),
            "--checkpoint",
            str(checkpoint),
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--out",
            str(output),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        stdout = (output / "stdout.log").open("w")
        stderr = (output / "stderr.log").open("w")
        process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
        processes.append((seed, arm, output, process, stdout, stderr))
    while any(item[3].poll() is None for item in processes):
        if elapsed() >= SESSION_CAP - SESSION_CLOSE_MARGIN or time.monotonic() - wave_started > wall_cap:
            for item in processes:
                if item[3].poll() is None:
                    item[3].terminate()
            for item in processes:
                try:
                    item[3].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    item[3].kill()
                    item[3].wait(timeout=10)
            save_ledger("stopped_confirmation_guard_partial")
            return False
        time.sleep(1.0)
    for seed, arm, output, process, stdout, stderr in processes:
        return_code = process.wait()
        stdout.close()
        stderr.close()
        metrics_path = output / "confirmation-metrics.json"
        cost_path = output / "confirmation-cost.json"
        if not cost_path.is_file():
            ledger["runs"].append(
                {"kind": "confirmation", "seed": seed, "arm": arm, "return_code": return_code, "status": "missing_confirmation_cost"}
            )
            save_ledger()
            continue
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        actual = float(cost.get("evaluation_gpu_seconds_estimate", 0.0))
        DEVICE_SPENT += actual
        ledger["runs"].append(
            {
                "kind": "confirmation",
                "seed": seed,
                "arm": arm,
                "return_code": return_code,
                "run_status": "scored" if return_code == 0 and metrics_path.is_file() else "scorer_failed",
                "evaluation_device_seconds": actual,
                "grid_id": cost.get("grid_id"),
                "metrics_path": str(metrics_path),
                "cost_path": str(cost_path),
            }
        )
        save_ledger()
    return True


confirmations = [(seed, arm) for seed in SEEDS for arm in ARMS]
confirm_waves = (len(confirmations) + 1) // 2
confirmation_budget = confirm_waves * (CONFIRMATION_WAVE_CAP + WAVE_OVERHEAD) + 40.0
if elapsed() + confirmation_budget > SESSION_CAP - SESSION_CLOSE_MARGIN:
    save_ledger("incomplete_confirmation_budget")
else:
    for offset in range(0, len(confirmations), 2):
        if not score_wave(confirmations[offset : offset + 2]):
            break

training = [item for item in ledger["runs"] if item.get("kind") == "tier2"]
scored = [item for item in ledger["runs"] if item.get("kind") == "confirmation" and item.get("run_status") == "scored"]
accepted_stop_reasons = {
    "optimizer_stopped_early",
    "max_iterations",
    "max_function_evaluations",
    "lbfgs_iteration_cap",
}
all_training_complete = len(training) == 10 and all(
    item.get("adam_updates") == 20_000 and item.get("lbfgs_stop_reason") in accepted_stop_reasons for item in training
)
campaign_status = "complete" if all_training_complete and len(scored) == 10 else "partial"
ledger["completion"] = {
    "training_runs": len(training),
    "runs_reaching_adam_20000": sum(item.get("adam_updates") == 20_000 for item in training),
    "confirmation_scores": len(scored),
    "scientific_accepted": False,
    "reference_accepted": False,
}
save_ledger(campaign_status)

artifacts = []
for path in sorted(WORK.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        artifacts.append(
            {
                "path": str(path.relative_to(WORK)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
manifest = {
    "schema_version": 1,
    "session": "S2",
    "kernel_slug": KERNEL_SLUG,
    "status": campaign_status,
    "started_at_utc": SESSION_STARTED_UTC,
    "ended_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "actual_session_seconds": elapsed(),
    "push_cap_seconds": SESSION_CAP,
    "session_quota_cap_hours": 1.25,
    "per_run_training_cap_seconds": RUN_CAP,
    "runtime": RUNTIME,
    "pool_hashes": POOL_EXPECTED,
    "optimizer": {
        "adam_lr": 1.0e-3,
        "adam_updates": 20_000,
        "lbfgs_maxiter": 15_000,
        "lbfgs_maxfun": 18_750,
        "history_size": 100,
        "tolerance_grad": 1.0e-8,
        "tolerance_change": 0.0,
        "line_search": "strong_wolfe",
        "chunk_iter": 25,
        "chunk_max_eval": 32,
    },
    "arms": ARMS,
    "seeds": SEEDS,
    "score_grid": "heat-confirmatory-512x256-v1",
    "kernel_sources": KERNEL_SOURCES,
    "scientific_accepted": False,
    "reference_accepted": False,
    "artifacts": artifacts,
}
(WORK / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "status": campaign_status,
            "session_seconds": elapsed(),
            "quota_hours": elapsed() / 3600.0,
            "device_seconds": DEVICE_SPENT,
            "training_runs": len(training),
            "confirmed_runs": len(scored),
            "scientific_accepted": False,
        },
        indent=2,
    )
)
