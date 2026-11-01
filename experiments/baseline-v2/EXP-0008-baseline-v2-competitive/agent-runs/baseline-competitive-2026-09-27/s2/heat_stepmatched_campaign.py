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

PAYLOAD_B64 = "__S2_PAYLOAD_B64__"
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
