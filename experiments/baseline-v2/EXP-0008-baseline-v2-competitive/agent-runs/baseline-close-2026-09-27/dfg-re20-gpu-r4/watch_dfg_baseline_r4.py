"""Persistent watcher for the already-submitted DFG Re20 r4 Kaggle run."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[4]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"}:
            os.environ[key] = value.strip().strip("\"'")

os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT / "var/agent_runs/kaggle-config")

from src.research_pipeline.remote.bundle import verify_transport
from src.research_pipeline.remote.evaluation import evaluate
from src.research_pipeline.remote.postprocess import sync_offline_wandb
from src.research_pipeline.remote.runtime import read_json, write_json
import src.research_pipeline.remote.kaggle as kaggle

BUNDLE = ROOT / "var/agent_runs/baseline-close-2026-09-27/dfg-re20-gpu-r4/bundle"
CAMPAIGN = ROOT / "var/agent_runs/baseline-close-2026-09-27/dfg-re20-gpu-r4"
OUT = CAMPAIGN / "download-attempts"
STATE = CAMPAIGN / "watcher-status.json"
PYTHON = ROOT / "var/agent_runs/remote-env/Scripts/python.exe"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(payload: dict) -> None:
    write_json(STATE, {**payload, "updated_at": stamp()})


_original_invoke = kaggle.invoke


def invoke_with_long_output_budget(arguments, timeout_sec=90):
    # Kaggle's output endpoint can take much longer than status/log queries.
    budget = 7200 if arguments[:2] == ["kernels", "output"] else timeout_sec
    return _original_invoke(arguments, timeout_sec=budget)


kaggle.invoke = invoke_with_long_output_budget


def main() -> None:
    spec = verify_transport(BUNDLE)
    base = {"bundle_id": spec["bundle_id"], "kernel": spec["kernel"], "poll_interval_sec": 60}
    save({**base, "status": "watching"})
    print(f"Watching {spec['kernel']} bundle={spec['bundle_id']}", flush=True)
    consecutive_errors = 0
    last_provider = None

    while True:
        try:
            provider = kaggle.status(BUNDLE)["provider_status"]
            consecutive_errors = 0
            save({**base, "status": "provider", "provider_status": provider})
            if provider != last_provider:
                print(f"{stamp()} provider={provider}", flush=True)
                last_provider = provider
        except Exception as exc:
            consecutive_errors += 1
            save({**base, "status": "status_error", "error_type": type(exc).__name__,
                  "consecutive_errors": consecutive_errors})
            print(f"{stamp()} status_error={type(exc).__name__}", flush=True)
            time.sleep(60)
            continue

        upper = provider.upper()
        if any(term in upper for term in ("COMPLETE", "ERROR", "CANCEL", "FAILED", "STOPPED")):
            break
        time.sleep(60)

    save({**base, "status": "terminal", "provider_status": provider})
    print(f"{stamp()} terminal={provider}; fetching existing bundle output", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, 4):
        destination = OUT / f"attempt-{attempt:02d}"
        save({**base, "status": "fetching", "provider_status": provider,
              "fetch_attempt": attempt, "fetch_timeout_sec": 7200})
        print(f"{stamp()} fetch_attempt={attempt}", flush=True)
        try:
            result = kaggle.fetch(BUNDLE, destination)
            runs = pathlib.Path(result["runs"])
            sync = sync_offline_wandb(runs, python_executable=str(PYTHON), timeout_sec=1200)
            write_json(CAMPAIGN / "wandb-sync.json", sync)
            final_status = "fetched" if result["execution_valid"] else "fetched_invalid"
            save({**base, "status": final_status, "provider_status": provider,
                  "fetch_attempt": attempt, **result,
                  "wandb_sync": {"status": sync.get("status"), "run_count": sync.get("run_count"),
                                 "synced_count": sync.get("synced_count")}})
            print(f"{stamp()} {final_status}; execution_valid={result['execution_valid']}; "
                  f"wandb={sync.get('status')}", flush=True)
            return
        except Exception as exc:
            # A CLI timeout may leave a complete output tree; validate before retrying.
            if isinstance(exc, subprocess.TimeoutExpired):
                for campaign_file in destination.rglob("campaign.json") if destination.exists() else []:
                    try:
                        campaign_data = read_json(campaign_file)
                        if campaign_data.get("bundle_id") != spec["bundle_id"]:
                            continue
                        report = evaluate(BUNDLE, campaign_file.parent)
                        if report.get("execution_valid"):
                            write_json(destination / "evaluation.json", report)
                            runs = campaign_file.parent
                            sync = sync_offline_wandb(runs, python_executable=str(PYTHON), timeout_sec=1200)
                            write_json(CAMPAIGN / "wandb-sync.json", sync)
                            save({**base, "status": "fetched", "provider_status": provider,
                                  "fetch_attempt": attempt, "runs": str(runs), "execution_valid": True,
                                  "report": str(destination / "evaluation.json"),
                                  "transport_warning": "CLI timed out after artifacts passed local validation",
                                  "wandb_sync": {"status": sync.get("status"),
                                                 "run_count": sync.get("run_count"),
                                                 "synced_count": sync.get("synced_count")}})
                            print(f"{stamp()} fetched after timeout; artifacts passed validation", flush=True)
                            return
                    except Exception:
                        pass
            save({**base, "status": "fetch_retry", "provider_status": provider,
                  "fetch_attempt": attempt, "error_type": type(exc).__name__})
            print(f"{stamp()} fetch_retry={attempt}; error={type(exc).__name__}", flush=True)
            time.sleep(min(300, 30 * attempt))

    save({**base, "status": "fetch_error", "provider_status": provider,
          "error": "Three download attempts failed; partial artifacts were retained."})
    print(f"{stamp()} fetch_error after three attempts; partial files preserved", flush=True)


if __name__ == "__main__":
    main()
