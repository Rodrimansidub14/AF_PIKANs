"""Quietly finish the remote EXP-0006 run and materialize its artifacts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        key, value = line.split('=', 1)
        if key.strip() in {'KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_PROJECT'}:
            os.environ[key.strip()] = value.strip().strip('"\'')

from src.research_pipeline.remote.bundle import verify_transport
from src.research_pipeline.remote.kaggle import fetch, status
from src.research_pipeline.remote.postprocess import sync_offline_wandb

BUNDLE = ROOT / 'var/agent_runs/burgers-rad-exp006/bundle-v5-gpu-20260924'
OUT = ROOT / 'var/agent_runs/burgers-rad-exp006-v3/download-gpu-20260924'
STATE = ROOT / 'var/agent_runs/burgers-rad-exp006-v3/watcher-status.json'
PYTHON = ROOT / 'var/agent_runs/remote-env/Scripts/python.exe'


def write_state(payload: dict) -> None:
    STATE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def main() -> None:
    spec = verify_transport(BUNDLE)
    write_state({'status': 'watching', 'bundle_id': spec['bundle_id'], 'kernel': spec['kernel']})
    while True:
        try:
            provider = status(BUNDLE)['provider_status']
        except Exception as exc:  # preserve the watcher for transient API failures
            write_state({'status': 'status_error', 'error': type(exc).__name__ + ': ' + str(exc)})
            time.sleep(60)
            continue
        write_state({'status': 'provider', 'provider_status': provider,
                     'bundle_id': spec['bundle_id'], 'kernel': spec['kernel']})
        terminal = any(token in provider for token in ('COMPLETE', 'ERROR', 'CANCEL', 'FAILED'))
        if terminal:
            try:
                result = fetch(BUNDLE, OUT)
                payload = {'status': 'fetched', 'provider_status': provider, **result}
                runs = Path(result['runs'])
                sync = sync_offline_wandb(runs, python_executable=str(PYTHON), timeout_sec=1200)
                payload['wandb_sync'] = sync
                write_state(payload)
            except Exception as exc:
                write_state({'status': 'fetch_error', 'provider_status': provider,
                             'error': type(exc).__name__ + ': ' + str(exc)})
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
