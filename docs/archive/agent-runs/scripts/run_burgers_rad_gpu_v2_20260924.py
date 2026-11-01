"""Read-only coordinator for the corrected EXP-0006 GPU kernel."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        key, value = line.split('=', 1)
        if key.strip() in {'KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_PROJECT'}:
            os.environ[key.strip()] = value.strip().strip('"\'')

from src.research_pipeline.remote.bundle import verify_transport
from src.research_pipeline.remote.kaggle import status

BUNDLE = ROOT / 'var/agent_runs/burgers-rad-exp006/bundle-v4-gpu-20260924'

if len(sys.argv) != 2 or sys.argv[1] != 'status':
    raise SystemExit('usage: run_burgers_rad_gpu_v2_20260924.py status')

verify_transport(BUNDLE)
print(json.dumps(status(BUNDLE)))
