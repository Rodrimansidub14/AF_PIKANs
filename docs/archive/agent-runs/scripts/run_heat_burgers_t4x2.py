"""Prepare, submit, inspect and fetch the approved Heat+Burgers T4x2 campaign."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def load_local_credentials():
    allowed = {'KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_PROJECT'}
    env_file = ROOT / '.env'
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding='utf-8-sig').splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        key, value = line.split('=', 1)
        key, value = key.strip(), value.strip().strip('"\'')
        if key in allowed and value:
            os.environ.setdefault(key, value)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=('prepare', 'submit', 'status', 'fetch', 'check'))
args = parser.parse_args()
if args.action in ('submit', 'status', 'fetch', 'check'):
    load_local_credentials()

from src.research_pipeline.remote.bundle import prepare, verify_transport
from src.research_pipeline.remote.kaggle import fetch, invoke, status, submit
from src.research_pipeline.remote.postprocess import (
    extract_results_archive, sync_offline_wandb, verify_results_hashes,
)
from src.research_pipeline.remote.runtime import write_json

BASE = ROOT / 'var' / 'agent_runs' / 'heat-burgers-002-t4x2'
BUNDLE = BASE / 'bundle-final-20260923'
APPROVAL = BASE / 'approval.json'
MANIFEST = ROOT / 'experiments' / 'EXP-0002-heat-burgers-sobol' / 'manifest-v3-t4x2.json'
NOTEBOOK = ROOT / 'notebooks' / 'experiments' / 'heat-burgers-baseline-t4x2.ipynb'

if args.action == 'check':
    reply = invoke(['kernels', 'list', '--mine', '--page-size', '1', '--csv'])
    print(json.dumps({'authenticated_listing': 'ref' in reply or 'No kernels found' in reply}))
elif args.action == 'prepare':
    owner = os.environ.get('KAGGLE_USERNAME') or 'rodrigomansilla'
    kernel = owner + '/af-pikans-heat-burgers-baseline-t4x2-002'
    source_files = [
        'src/research_pipeline/sciml/__init__.py',
        'src/research_pipeline/sciml/model.py',
        'src/research_pipeline/sciml/sampling.py',
        'src/research_pipeline/sciml/train.py',
        'src/research_pipeline/sciml/campaign.py',
        'src/research_pipeline/remote/__init__.py',
        'src/research_pipeline/remote/reporting.py',
        'evals/__init__.py',
        'evals/checkpoint_validation.py',
        'evals/problems/__init__.py',
        'evals/problems/_grids.py',
        'evals/problems/heat_1d/__init__.py',
        'evals/problems/heat_1d/reference.py',
        'evals/problems/burgers_1d/__init__.py',
        'evals/problems/burgers_1d/reference.py',
    ]
    spec = prepare(MANIFEST, BUNDLE, kernel, source_files,
                   timeout=2100, evidence_kind='research', gpu=True)
    BASE.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLE / 'experiment.ipynb', NOTEBOOK)
    write_json(BASE / 'manifest.json', json.loads(MANIFEST.read_text(encoding='utf-8')))
    (BASE / 'parameter-decisions.md').write_text(
        json.loads(MANIFEST.read_text(encoding='utf-8'))['parameter_decisions_markdown'],
        encoding='utf-8')
    approval = {
        'bundle_id': spec['bundle_id'],
        'approved_by': 'Rodrigo Mansilla',
        'scope': 'approved',
        'hypothesis': 'approved',
        'code_surface': 'approved',
        'remote_execution': True,
        'authorization_source': 'Conversation: authorized Heat and Burgers exploratory rerun, T4x2 in parallel, PINA/PyTorch RBF-PIKAN, W&B, quantitative tables and Plotly outputs.',
        'interpretation': 'Exploratory evidence only. Scientific settings retained; instrumentation and execution optimized. No SOTA or claim acceptance.',
    }
    write_json(APPROVAL, approval)
    verify_transport(BUNDLE)
    print(json.dumps({'bundle_id': spec['bundle_id'], 'kernel': kernel,
                      'machine_shape': spec.get('machine_shape'),
                      'max_parallel_seeds': spec['max_parallel_seeds'],
                      'notebook': str(NOTEBOOK.relative_to(ROOT)),
                      'source_file_count': len(spec['files'])}, indent=2))
elif args.action == 'submit':
    print(json.dumps(submit(BUNDLE, APPROVAL), indent=2))
elif args.action == 'status':
    print(json.dumps(status(BUNDLE), indent=2))
elif args.action == 'fetch':
    result = fetch(BUNDLE, BASE / 'download-retry')
    archives = list((BASE / 'download-retry').rglob('heat-burgers-t4x2-results.zip'))
    if len(archives) != 1:
        raise ValueError('Expected exactly one reproducible results archive in Kaggle output')
    extracted = extract_results_archive(archives[0], BASE / 'download-extracted')
    result['result_file_verification'] = verify_results_hashes(extracted)
    candidates = list(extracted.rglob('heat-burgers-results.ipynb'))
    if len(candidates) != 1:
        raise ValueError('Expected exactly one generated results notebook in the verified archive')
    destination = NOTEBOOK.parent / 'heat-burgers-baseline-t4x2-results.ipynb'
    shutil.copy2(candidates[0], destination)
    result['results_notebook'] = str(destination.relative_to(ROOT))
    result['extracted_results'] = str(extracted.relative_to(ROOT))
    sync_result = sync_offline_wandb(extracted / 'runs', os.environ)
    write_json(BASE / 'wandb-sync.json', sync_result)
    result['wandb_sync'] = sync_result
    print(json.dumps(result, indent=2))
