"""Official Kaggle CLI adapter. No credential files are read by this module."""
from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

from .bundle import ROOT, generated_path, repository_path, verify_transport
from .runtime import read_json, write_json


def require_approval(spec, approval_path, remote=False):
    if not approval_path:
        raise ValueError('Human approval record required; see remote/README.md')
    approval = read_json(repository_path(approval_path))
    if approval.get('bundle_id') != spec['bundle_id'] or not approval.get('approved_by'):
        raise ValueError('Approval must identify the human reviewer and exact bundle_id')
    for gate in ('scope', 'hypothesis', 'code_surface'):
        if approval.get(gate) != 'approved':
            raise ValueError('Missing human approval: ' + gate)
    if remote and approval.get('remote_execution') is not True:
        raise ValueError('Missing remote execution approval')
    return approval


def require_remote_policy():
    with (ROOT / 'context/agent-policy.toml').open('rb') as handle:
        policy = tomllib.load(handle)
    scope = policy['scope']
    if (scope.get('external_write_access') != 'approved_services' or
            'kaggle' not in scope.get('external_write_services', [])):
        raise ValueError('Blocked by context/agent-policy.toml: external_write_access; '
                         'human-approved permission change required for Kaggle writes')


def executable_path():
    local = ROOT / 'var/agent_runs/remote-env' / ('Scripts/kaggle.exe' if os.name == 'nt' else 'bin/kaggle')
    if local.is_file():
        return local
    executable = shutil.which('kaggle')
    if not executable:
        raise ValueError('Kaggle CLI unavailable; install optional requirements in a repository-local venv')
    return Path(executable)


def invoke(arguments, timeout_sec=90):
    executable = executable_path()
    # Constrain CLI config/cache to the repository. Supply API credentials by environment.
    config_dir = ROOT / 'var/agent_runs/kaggle-config'
    config_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['KAGGLE_CONFIG_DIR'] = str(config_dir)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Windows console launchers may retain the old path after a repo move.
    python=Path(executable).with_name('python.exe')
    command=[str(python),'-m','kaggle'] if python.is_file() else [str(executable)]
    process = subprocess.run([*command, *arguments], env=env, capture_output=True,
                             text=True, timeout=timeout_sec, check=False)
    if process.returncode:
        # Avoid persisting provider messages which could include credentials.
        raise ValueError(f'Kaggle CLI failed (exit {process.returncode}); verify authentication/quota')
    return process.stdout.strip()


def submit(bundle, approval_path):
    require_remote_policy()
    bundle = generated_path(bundle)
    spec = verify_transport(bundle)
    approval = require_approval(spec, approval_path, remote=True)
    state_path = bundle / 'submission.json'
    # Exclusive creation also prevents duplicate requests after uncertain responses.
    try:
        with state_path.open('x', encoding='utf-8') as handle:
            import json
            json.dump({'status': 'submitting', 'bundle_id': spec['bundle_id'],
                       'kernel': spec['kernel'], 'approved_by': approval['approved_by']}, handle)
    except FileExistsError as exc:
        raise ValueError('Submission already recorded; reconcile with status before any resubmission') from exc
    try:
        invoke(['kernels', 'push', '-p', str(bundle)])
    except (ValueError, OSError, subprocess.TimeoutExpired):
        write_json(state_path, {'status': 'submission_unknown', 'bundle_id': spec['bundle_id'],
                                'kernel': spec['kernel']})
        raise
    state = {'status': 'submitted', 'bundle_id': spec['bundle_id'], 'kernel': spec['kernel']}
    write_json(state_path, state)
    return state


def status(bundle):
    spec = verify_transport(bundle)
    return {'kernel': spec['kernel'], 'provider_status': invoke(['kernels', 'status', spec['kernel']])}


def fetch(bundle, output):
    spec = verify_transport(bundle)
    output = generated_path(output)
    if output.exists():
        raise ValueError('Download directory already exists; use a fresh destination')
    output.mkdir(parents=True)
    transport_error = None
    try:
        invoke(['kernels', 'output', spec['kernel'], '-p', str(output)], timeout_sec=1200)
    except ValueError as exc:
        # Kaggle may return a non-zero status after writing the full output tree.
        # Trust only a uniquely matching campaign whose required run artifacts validate.
        transport_error = exc
    candidates = list(output.rglob('campaign.json'))
    matching = [p.parent for p in candidates if read_json(p).get('bundle_id') == spec['bundle_id']]
    if len(matching) != 1:
        if transport_error is not None:
            raise transport_error
        raise ValueError('Downloaded output does not contain exactly one matching campaign; '
                         'it may belong to a different kernel version')
    from .evaluation import evaluate
    report = evaluate(bundle, matching[0])
    if transport_error is not None:
        if not report['execution_valid']:
            raise ValueError('Kaggle CLI returned a non-zero exit and downloaded artifacts failed validation') from transport_error
        report['transport_warning'] = (
            'Kaggle CLI returned a non-zero exit after writing output; the matching campaign and '
            'required artifacts passed local validation.'
        )
    write_json(output / 'evaluation.json', report)
    result = {'runs': str(matching[0]), 'execution_valid': report['execution_valid'],
              'report': str(output / 'evaluation.json')}
    if transport_error is not None:
        result['transport_warning'] = report['transport_warning']
    return result
