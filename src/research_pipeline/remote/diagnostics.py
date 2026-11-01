"""Credential-free local readiness inspection; never imports provider clients."""
from __future__ import annotations

import json
import os
import subprocess
import tomllib

from .bundle import ROOT


def local_python():
    return ROOT / 'var/agent_runs/remote-env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def doctor():
    python = local_python()
    packages = {}
    error = None
    if python.is_file():
        code = """import importlib.metadata as m, json, sys
result = {'python': sys.version.split()[0], 'packages': {}}
for name in ('kaggle', 'colab-mcp', 'mcp', 'fastmcp', 'PyYAML'):
    try:
        result['packages'][name] = m.version(name)
    except m.PackageNotFoundError:
        result['packages'][name] = None
print(json.dumps(result))
"""
        try:
            result = subprocess.run([str(python), '-I', '-c', code], capture_output=True,
                                    text=True, timeout=15, check=True)
            packages = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            error = 'Local environment cannot be inspected; rerun setup.ps1'
    else:
        error = 'Local environment missing; run setup.ps1'
    config = tomllib.loads((ROOT / '.codex/config.toml').read_text(encoding='utf-8'))
    policy = tomllib.loads((ROOT / 'context/agent-policy.toml').read_text(encoding='utf-8'))
    allowed = (policy['scope'].get('external_write_access') == 'approved_services' and
               'kaggle' in policy['scope'].get('external_write_services', []))
    return {
        'local_dependencies_ready': bool(packages.get('packages')) and all(packages['packages'].values()),
        'python': str(python), 'environment': packages, 'error': error,
        'mcp_enabled': {name: value.get('enabled', True) for name, value in config['mcp_servers'].items()},
        'external_write_access': policy['scope']['external_write_access'],
        'kaggle_permission_approved': allowed,
        'authentication': 'deferred', 'network_checked': False, 'remote_execution_ready': False,
        'remaining': ([] if allowed else ['Explicit external-write permission']) +
                     ['Account authentication', 'Approval of a concrete scientific campaign',
                      'Real provider execution/round-trip verification']}
