"""Launch the installed official Colab MCP with repository-local logs/temp files."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    target = root / 'var/agent_runs/remote-env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if Path(sys.executable).resolve() != target.resolve():
        raise SystemExit('Launch with the repository-local Python from .codex/config.toml; run setup.ps1 first')
    state = root / 'var/agent_runs/colab-state'
    state.mkdir(parents=True, exist_ok=True)
    for name in ['TEMP', 'TMP', 'TMPDIR', 'FASTMCP_HOME']:
        os.environ[name] = str(state)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['FASTMCP_CHECK_FOR_UPDATES'] = 'off'
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(state)
    # Import only after localizing temporary files. No browser/auth tool is called.
    from colab_mcp import main as official_main
    sys.argv = ['colab-mcp', '--log', str(state), *sys.argv[1:]]
    official_main()


if __name__ == '__main__':
    main()
