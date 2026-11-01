from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[4]
KAGGLE_EXE = Path(__file__).resolve().parent / 'kaggle-venv' / 'Scripts' / 'kaggle.exe'
REQUIRED = ('KAGGLE_API_TOKEN', 'KAGGLE_USERNAME')


def main() -> int:
    values = dotenv_values(ROOT / '.env')
    secrets = {key: values.get(key) for key in REQUIRED}
    if any(not isinstance(value, str) or not value for value in secrets.values()):
        print('Required Kaggle credentials are missing from the repository .env file.', file=sys.stderr)
        return 2
    child_env = os.environ.copy()
    child_env.update(secrets)
    completed = subprocess.run(
        [str(KAGGLE_EXE), *sys.argv[1:]],
        cwd=ROOT,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors='replace',
        check=False,
    )
    for stream, destination in ((completed.stdout, sys.stdout), (completed.stderr, sys.stderr)):
        for secret in secrets.values():
            stream = stream.replace(secret, '[REDACTED]')
        destination.write(stream)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
