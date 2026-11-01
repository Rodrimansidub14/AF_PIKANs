from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python var/agent_runs/kaggle_cli.py <kaggle arguments>")
        return 2

    root = Path(__file__).resolve().parents[2]
    cli = Path(__file__).resolve().parent / "kaggle-cli-env" / "Scripts" / "kaggle.exe"
    values = dotenv_values(root / ".env")

    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        if not os.environ.get(name) and values.get(name):
            os.environ[name] = values[name]

    authenticated = os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    if not authenticated:
        print("Kaggle credentials were not found in the environment or repository .env.")
        return 3
    if not cli.is_file():
        print(f"Kaggle CLI executable is missing: {cli}")
        return 4

    result = subprocess.run([str(cli), *sys.argv[1:]], env=os.environ.copy(), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
