"""EXP-0013 — utilidades comunes: registro de ejecución (JSONL), entorno, hashes, carga de módulos externos.

Todo cálculo del estudio escribe en `results/run-log.jsonl` (un evento por línea, con marca de tiempo) para que nada quede sin registrar.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = HERE / "results"
BL = REPO / "var/agent_runs/baseline-competitive-2026-09-27"
EX = REPO / "experiments"
RESULTS.mkdir(exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(event: str, **fields) -> None:
    rec = {"ts": now(), "event": event, **fields}
    with open(RESULTS / "run-log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def env_info() -> dict:
    import numpy
    import scipy
    import torch

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "cpu_threads": torch.get_num_threads(), "git_commit": commit, "working_tree_dirty": True}


_MODS: dict[str, object] = {}


def load_module(name: str, path: Path, extra_path: Path | None = None):
    if name in _MODS:
        return _MODS[name]
    if extra_path is not None and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODS[name] = mod
    return mod


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0
