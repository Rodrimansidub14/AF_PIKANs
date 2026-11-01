"""Local post-processing for reproducible remote research outputs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
import zipfile


def create_results_archive(work_root, analysis_dir, archive_path):
    """Hash and archive the run outputs, including W&B offline data, without self-inclusion."""
    root = Path(work_root).resolve()
    analysis = Path(analysis_dir).resolve()
    archive_path = Path(archive_path).resolve()
    analysis.relative_to(root)
    archive_path.relative_to(root)
    analysis.mkdir(parents=True, exist_ok=True)
    hash_path = analysis / "artifact-hashes.json"
    excluded_names = {".env", "kaggle.json", "access_token"}
    excluded_dirs = {".git", ".kaggle"}
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() in {archive_path, hash_path.resolve()}:
            continue
        relative = path.relative_to(root)
        if path.name in excluded_names or any(part in excluded_dirs for part in relative.parts):
            continue
        files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    hash_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files.append(hash_path)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return {"archive": archive_path, "file_count": len(hashes)}


def extract_results_archive(archive_path, destination):
    """Extract a results ZIP only after validating every member stays in the target."""
    archive_path = Path(archive_path).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Results extraction destination already exists")
    members = []
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = info.filename
            if not name or "\\" in name or "\x00" in name:
                raise ValueError("Unsafe path in results archive")
            posix = PurePosixPath(name)
            windows = PureWindowsPath(name)
            parts = posix.parts
            if (posix.is_absolute() or windows.is_absolute() or windows.drive
                    or ".." in parts or not parts):
                raise ValueError("Unsafe path in results archive")
            relative = Path(*parts)
            if relative.as_posix() in seen:
                raise ValueError("Duplicate path in results archive")
            seen.add(relative.as_posix())
            target = (destination / relative).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ValueError("Unsafe path in results archive") from exc
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError("Non-regular file in results archive")
            is_directory = info.is_dir() or kind == stat.S_IFDIR
            members.append((info, target, is_directory))

    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info, target, is_directory in members:
            if is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return destination


def verify_results_hashes(results_root):
    """Validate every file listed by the remote result's SHA-256 manifest."""
    root = Path(results_root).resolve()
    index_path = root / "analysis/artifact-hashes.json"
    try:
        hashes = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Results artifact hash manifest is missing or invalid") from exc
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Results artifact hash manifest is empty or invalid")
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe path in results hash manifest")
        path = (root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Unsafe path in results hash manifest") from exc
        if not path.is_file():
            raise ValueError("Hashed result file is missing: " + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Result hash mismatch: " + name)
    return {"verified_file_count": len(hashes)}


def sync_offline_wandb(runs_root, environment=None, python_executable=None, timeout_sec=600):
    """Sync archived offline runs using a W&B key held only in the child process environment."""
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise ValueError("W&B runs directory does not exist")
    source_env = os.environ if environment is None else environment
    run_dirs = sorted(
        path for path in root.rglob("offline-run-*")
        if path.is_dir() and any(path.glob("*.wandb"))
    )
    if not source_env.get("WANDB_API_KEY"):
        return {"status": "deferred", "reason": "missing_api_key", "run_count": len(run_dirs)}
    if not run_dirs:
        return {"status": "no_runs", "run_count": 0}

    inherited_names = (
        "PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "USERPROFILE",
        "APPDATA", "LOCALAPPDATA", "PYTHONHOME", "PYTHONPATH",
    )
    child_env = {name: os.environ[name] for name in inherited_names if name in os.environ}
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_BASE_URL"):
        if source_env.get(name):
            child_env[name] = source_env[name]
    child_env["WANDB_MODE"] = "online"
    python_executable = python_executable or sys.executable
    results = []
    for run_dir in run_dirs:
        seed_root = run_dir.parents[2] if len(run_dir.parents) > 2 else run_dir.parent
        run_env = dict(child_env)
        for name, directory in (
            ("WANDB_DIR", "wandb"), ("WANDB_CACHE_DIR", "wandb-cache"),
            ("WANDB_CONFIG_DIR", "wandb-config"), ("WANDB_DATA_DIR", "wandb-data"),
        ):
            local_path = seed_root / directory
            if local_path.is_dir():
                run_env[name] = str(local_path)
        command = [python_executable, "-m", "wandb", "sync", "--yes"]
        if source_env.get("WANDB_ENTITY"):
            command.extend(("--entity", source_env["WANDB_ENTITY"]))
        if source_env.get("WANDB_PROJECT"):
            command.extend(("--project", source_env["WANDB_PROJECT"]))
        command.append(str(run_dir))
        try:
            completed = subprocess.run(
                command,
                env=run_env, capture_output=True, text=True, timeout=timeout_sec, check=False,
            )
            outcome = {"run": run_dir.name,
                       "status": "synced" if completed.returncode == 0 else "failed",
                       "exit_code": int(completed.returncode)}
        except subprocess.TimeoutExpired:
            outcome = {"run": run_dir.name, "status": "timed_out"}
        except OSError:
            outcome = {"run": run_dir.name, "status": "unavailable"}
        results.append(outcome)
    synced = sum(item["status"] == "synced" for item in results)
    status = "synced" if synced == len(results) else "partial" if synced else "failed"
    return {"status": status, "run_count": len(results), "synced_count": synced, "runs": results}
