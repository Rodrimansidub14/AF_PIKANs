from __future__ import annotations

import base64
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / "run_campaign.py"
WORKER = HERE / "dfg_lbfgs_worker.py"
NOTEBOOK = HERE / "dfg-re20-lbfgs-refinement.ipynb"

CELL = r'''import base64, csv, hashlib, importlib.util, json, math, os, signal, subprocess, sys, threading, time, zipfile
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor

SESSION_STARTED = time.perf_counter()
QUOTA_CAP_SECONDS = 7200.0
SOFT_DEADLINE_SECONDS = 6500.0
WATCHDOG_SECONDS = 6900.0
DEADLINE = SESSION_STARTED + SOFT_DEADLINE_SECONDS
SOURCE_KERNEL = "rodrimandub/af-pikans-dfg-re20-competitive-pilot-t4x2"
SEEDS = [501, 502, 503, 504, 505]
ARMS = ["DFG-RBF-PIKAN", "DFG-Tanh-MLP"]
PAIR_CAP_SECONDS = 540.0
PAIR_CLOSE_SECONDS = 580.0

def stop_kernel():
    os.kill(os.getpid(), signal.SIGTERM)

watchdog = threading.Timer(WATCHDOG_SECONDS, stop_kernel)
watchdog.daemon = True
watchdog.start()

SESSION = Path("/kaggle/working/dfg-re20-lbfgs-refinement") / ("run-" + str(time.time_ns()))
SESSION.mkdir(parents=True, exist_ok=False)
RUNNER_BYTES = base64.b64decode("__RUNNER_PAYLOAD__")
WORKER_BYTES = base64.b64decode("__WORKER_PAYLOAD__")
RUNNER_PATH = SESSION / "run_campaign_source.py"
WORKER_PATH = SESSION / "dfg_lbfgs_worker.py"
RUNNER_PATH.write_bytes(RUNNER_BYTES)
WORKER_PATH.write_bytes(WORKER_BYTES)

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def require_t4x2():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Expected exactly two CUDA devices; CPU fallback is disabled.")
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if any("T4" not in name.upper() for name in names):
        raise RuntimeError(f"Expected two Tesla T4 devices; found {names!r}")
    return names

def source_missing(root):
    missing = []
    for name in ("campaign_results.json", "hardware.json"):
        if not (root / name).is_file():
            missing.append(name)
    for seed in SEEDS:
        if not (root / "pools" / f"seed-{seed}.npz").is_file():
            missing.append(f"pools/seed-{seed}.npz")
        for arm in ARMS:
            directory = root / arm / f"seed-{seed}"
            for name in ("checkpoint-final.pt", "training_summary.json"):
                if not (directory / name).is_file():
                    missing.append(f"{arm}/seed-{seed}/{name}")
    return missing

def campaign_roots(base):
    return sorted({p.parent.resolve() for p in base.rglob("campaign_results.json")})

def extract_campaign_archive(archive_path, target):
    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            member = PurePosixPath(item.filename)
            parts = member.parts
            if item.is_dir() or not parts or member.is_absolute() or ".." in parts:
                continue
            name = parts[-1]
            wanted = name in ("campaign_results.json", "hardware.json", "checkpoint-final.pt", "training_summary.json")
            wanted = wanted or (name.startswith("seed-") and name.endswith(".npz") and "pools" in parts)
            if not wanted:
                continue
            destination = target.joinpath(*parts).resolve()
            if os.path.commonpath((str(target.resolve()), str(destination))) != str(target.resolve()):
                raise RuntimeError("Unsafe Kaggle source archive path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as src, destination.open("wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
            extracted.append(str(destination))
    return extracted

def find_source_campaign():
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        raise FileNotFoundError("Kaggle source-kernel input mount is missing")
    roots = campaign_roots(input_root)
    complete = [root for root in roots if not source_missing(root)]
    extracted = []
    if not complete:
        archive_root = SESSION / "source-extracts"
        for archive_path in sorted(input_root.rglob("*.zip")):
            if time.perf_counter() >= DEADLINE - 60:
                break
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    if not any(PurePosixPath(name).name == "campaign_results.json" for name in archive.namelist()):
                        continue
                extracted.extend(extract_campaign_archive(archive_path, archive_root / (archive_path.stem + "-" + sha256(archive_path)[:12])))
            except zipfile.BadZipFile:
                continue
        roots = campaign_roots(input_root) + campaign_roots(archive_root)
        complete = [root for root in roots if not source_missing(root)]
    if len(complete) != 1:
        raise FileNotFoundError(json.dumps({"message": "Expected one complete source DFG campaign", "roots": [str(p) for p in roots], "extracted": extracted[:60]}))
    return complete[0]

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

try:
    import numpy as np
    import scipy
    import torch

    GPU_NAMES = require_t4x2()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    if sha256(RUNNER_PATH) != json.loads((find_source_campaign() / "campaign_results.json").read_text(encoding="utf-8")).get("source_sha256"):
        raise RuntimeError("Embedded DFG runner hash does not match the source campaign")
    CAMPAIGN_ROOT = find_source_campaign()
    SOURCE_SUMMARY = json.loads((CAMPAIGN_ROOT / "campaign_results.json").read_text(encoding="utf-8"))
    SOURCE_HARDWARE = json.loads((CAMPAIGN_ROOT / "hardware.json").read_text(encoding="utf-8"))
    if "T4" not in str(SOURCE_HARDWARE.get("device", "")).upper():
        raise RuntimeError("Source campaign was not recorded on Tesla T4")
    statuses = {(row.get("arm"), int(row.get("seed"))): row.get("status") for row in SOURCE_SUMMARY.get("results", []) if isinstance(row, dict)}
    if any(statuses.get((arm, seed)) != "score_budget_exhausted" for seed in SEEDS for arm in ARMS):
        raise RuntimeError("Source checkpoints do not all have the expected completed training status")

    campaign = load_module(RUNNER_PATH, "dfg_refinement_campaign")
    FORCE_CONVENTION = "FEATFLOW sigma=nu*grad(u)-pI; direct sigma*n traction with radial-outward cylinder normal"
    def corrected_qoi_for_model(model, device, surface_count=512, deadline=None):
        theta = campaign.math.pi * 2.0 * campaign.np.arange(surface_count, dtype=campaign.np.float64) / surface_count
        xy_np = campaign.np.column_stack((campaign.CX + campaign.RADIUS * campaign.np.cos(theta), campaign.CY + campaign.RADIUS * campaign.np.sin(theta)))
        surface = torch.as_tensor(xy_np, dtype=torch.float32, device=device).requires_grad_(True)
        fields = model(surface)
        gu = campaign.grad_scalar(fields[:, 0:1], surface)
        gv = campaign.grad_scalar(fields[:, 1:2], surface)
        nx = torch.as_tensor(campaign.np.cos(theta), dtype=torch.float32, device=device).reshape(-1, 1)
        ny = torch.as_tensor(campaign.np.sin(theta), dtype=torch.float32, device=device).reshape(-1, 1)
        ds = 2.0 * campaign.math.pi * campaign.RADIUS / surface_count
        drag = (campaign.NU * (gu[:, 0:1] * nx + gu[:, 1:2] * ny) - fields[:, 2:3] * nx).sum() * ds
        lift = (campaign.NU * (gv[:, 0:1] * nx + gv[:, 1:2] * ny) - fields[:, 2:3] * ny).sum() * ds
        scale = 2.0 / (campaign.U_MEAN ** 2 * campaign.DIAMETER)
        front = torch.tensor([[campaign.CX - campaign.RADIUS, campaign.CY]], dtype=torch.float32, device=device)
        rear = torch.tensor([[campaign.CX + campaign.RADIUS, campaign.CY]], dtype=torch.float32, device=device)
        p_front = model(front)[0, 2]
        p_rear = model(rear)[0, 2]
        return {
            "drag_force": float(drag.detach().cpu()),
            "lift_force": float(lift.detach().cpu()),
            "drag_coefficient": float((scale * drag).detach().cpu()),
            "lift_coefficient": float((scale * lift).detach().cpu()),
            "pressure_front": float(p_front.detach().cpu()),
            "pressure_rear": float(p_rear.detach().cpu()),
            "pressure_difference": float((p_front - p_rear).detach().cpu()),
            "surface_count": surface_count,
            "qoi_force_convention": FORCE_CONVENTION,
        }

    campaign.qoi_for_model = corrected_qoi_for_model
    TRAIN_ROOT = SESSION / "training"
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    train_status = []

    def launch_pair(seed):
        if time.perf_counter() >= DEADLINE - 900:
            return {arm: False for arm in ARMS}, "global_safety_deadline"
        child_records = []
        pair_started = time.perf_counter()
        pair_deadline = min(pair_started + PAIR_CLOSE_SECONDS, DEADLINE - 900)
        for gpu_index, arm in enumerate(ARMS):
            out = TRAIN_ROOT / arm / f"seed-{seed}"
            out.mkdir(parents=True, exist_ok=True)
            source_checkpoint = CAMPAIGN_ROOT / arm / f"seed-{seed}" / "checkpoint-final.pt"
            source_pool = CAMPAIGN_ROOT / "pools" / f"seed-{seed}.npz"
            command = [sys.executable, str(WORKER_PATH), "--runner", str(RUNNER_PATH), "--checkpoint", str(source_checkpoint), "--pool", str(source_pool), "--output", str(out), "--arm", arm, "--seed", str(seed), "--seconds", str(PAIR_CAP_SECONDS)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            log = (out / "worker.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            child_records.append((arm, out, process, log))
        while any(proc.poll() is None for _, _, proc, _ in child_records):
            if time.perf_counter() >= pair_deadline:
                for _, _, proc, _ in child_records:
                    if proc.poll() is None:
                        proc.terminate()
                for _, _, proc, _ in child_records:
                    if proc.poll() is None:
                        try:
                            proc.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                break
            time.sleep(1.0)
        statuses = {}
        for arm, out, proc, log in child_records:
            proc.poll()
            log.close()
            summary_path = out / "lbfgs_summary.json"
            statuses[arm] = bool(proc.returncode == 0 and summary_path.is_file() and (out / "checkpoint-lbfgs.pt").is_file())
            if summary_path.is_file():
                train_status.append(json.loads(summary_path.read_text(encoding="utf-8")))
            else:
                write_json(out / "failure.json", {"status": "worker_failed_or_pair_timeout", "arm": arm, "seed": seed, "returncode": proc.returncode})
        write_json(SESSION / f"seed-{seed}-training-status.json", {"statuses": statuses, "pair_wall_seconds": time.perf_counter() - pair_started})
        return statuses, "complete" if all(statuses.values()) else "partial"

    for seed in SEEDS:
        statuses, status = launch_pair(seed)
        if not any(statuses.values()) and time.perf_counter() >= DEADLINE - 900:
            break
        if not any(statuses.values()) and status == "global_safety_deadline":
            break

    def score_checkpoint(arm, seed, checkpoint_path, stage):
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        model = campaign.build_model(arm, seed).to(device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        q512 = campaign.qoi_for_model(model, device, 512, DEADLINE - 90)
        q1024 = campaign.qoi_for_model(model, device, 1024, DEADLINE - 90)
        pools, _ = campaign.make_pools(seed)
        diagnostics = campaign.heldout_diagnostics(model, pools["heldout"], pools, device, DEADLINE - 90)
        ref = campaign.FEATFLOW
        return {
            "status": "scored",
            "arm": arm,
            "seed": seed,
            "stage": stage,
            "qois": q512,
            "featflow_reference": ref,
            "qoi_errors": {
                "drag_relative_error": abs(q512["drag_coefficient"] - ref["drag_coefficient"]) / ref["drag_coefficient"],
                "lift_absolute_error": abs(q512["lift_coefficient"] - ref["lift_coefficient"]),
                "pressure_difference_relative_error": abs(q512["pressure_difference"] - ref["pressure_difference"]) / ref["pressure_difference"],
            },
            "quadrature_512_to_1024_absolute_change": {key: abs(q1024[key] - q512[key]) for key in ("drag_coefficient", "lift_coefficient", "pressure_difference")},
            "diagnostics": diagnostics,
            "checkpoint_sha256": sha256(checkpoint_path),
            "field_relative_l2": None,
            "field_reference_status": "missing_converged_FEM_field",
            "scientific_accepted": False,
            "device": torch.cuda.get_device_name(device),
            "cuda_device_index": int(device.index),
        }

    results = []
    score_before = SESSION / "scores-before"
    score_after = SESSION / "scores-after"
    for seed in SEEDS:
        for arm in ARMS:
            source_checkpoint = CAMPAIGN_ROOT / arm / f"seed-{seed}" / "checkpoint-final.pt"
            before = score_checkpoint(arm, seed, source_checkpoint, "adam_checkpoint_before_lbfgs")
            write_json(score_before / f"{arm}-seed-{seed}.json", before)
            refined = TRAIN_ROOT / arm / f"seed-{seed}" / "checkpoint-lbfgs.pt"
            if refined.is_file():
                after = score_checkpoint(arm, seed, refined, "lbfgs_refined")
                write_json(score_after / f"{arm}-seed-{seed}.json", after)
                after["qoi_delta_vs_adam"] = {key: after["qois"][key] - before["qois"][key] for key in ("drag_coefficient", "lift_coefficient", "pressure_difference")}
                results.append(after)
            else:
                results.append({"status": "not_refined", "arm": arm, "seed": seed, "baseline_qois": before["qois"], "field_relative_l2": None, "scientific_accepted": False})

    columns = ("arm", "seed", "status", "stage", "drag_coefficient", "lift_coefficient", "pressure_difference", "drag_relative_error", "lift_absolute_error", "pressure_difference_relative_error", "heldout_momentum_x_rms", "heldout_momentum_y_rms", "heldout_continuity_rms", "scientific_accepted")
    with (SESSION / "lbfgs-refinement-results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            flat = {key: row.get(key) for key in columns}
            flat.update(row.get("qois", {}))
            flat.update(row.get("qoi_errors", {}))
            flat.update(row.get("diagnostics", {}))
            writer.writerow(flat)

    completed = sum(row.get("status") == "scored" for row in results)
    report = {
        "status": "lbfgs_refinement_complete" if completed == 10 else "lbfgs_refinement_partial",
        "source_kernel": SOURCE_KERNEL,
        "source_campaign_root": str(CAMPAIGN_ROOT),
        "source_runner_sha256": sha256(RUNNER_PATH),
        "source_campaign_sha256": SOURCE_SUMMARY.get("source_sha256"),
        "source_hardware": SOURCE_HARDWARE,
        "force_convention": FORCE_CONVENTION,
        "seeds": SEEDS,
        "arms": ARMS,
        "lbfgs_pair_cap_seconds": PAIR_CAP_SECONDS,
        "session_quota_cap_seconds": QUOTA_CAP_SECONDS,
        "watchdog_seconds_from_first_code_cell": WATCHDOG_SECONDS,
        "session_wall_seconds_from_first_code_cell": time.perf_counter() - SESSION_STARTED,
        "gpu_names": GPU_NAMES,
        "training_summaries": train_status,
        "successful_refined_scores": completed,
        "expected_refined_scores": 10,
        "results": results,
        "field_relative_l2": None,
        "field_reference_status": "missing_converged_FEM_field",
        "scientific_accepted": False,
    }
    write_json(SESSION / "lbfgs-refinement-results.json", report)
    print(json.dumps({"status": report["status"], "scores": completed, "session": str(SESSION), "wall_seconds": report["session_wall_seconds_from_first_code_cell"], "quota_cap_seconds": QUOTA_CAP_SECONDS, "remaining_field_reference": report["field_reference_status"], "scientific_accepted": False}, indent=2))
finally:
    watchdog.cancel()
'''

runner_b64 = base64.b64encode(RUNNER.read_bytes()).decode("ascii")
worker_b64 = base64.b64encode(WORKER.read_bytes()).decode("ascii")
code = CELL.replace("__RUNNER_PAYLOAD__", runner_b64).replace("__WORKER_PAYLOAD__", worker_b64)
markdown = """# DFG Re=20: L-BFGS refinement of existing GPU checkpoints\n\nThis Kaggle-only continuation loads the five paired RBF/MLP checkpoints and their saved Sobol pools from the completed T4x2 kernel. It applies deterministic full-pool L-BFGS to both models for up to 540 seconds per seed pair, then reports before/after FEATFLOW QoIs using direct stress traction. It does not repeat Adam training and does not claim a converged FEM field or scientific acceptance.\n\nHard session cap: 7,200 seconds; internal watchdog: 6,900 seconds; soft deadline: 6,500 seconds.\n"""

def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)

payload = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": source_lines(markdown)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(code)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Generated {NOTEBOOK} ({NOTEBOOK.stat().st_size} bytes)")
