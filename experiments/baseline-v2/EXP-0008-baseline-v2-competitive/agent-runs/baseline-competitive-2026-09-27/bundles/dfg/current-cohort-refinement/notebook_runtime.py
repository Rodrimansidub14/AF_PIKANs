import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

SESSION_STARTED = time.perf_counter()
QUOTA_CAP_SECONDS = 7200.0
SOFT_DEADLINE_SECONDS = 3500.0
WATCHDOG_SECONDS = 6800.0
DEADLINE = SESSION_STARTED + SOFT_DEADLINE_SECONDS
SOURCE_KERNELS = {
    "r3": "rodrimandub/af-pikans-dfg-re20-baseline-20260927-r3",
    "r4": "rodrimandub/af-pikans-dfg-re20-baseline-20260927-r4",
}
INPUT_DATASET_ID = "rodrimandub/af-pikans-dfg-rbf-inputs-20260928"
INPUT_DATASET_MOUNT_NAME = INPUT_DATASET_ID.rsplit("/", 1)[-1]
SOURCE_ROUTE_NAMES = {name: slug.rsplit("/", 1)[-1] for name, slug in SOURCE_KERNELS.items()}
SEEDS_BY_SOURCE = {"r3": (301, 302), "r4": (303, 304, 305)}
SEEDS = (301, 302, 303, 304, 305)
SEED_SOURCE = {seed: source for source, seeds in SEEDS_BY_SOURCE.items() for seed in seeds}
REQUIRED_SEED_FILES = (
    "dfg_rbf_pikan.pt",
    "dfg_training_pools.npz",
    "run.json",
    "provenance.json",
    "manifest_snapshot.json",
)
PAIR_SEED_SECONDS = 540.0
PAIR_WALL_LIMIT_SECONDS = 610.0
LBFGS_CHUNK_SIZE = 1024
HELDOUT_COUNT = 16384
HELDOUT_BOUNDARY_COUNT = 512

SESSION = Path("/kaggle/working/dfg-current-cohort-refinement") / ("run-" + str(time.time_ns()))
SESSION.mkdir(parents=True, exist_ok=False)
VENDOR_ROOT = SESSION / "vendor"
WORKER_PATH = SESSION / "dfg_rbf_lbfgs_worker.py"
SOURCE_MODULES = __EMBEDDED_MODULES__
SOURCE_MODULE_HASHES = __SOURCE_MODULE_HASHES__
WORKER_BYTES = base64.b64decode("__WORKER_BASE64__")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_t4x2(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Expected exactly two CUDA devices; CPU fallback is disabled")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name.upper() for name in names):
        raise RuntimeError(f"Expected two Tesla T4 devices; found {names!r}")
    return names


def install_official_modules():
    module_hashes = {}
    for relative_path, encoded in SOURCE_MODULES.items():
        payload = base64.b64decode(encoded)
        destination = VENDOR_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        module_hashes[relative_path] = sha256(destination)
        if module_hashes[relative_path] != SOURCE_MODULE_HASHES[relative_path]:
            raise RuntimeError(f"Embedded official module hash mismatch: {relative_path}")
    for relative_path in (
        "src/__init__.py",
        "src/research_pipeline/__init__.py",
        "src/research_pipeline/sciml/__init__.py",
    ):
        destination = VENDOR_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")
    WORKER_PATH.write_bytes(WORKER_BYTES)
    return module_hashes


def json_file(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse {label} at {path}: {type(exc).__name__}: {exc}") from exc


def seed_file_candidates(mount, seed, filename):
    wanted_parent = f"seed-{seed}"
    return sorted(
        path.resolve()
        for path in Path(mount).rglob(filename)
        if path.is_file() and path.parent.name == wanted_parent
    )


def seed_checkpoint_files(mount):
    found = {}
    for path in Path(mount).rglob("dfg_rbf_pikan.pt"):
        if path.is_file() and path.parent.name.startswith("seed-"):
            suffix = path.parent.name.removeprefix("seed-")
            if suffix.isdigit():
                found.setdefault(int(suffix), []).append(path.resolve())
    return found


def find_exact_source_mounts(input_root):
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError("Kaggle input mount /kaggle/input is missing")
    dataset_mounts = [
        path.resolve()
        for path in input_root.iterdir()
        if path.is_dir() and path.name.casefold() == INPUT_DATASET_MOUNT_NAME.casefold()
    ]
    if len(dataset_mounts) != 1:
        raise RuntimeError(
            json.dumps(
                {
                    "message": "Expected exactly one mounted private source dataset",
                    "dataset_id": INPUT_DATASET_ID,
                    "expected_mount_name": INPUT_DATASET_MOUNT_NAME,
                    "matching_mounts": [str(path) for path in dataset_mounts],
                },
                sort_keys=True,
            )
        )
    dataset_root = dataset_mounts[0]

    def is_within(path, root):
        try:
            return os.path.commonpath((str(Path(path).resolve()), str(Path(root).resolve()))) == str(Path(root).resolve())
        except ValueError:
            return False

    def route_candidates(search_root, source_name):
        owner = SOURCE_KERNELS[source_name].split("/", 1)[0]
        route_name = SOURCE_ROUTE_NAMES[source_name]
        matches = []
        for current, directory_names, _file_names in os.walk(search_root):
            if Path(current).name.casefold() != owner.casefold():
                continue
            matches.extend(
                (Path(current) / directory_name).resolve()
                for directory_name in directory_names
                if directory_name.casefold() == route_name.casefold()
            )
        return sorted(matches)

    selected = {}
    for source_name, expected_route_name in SOURCE_ROUTE_NAMES.items():
        stray_matches = route_candidates(input_root, source_name)
        matches = route_candidates(dataset_root, source_name)
        if len(stray_matches) != 1 or len(matches) != 1 or stray_matches[0] != matches[0]:
            raise RuntimeError(
                json.dumps(
                    {
                        "message": "Expected exactly one exact source-kernel route inside the configured dataset",
                        "dataset_id": INPUT_DATASET_ID,
                        "dataset_mount": str(dataset_root),
                        "source": source_name,
                        "kernel_slug": SOURCE_KERNELS[source_name],
                        "expected_route_name": expected_route_name,
                        "matching_routes_in_dataset": [str(path) for path in matches],
                        "matching_routes_under_input": [str(path) for path in stray_matches],
                    },
                    sort_keys=True,
                )
            )
        selected[source_name] = matches[0]

    for current, _directory_names, file_names in os.walk(input_root):
        if "dfg_rbf_pikan.pt" not in file_names:
            continue
        checkpoint = Path(current) / "dfg_rbf_pikan.pt"
        parent_name = checkpoint.parent.name
        if not parent_name.startswith("seed-") or not parent_name.removeprefix("seed-").isdigit():
            continue
        seed = int(parent_name.removeprefix("seed-"))
        if seed not in SEED_SOURCE:
            continue
        expected_root = selected[SEED_SOURCE[seed]]
        if not is_within(checkpoint, expected_root):
            raise RuntimeError(
                f"Target seed {seed} checkpoint appears outside its exact source route "
                f"{SOURCE_KERNELS[SEED_SOURCE[seed]]}: {checkpoint}"
            )
    return dataset_root, selected


def validate_source_seed(source_name, mount, seed):
    candidates = seed_file_candidates(mount, seed, "dfg_rbf_pikan.pt")
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint for seed {seed} under {SOURCE_KERNELS[source_name]}; found {len(candidates)}"
        )
    seed_dir = candidates[0].parent
    files = {}
    for filename in REQUIRED_SEED_FILES:
        matches = seed_file_candidates(mount, seed, filename)
        if len(matches) != 1 or matches[0].parent != seed_dir:
            raise RuntimeError(
                f"Expected one {filename} beside seed {seed} checkpoint under {SOURCE_KERNELS[source_name]}; "
                f"found {[str(path) for path in matches]}"
            )
        files[filename] = matches[0]

    run = json_file(files["run.json"], "run.json")
    provenance = json_file(files["provenance.json"], "provenance.json")
    manifest = json_file(files["manifest_snapshot.json"], "manifest_snapshot.json")
    try:
        run_seed = int(run["seed"])
        training_seed = int(run["training_config"]["seed"])
        provenance_seed = int(provenance["sampling_seed_stream"]["training"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing or invalid run/provenance seed id for seed {seed}") from exc
    if run_seed != seed or training_seed != seed or provenance_seed != seed:
        raise RuntimeError(
            f"Seed identity mismatch at {seed_dir}: run.seed={run_seed}, training.seed={training_seed}, "
            f"provenance.training={provenance_seed}"
        )
    if run.get("problem") != "dfg_2d_1_steady_re20" or provenance.get("problem_id") != "dfg_2d_1_steady_re20":
        raise RuntimeError(f"Seed {seed} is not the expected DFG Re=20 problem")
    expected_model = "dfg_rbf_fixed_2_16_16_3"
    run_model = run.get("model", {})
    provenance_model = provenance.get("model", {})
    if run_model.get("id") != expected_model or provenance_model.get("id") != expected_model:
        raise RuntimeError(f"Seed {seed} does not identify the expected r4 RBF-PIKAN model")
    if run_model.get("architecture") != [2, 16, 16, 3] or provenance_model.get("architecture") != [2, 16, 16, 3]:
        raise RuntimeError(f"Seed {seed} architecture does not match [2,16,16,3]")
    if int(run_model.get("basis_count", -1)) != 11 or int(provenance_model.get("basis_count", -1)) != 11:
        raise RuntimeError(f"Seed {seed} does not use 11 RBF bases")

    checkpoint_hash = sha256(files["dfg_rbf_pikan.pt"])
    pool_hash = sha256(files["dfg_training_pools.npz"])
    run_pool_hash = run.get("pool_archive_sha256")
    if run.get("pool_archive") != "dfg_training_pools.npz" or run_pool_hash != pool_hash:
        raise RuntimeError(f"Seed {seed} saved-pool SHA-256 does not match run.json")
    file_hashes = {filename: sha256(path) for filename, path in files.items()}
    return {
        "seed": seed,
        "source": source_name,
        "source_kernel": SOURCE_KERNELS[source_name],
        "source_mount": str(Path(mount).resolve()),
        "seed_directory": str(seed_dir),
        "files": {filename: str(path) for filename, path in files.items()},
        "sha256": file_hashes,
        "checkpoint_sha256": checkpoint_hash,
        "pool_sha256": pool_hash,
        "run_seed": run_seed,
        "provenance_training_seed": provenance_seed,
        "manifest_model_id": manifest.get("model", {}).get("id"),
    }


def discover_sources():
    dataset_root, selected_mounts = find_exact_source_mounts("/kaggle/input")
    source_by_seed = {}
    for source_name, mount in selected_mounts.items():
        found = seed_checkpoint_files(mount)
        expected_seeds = set(SEEDS_BY_SOURCE[source_name])
        present_target_seeds = set(found).intersection(SEEDS)
        if present_target_seeds != expected_seeds:
            raise RuntimeError(
                f"Source {SOURCE_KERNELS[source_name]} must contain exactly target seeds {sorted(expected_seeds)}; "
                f"found target seeds {sorted(present_target_seeds)}"
            )
        for seed in SEEDS_BY_SOURCE[source_name]:
            if seed in source_by_seed:
                raise RuntimeError(f"Seed {seed} maps to more than one source kernel")
            source_by_seed[seed] = validate_source_seed(source_name, mount, seed)
    if set(source_by_seed) != set(SEEDS):
        raise RuntimeError(f"The source loader did not resolve exactly the required seed set {list(SEEDS)}")
    for seed, source in SEED_SOURCE.items():
        if source_by_seed[seed]["source"] != source:
            raise RuntimeError(f"Seed {seed} resolved to {source_by_seed[seed]['source']}, expected {source}")
    return dataset_root, selected_mounts, source_by_seed


def prepare_heldout(dfg_module, seed):
    heldout_seed = (int(seed) + 100000) % (2**32 - 1)
    pools, provenance = dfg_module.generate_dfg_pools(
        heldout_seed,
        interior_count=HELDOUT_COUNT,
        boundary_count=HELDOUT_BOUNDARY_COUNT,
        return_provenance=True,
    )
    points = pools["interior"]
    return {
        "seed": heldout_seed,
        "points": points,
        "points_sha256": provenance["pool_sha256"]["interior"],
        "pool_provenance": provenance,
        "origin": "official_r4_scrambled_sobol_generator_reconstructed_from_heldout_seed",
    }


def load_model(dfg_module, source_record, checkpoint_path, device):
    seed = int(source_record["seed"])
    payload = __import__("torch").load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError(f"Checkpoint for seed {seed} has no state_dict")
    if "output_scales" not in payload["state_dict"]:
        raise RuntimeError(f"Checkpoint for seed {seed} lacks the r4 output_scales buffer")
    saved_run = payload.get("run", {})
    saved_seed = saved_run.get("seed") if isinstance(saved_run, dict) else None
    if saved_seed is not None and int(saved_seed) != seed:
        raise RuntimeError(f"Checkpoint payload seed {saved_seed} does not match seed directory {seed}")
    model = dfg_module.DFGRBFPIKAN(seed=seed).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def score_checkpoint(dfg_module, campaign_module, source_record, checkpoint_path, stage, device, heldout):
    import torch

    seed = int(source_record["seed"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model = load_model(dfg_module, source_record, checkpoint_path, device)
    qoi512 = dfg_module.evaluate_dfg_qois(model, device=device, surface_count=512)
    qoi1024 = dfg_module.evaluate_dfg_qois(model, device=device, surface_count=1024)
    residual_arrays, residual_seconds = campaign_module._evaluate_heldout(
        model, device, heldout["points"], chunk_size=1024
    )
    residual_regions = campaign_module.summarize_residual_regions(
        heldout["points"], residual_arrays, near_distance=0.15
    )
    reference = dict(dfg_module.FEATFLOW_REFERENCE)
    errors = {
        "drag_relative_error": abs(qoi512["drag_coefficient"] - reference["drag_coefficient"])
        / reference["drag_coefficient"],
        "lift_absolute_error": abs(qoi512["lift_coefficient"] - reference["lift_coefficient"]),
        "pressure_difference_relative_error": abs(qoi512["pressure_difference"] - reference["pressure_difference"])
        / reference["pressure_difference"],
    }
    quadrature_change = {
        key: abs(qoi1024[key] - qoi512[key])
        for key in ("drag_coefficient", "lift_coefficient", "pressure_difference")
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "status": "scored",
        "seed": seed,
        "source": source_record["source"],
        "source_kernel": source_record["source_kernel"],
        "stage": stage,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "qois_512": qoi512,
        "qois_1024": qoi1024,
        "quadrature_512_to_1024_absolute_change": quadrature_change,
        "featflow_reference": reference,
        "qoi_errors_vs_featflow": errors,
        "heldout_residual_regions": residual_regions,
        "heldout_residual_seconds": residual_seconds,
        "heldout_point_count": int(len(heldout["points"])),
        "heldout_seed": heldout["seed"],
        "heldout_points_sha256": heldout["points_sha256"],
        "heldout_points_origin": heldout["origin"],
        "device": torch.cuda.get_device_name(device),
        "cuda_device_index": int(device.index),
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "field_relative_l2": None,
        "scientific_accepted": False,
    }
    del model, residual_arrays
    torch.cuda.empty_cache()
    return result


def launch_wave(wave_number, seeds, source_by_seed, worker_path, module_root, gpu_names, train_root):
    children = []
    started = time.perf_counter()
    launch_deadline = min(started + PAIR_WALL_LIMIT_SECONDS, DEADLINE - 180.0)
    for offset, seed in enumerate(seeds):
        device_index = offset
        source = source_by_seed[seed]
        out = train_root / f"seed-{seed}"
        out.mkdir(parents=True, exist_ok=True)
        files = source["files"]
        command = [
            sys.executable,
            str(worker_path),
            "--module-root",
            str(module_root),
            "--checkpoint",
            files["dfg_rbf_pikan.pt"],
            "--pool",
            files["dfg_training_pools.npz"],
            "--output",
            str(out),
            "--seed",
            str(seed),
            "--source-kernel",
            source["source_kernel"],
            "--source-checkpoint-sha256",
            source["checkpoint_sha256"],
            "--source-pool-sha256",
            source["pool_sha256"],
            "--source-run-sha256",
            source["sha256"]["run.json"],
            "--source-provenance-sha256",
            source["sha256"]["provenance.json"],
            "--device-name",
            gpu_names[device_index],
            "--global-device-index",
            str(device_index),
            "--seconds",
            str(PAIR_SEED_SECONDS),
            "--chunk-size",
            str(LBFGS_CHUNK_SIZE),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
        log_path = out / "worker.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        children.append(
            {
                "seed": seed,
                "source": source["source"],
                "device_index": device_index,
                "device_name": gpu_names[device_index],
                "output": out,
                "log_path": log_path,
                "log": log,
                "process": process,
                "started_offset_seconds": time.perf_counter() - started,
            }
        )

    timed_out = False
    while any(child["process"].poll() is None for child in children):
        if time.perf_counter() >= launch_deadline:
            timed_out = True
            for child in children:
                if child["process"].poll() is None:
                    child["process"].terminate()
            for child in children:
                process = child["process"]
                if process.poll() is None:
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=8)
            break
        time.sleep(1.0)

    outcomes = {}
    for child in children:
        process = child["process"]
        return_code = process.poll()
        child["log"].close()
        summary_path = child["output"] / "lbfgs_summary.json"
        if summary_path.is_file():
            summary = json_file(summary_path, "L-BFGS worker summary")
        else:
            summary = {
                "status": "worker_failed_or_wave_timeout",
                "seed": child["seed"],
                "source": child["source"],
                "device": child["device_name"],
                "global_device_index": child["device_index"],
                "returncode": return_code,
                "wave_timed_out": timed_out,
                "log_tail": child["log_path"].read_text(encoding="utf-8", errors="replace")[-5000:],
                "elapsed_seconds": time.perf_counter() - started,
                "field_relative_l2": None,
                "scientific_accepted": False,
            }
            write_json(child["output"] / "failure.json", summary)
        summary["wave"] = wave_number
        summary["device"] = summary.get("device", child["device_name"])
        summary["global_device_index"] = child["device_index"]
        summary["process_returncode"] = return_code
        summary["wave_timed_out"] = timed_out
        summary["status"] = "lbfgs_refined" if return_code == 0 and int(summary.get("lbfgs_steps", 0)) > 0 else summary.get("status", "worker_failed")
        write_json(summary_path, summary)
        outcomes[child["seed"]] = summary
    return outcomes, time.perf_counter() - started


def main():
    import numpy as np
    import scipy
    import torch

    watchdog = threading.Timer(WATCHDOG_SECONDS, lambda: os.kill(os.getpid(), signal.SIGTERM))
    watchdog.daemon = True
    watchdog.start()
    try:
        module_hashes = install_official_modules()
        GPU_NAMES = require_t4x2(torch)
        torch.set_num_threads(2)
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        sys.path.insert(0, str(VENDOR_ROOT))
        from src.research_pipeline.sciml import dfg as dfg_module
        from src.research_pipeline.sciml import dfg_campaign as campaign_module

        dataset_root, mounts, source_by_seed = discover_sources()
        write_json(
            SESSION / "source-validation.json",
            {
                "status": "validated",
                "input_dataset_id": INPUT_DATASET_ID,
                "input_dataset_mount": str(dataset_root),
                "source_kernels": SOURCE_KERNELS,
                "source_mounts": {key: str(path.resolve()) for key, path in mounts.items()},
                "seeds_by_source": SEEDS_BY_SOURCE,
                "source_by_seed": source_by_seed,
                "official_module_sha256": module_hashes,
                "gpu_names": GPU_NAMES,
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
        )
        heldout_by_seed = {seed: prepare_heldout(dfg_module, seed) for seed in SEEDS}
        before_scores = {}
        after_scores = {}
        for seed in SEEDS:
            if time.perf_counter() >= DEADLINE - 220.0:
                break
            score = score_checkpoint(
                dfg_module,
                campaign_module,
                source_by_seed[seed],
                source_by_seed[seed]["files"]["dfg_rbf_pikan.pt"],
                "adam_checkpoint_before_lbfgs",
                torch.device("cuda:0"),
                heldout_by_seed[seed],
            )
            before_scores[seed] = score
            write_json(SESSION / "scores-before" / f"seed-{seed}.json", score)

        train_root = SESSION / "training"
        train_root.mkdir(parents=True, exist_ok=True)
        worker_summaries = {}
        wave_summaries = []
        waves = [SEEDS[index : index + 2] for index in range(0, len(SEEDS), 2)]
        for wave_number, wave in enumerate(waves, start=1):
            if time.perf_counter() >= DEADLINE - 800.0:
                for seed in wave:
                    worker_summaries[seed] = {
                        "status": "not_started_soft_deadline",
                        "seed": seed,
                        "source": SEED_SOURCE[seed],
                        "field_relative_l2": None,
                        "scientific_accepted": False,
                    }
                continue
            outcomes, wall_seconds = launch_wave(
                wave_number,
                wave,
                source_by_seed,
                WORKER_PATH,
                VENDOR_ROOT,
                GPU_NAMES,
                train_root,
            )
            worker_summaries.update(outcomes)
            wave_summaries.append(
                {
                    "wave": wave_number,
                    "seeds": list(wave),
                    "wall_seconds": wall_seconds,
                    "workers": {str(seed): outcomes.get(seed) for seed in wave},
                }
            )
            write_json(SESSION / "training" / f"wave-{wave_number}.json", wave_summaries[-1])

        for seed in SEEDS:
            training = worker_summaries.get(seed, {})
            checkpoint_path = train_root / f"seed-{seed}" / "checkpoint-lbfgs.pt"
            if (
                int(training.get("lbfgs_steps", 0)) > 0
                and checkpoint_path.is_file()
                and time.perf_counter() < DEADLINE - 100.0
            ):
                score = score_checkpoint(
                    dfg_module,
                    campaign_module,
                    source_by_seed[seed],
                    checkpoint_path,
                    "lbfgs_refined",
                    torch.device("cuda:0"),
                    heldout_by_seed[seed],
                )
                score["qoi_delta_vs_before"] = {
                    key: score["qois_512"][key] - before_scores[seed]["qois_512"][key]
                    for key in ("drag_coefficient", "lift_coefficient", "pressure_difference")
                } if seed in before_scores else None
                after_scores[seed] = score
                write_json(SESSION / "scores-after" / f"seed-{seed}.json", score)

        results = []
        for seed in SEEDS:
            training = worker_summaries.get(
                seed,
                {"status": "not_started", "seed": seed, "source": SEED_SOURCE[seed]},
            )
            before = before_scores.get(seed)
            after = after_scores.get(seed)
            status = "complete" if before is not None and after is not None and int(training.get("lbfgs_steps", 0)) > 0 else "partial"
            results.append(
                {
                    "seed": seed,
                    "source": SEED_SOURCE[seed],
                    "source_kernel": SOURCE_KERNELS[SEED_SOURCE[seed]],
                    "source_artifacts": source_by_seed[seed],
                    "training": training,
                    "before": before,
                    "after": after,
                    "status": status,
                    "field_relative_l2": None,
                    "scientific_accepted": False,
                }
            )

        columns = (
            "seed",
            "source",
            "status",
            "lbfgs_steps",
            "training_elapsed_seconds",
            "device",
            "before_drag_coefficient",
            "after_drag_coefficient",
            "before_lift_coefficient",
            "after_lift_coefficient",
            "before_pressure_difference",
            "after_pressure_difference",
            "field_relative_l2",
            "scientific_accepted",
        )
        with (SESSION / "dfg-current-cohort-refinement.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                before = row.get("before") or {}
                after = row.get("after") or {}
                writer.writerow(
                    {
                        "seed": row["seed"],
                        "source": row["source"],
                        "status": row["status"],
                        "lbfgs_steps": row.get("training", {}).get("lbfgs_steps"),
                        "training_elapsed_seconds": row.get("training", {}).get("elapsed_seconds"),
                        "device": row.get("training", {}).get("device"),
                        "before_drag_coefficient": (before.get("qois_512") or {}).get("drag_coefficient"),
                        "after_drag_coefficient": (after.get("qois_512") or {}).get("drag_coefficient"),
                        "before_lift_coefficient": (before.get("qois_512") or {}).get("lift_coefficient"),
                        "after_lift_coefficient": (after.get("qois_512") or {}).get("lift_coefficient"),
                        "before_pressure_difference": (before.get("qois_512") or {}).get("pressure_difference"),
                        "after_pressure_difference": (after.get("qois_512") or {}).get("pressure_difference"),
                        "field_relative_l2": None,
                        "scientific_accepted": False,
                    }
                )

        completed = sum(row["status"] == "complete" for row in results)
        report = {
            "status": "lbfgs_refinement_complete" if completed == len(SEEDS) else "lbfgs_refinement_partial",
            "model_family": "RBF-PIKAN only",
            "input_dataset_id": INPUT_DATASET_ID,
            "input_dataset_mount": str(dataset_root),
            "source_kernels": SOURCE_KERNELS,
            "seeds_by_source": SEEDS_BY_SOURCE,
            "seeds": list(SEEDS),
            "lbfgs_seconds_per_seed_cap": PAIR_SEED_SECONDS,
            "lbfgs_chunk_size": LBFGS_CHUNK_SIZE,
            "training_waves": wave_summaries,
            "source_mounts": {key: str(path.resolve()) for key, path in mounts.items()},
            "source_by_seed": source_by_seed,
            "official_module_sha256": module_hashes,
            "gpu_names": GPU_NAMES,
            "quota_cap_seconds": QUOTA_CAP_SECONDS,
            "soft_deadline_seconds": SOFT_DEADLINE_SECONDS,
            "watchdog_seconds_from_first_code_cell": WATCHDOG_SECONDS,
            "session_wall_seconds_from_first_code_cell": time.perf_counter() - SESSION_STARTED,
            "successful_refined_and_scored_seeds": completed,
            "expected_seeds": len(SEEDS),
            "results": results,
            "field_relative_l2": None,
            "field_reference_status": "missing_converged_FEM_field",
            "scientific_accepted": False,
        }
        write_json(SESSION / "dfg-current-cohort-refinement.json", report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "refined_and_scored": completed,
                    "session": str(SESSION),
                    "wall_seconds": report["session_wall_seconds_from_first_code_cell"],
                    "quota_cap_seconds": QUOTA_CAP_SECONDS,
                    "watchdog_seconds": WATCHDOG_SECONDS,
                    "scientific_accepted": False,
                },
                indent=2,
            )
        )
    finally:
        watchdog.cancel()


main()
