"""Standard-library runtime shipped with every experiment (no network calls).

This executes trusted research code; it is not an OS sandbox.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    # Windows scanners can briefly hold the destination after a read/write.
    # Keep atomic replace, bound retries, and propagate persistent permission errors.
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt==4: raise
            time.sleep(.05*(attempt+1))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def contained(root, relative):
    relative = str(relative)
    parts = PurePosixPath(relative)
    if '\\' in relative or ':' in relative or parts.is_absolute() or '..' in parts.parts:
        raise ValueError('Unsafe relative path: ' + relative)
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Path escapes root: ' + relative)
    return target


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    spec = read_json(bundle / 'bundle.json')
    identity = {k: v for k, v in spec.items() if k != 'bundle_id'}
    if spec.get('bundle_id') != digest(canonical(identity)):
        raise ValueError('Bundle identity hash mismatch')
    for name, expected in spec['files'].items():
        if digest(contained(bundle, name).read_bytes()) != expected:
            raise ValueError('File hash mismatch: ' + name)
    return spec


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def stop_process(process):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _tee_pipe(source, log_path, live_stream, prefix=''):
    """Persist child output and forward it to the Kaggle notebook in real time."""
    with Path(log_path).open('w', encoding='utf-8', errors='replace') as log:
        for line in iter(source.readline, ''):
            log.write(line)
            log.flush()
            live_stream.write(prefix + line)
            live_stream.flush()
    source.close()


def _gpu_monitor(output, gpu_slot, stop, status):
    columns = ['elapsed_sec', 'utilization_gpu_pct', 'memory_used_mib', 'memory_total_mib',
               'power_draw_w', 'temperature_c']
    path = Path(output) / 'gpu_telemetry.csv'
    started = time.monotonic()
    samples = 0
    failures = 0
    last_error = None
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        while True:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '-i', str(gpu_slot),
                     '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=8, check=True)
                parsed = next(csv.reader(result.stdout.splitlines()))
                row = {'elapsed_sec': round(time.monotonic() - started, 3)}
                for key, value in zip(columns[1:], parsed):
                    try:
                        row[key] = float(value.strip())
                    except ValueError:
                        row[key] = ''
                writer.writerow(row)
                handle.flush()
                samples += 1
                last_error = None
            except (OSError, subprocess.SubprocessError, StopIteration, csv.Error) as exc:
                failures += 1
                last_error = type(exc).__name__ + ': ' + str(exc)
            if stop.wait(10):
                break
    status.update({'available': samples > 0, 'samples': samples, 'failures': failures,
                   'reason': last_error if samples == 0 else None,
                   'interval_sec': 10, 'gpu_slot': str(gpu_slot)})


def _run_one_seed(bundle, output, spec, manifest, seed, gpu_slot):
    run = contained(output, f'seed-{seed}')
    summary_path = run / 'run_summary.json'
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        attempt = 1
        while (run / f'attempt-{attempt}').exists():
            attempt += 1
        previous = run / f'attempt-{attempt}'
        previous.mkdir()
        for item in list(run.iterdir()):
            if not (item.is_dir() and re.fullmatch(r'attempt-\d+', item.name)):
                item.replace(previous / item.name)
    snapshot = json.loads(json.dumps(manifest))
    snapshot['training']['seeds'] = [seed]
    snapshot['training']['seed'] = seed
    write_json(run / 'manifest_snapshot.json', snapshot)
    env = os.environ.copy()
    env.update(HARNESS_RUN_DIR=str(run), HARNESS_MANIFEST=str(run / 'manifest_snapshot.json'),
               HARNESS_SEED=str(seed), PYTHONHASHSEED=str(seed), PYTHONUNBUFFERED='1')
    if gpu_slot is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_slot)
        env['HARNESS_GPU_SLOT'] = str(gpu_slot)
    else:
        env.pop('HARNESS_GPU_SLOT', None)
    env['PYTHONPATH'] = str(bundle)
    command = [sys.executable, '-u', str(contained(bundle, spec['entrypoint']))]
    started = time.monotonic()
    slot_label = str(gpu_slot) if gpu_slot is not None else 'CPU'
    print(f'[campaign] seed={seed} assigned_gpu={slot_label}', flush=True)
    status, reason, code = 'failed', None, None
    telemetry_stop = threading.Event()
    telemetry_status = {'available': False, 'samples': 0, 'failures': 0,
                        'reason': 'No assigned GPU slot' if gpu_slot is None else None}
    telemetry_thread = None
    if gpu_slot is not None:
        telemetry_thread = threading.Thread(target=_gpu_monitor,
            args=(run, gpu_slot, telemetry_stop, telemetry_status), daemon=True)
        telemetry_thread.start()
    process = subprocess.Popen(command, cwd=bundle, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding='utf-8', errors='replace', bufsize=1,
                               start_new_session=os.name != 'nt')
    stdout_thread = threading.Thread(target=_tee_pipe,
        args=(process.stdout, run / 'stdout.log', sys.stdout, f'[seed {seed}] '), daemon=True)
    stderr_thread = threading.Thread(target=_tee_pipe,
        args=(process.stderr, run / 'stderr.log', sys.stderr, f'[seed {seed} stderr] '), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        code = process.wait(timeout=spec['timeout_sec'])
        status = 'completed' if code == 0 else 'failed'
    except subprocess.TimeoutExpired:
        stop_process(process)
        status, reason = 'timed_out', 'Per-seed time budget exhausted'
    except BaseException:
        stop_process(process)
        raise
    finally:
        telemetry_stop.set()
        if telemetry_thread is not None:
            telemetry_thread.join()
            write_json(run / 'gpu_telemetry_status.json', telemetry_status)
        stdout_thread.join()
        stderr_thread.join()
    metrics = {}
    try:
        metrics = read_json(run / 'metrics.json')
        if not isinstance(metrics, dict) or not all(finite_number(value) for value in metrics.values()):
            raise ValueError('Metrics must be finite numbers')
    except (OSError, ValueError, TypeError) as exc:
        metrics = {}
        if status == 'completed':
            status = 'invalid_artifacts'
        reason = str(exc)
    missing = [name for name in spec['required_artifacts'] if not contained(run, name).is_file()]
    if missing and status == 'completed':
        status, reason = 'invalid_artifacts', 'Missing artifacts: ' + ', '.join(missing)
    artifacts = {item.relative_to(run).as_posix(): digest(contained(run, item.relative_to(run).as_posix()).read_bytes())
                 for item in run.rglob('*') if item.is_file() and
                 not any(part.startswith('attempt-') for part in item.relative_to(run).parts)}
    summary = {
        'bundle_id': spec['bundle_id'], 'seed': seed, 'evidence_kind': spec['evidence_kind'],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'gpu_slot': str(gpu_slot) if gpu_slot is not None else None,
                        'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES')},
        'iteration': manifest['iteration'], 'artifact_hashes': artifacts,
        'result': {'status': status, 'returncode': code, 'reason': reason, 'metrics': metrics,
                   'elapsed_sec': time.monotonic() - started}}
    write_json(summary_path, summary)
    return {'seed': seed, 'status': status, 'summary': summary}


def run_bundle(bundle, output, retry_failed=False, max_parallel=None, gpu_slots=None):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    spec = verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'campaign.json'
    lock = output / '.running.lock'
    try:
        lock_handle = lock.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise ValueError('Campaign locked; inspect the process before removing .running.lock') from exc
    try:
        with lock_handle:
            lock_handle.write(str(os.getpid()))
        if state_path.exists():
            state = read_json(state_path)
            if state['bundle_id'] != spec['bundle_id']:
                raise ValueError('Output belongs to a different bundle')
        else:
            state = {'bundle_id': spec['bundle_id'], 'seeds': {}}
        manifest = read_json(bundle / 'manifest.json')
        workers = spec.get('max_parallel_seeds', 1) if max_parallel is None else max_parallel
        if type(workers) is not int or not 1 <= workers <= min(8, len(spec['seeds'])):
            raise ValueError('max_parallel must be an integer from 1 to min(8, seed count)')
        if workers > spec.get('max_parallel_seeds', 1):
            raise ValueError('max_parallel exceeds the reviewed bundle limit')
        if gpu_slots is None:
            if spec.get('enable_gpu') and workers > 1:
                raise ValueError('Multi-GPU execution requires explicit gpu_slots')
            slots = ['0'] if spec.get('enable_gpu') else [None] * workers
        else:
            slots = [str(slot) for slot in gpu_slots]
            if len(slots) != workers or len(set(slots)) != workers:
                raise ValueError('gpu_slots must contain one unique slot per worker')
        pending = []
        for seed in spec['seeds']:
            run = contained(output, f'seed-{seed}')
            summary_path = run / 'run_summary.json'
            if summary_path.exists():
                summary = read_json(summary_path)
                if summary.get('bundle_id') != spec['bundle_id']:
                    raise ValueError('Run belongs to a different bundle')
                if summary['result']['status'] == 'completed':
                    hashes = summary.get('artifact_hashes', {})
                    for name in set(spec['required_artifacts']) | {'manifest_snapshot.json'}:
                        artifact = contained(run, name)
                        if (name not in hashes or not artifact.is_file() or
                                digest(artifact.read_bytes()) != hashes[name]):
                            raise ValueError('Completed artifact missing/modified: ' + name)
                if summary['result']['status'] == 'completed' or not retry_failed:
                    state['seeds'][str(seed)] = summary['result']['status']
                    continue
            elif run.exists() and not retry_failed:
                raise ValueError('Interrupted run; use --retry-failed after inspecting its artifacts')
            state['seeds'][str(seed)] = 'queued'
            pending.append(seed)
        write_json(state_path, state)
        queue = iter(pending)

        if pending:
            active_slots = slots[:min(workers, len(pending))]
            with ThreadPoolExecutor(max_workers=len(active_slots)) as pool:
                futures = {}

                def launch(worker_index, slot, seed):
                    state['seeds'][str(seed)] = 'running'
                    write_json(state_path, state)
                    print(f'[campaign] worker={worker_index} started seed={seed} slot={slot}', flush=True)
                    future = pool.submit(_run_one_seed, bundle, output, spec, manifest, seed, slot)
                    futures[future] = (worker_index, slot, seed)

                # Assign the initial seeds in declared order, then refill the GPU that frees first.
                for index, slot in enumerate(active_slots):
                    seed = next(queue, None)
                    if seed is not None:
                        launch(index, slot, seed)
                while futures:
                    future = next(as_completed(tuple(futures)))
                    worker_index, slot, seed = futures.pop(future)
                    result = future.result()
                    state['seeds'][str(seed)] = result['status']
                    write_json(state_path, state)
                    print(f'[campaign] worker={worker_index} finished seed={seed} status={result["status"]}', flush=True)
                    next_seed = next(queue, None)
                    if next_seed is not None:
                        launch(worker_index, slot, next_seed)
        write_json(state_path, state)
        return state
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--out', required=True)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    state = run_bundle(args.bundle, args.out, args.retry_failed)
    print(json.dumps(state, indent=2))
    sys.exit(0 if all(s == 'completed' for s in state['seeds'].values()) else 1)
