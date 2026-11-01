"""Standard-library runtime shipped with every experiment (no network calls).

This executes trusted research code; it is not an OS sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


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


def run_bundle(bundle, output, retry_failed=False):
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
            run.mkdir(parents=True, exist_ok=True)
            # Failed attempts retain their files in numbered directories.
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
            state['seeds'][str(seed)] = 'running'
            write_json(state_path, state)
            env = os.environ.copy()
            env.update(HARNESS_RUN_DIR=str(run), HARNESS_MANIFEST=str(run / 'manifest_snapshot.json'),
                       HARNESS_SEED=str(seed), PYTHONHASHSEED=str(seed))
            # Training code is responsible for setting numpy/torch RNGs from HARNESS_SEED.
            env['PYTHONPATH'] = str(bundle)
            command = [sys.executable, str(contained(bundle, spec['entrypoint']))]
            started = time.monotonic()
            status, reason, code = 'failed', None, None
            with (run / 'stdout.log').open('w', encoding='utf-8') as stdout, \
                    (run / 'stderr.log').open('w', encoding='utf-8') as stderr:
                process = subprocess.Popen(command, cwd=bundle, env=env, stdout=stdout, stderr=stderr,
                                           start_new_session=os.name != 'nt')
                try:
                    code = process.wait(timeout=spec['timeout_sec'])
                    status = 'completed' if code == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    stop_process(process)
                    status, reason = 'timed_out', 'Per-seed time budget exhausted'
                except BaseException:
                    stop_process(process)
                    raise
            metrics = {}
            try:
                metrics = read_json(run / 'metrics.json')
                if not isinstance(metrics, dict) or not all(finite_number(v) for v in metrics.values()):
                    raise ValueError('Metrics must be finite numbers')
            except (OSError, ValueError, TypeError) as exc:
                metrics = {}
                if status == 'completed':
                    status = 'invalid_artifacts'
                reason = str(exc)
            missing = [name for name in spec['required_artifacts'] if not contained(run, name).is_file()]
            if missing and status == 'completed':
                status, reason = 'invalid_artifacts', 'Missing artifacts: ' + ', '.join(missing)
            artifacts = {p.relative_to(run).as_posix(): digest(contained(run, p.relative_to(run).as_posix()).read_bytes())
                         for p in run.rglob('*') if p.is_file() and
                         not any(part.startswith('attempt-') for part in p.relative_to(run).parts)}
            summary = {
                'bundle_id': spec['bundle_id'], 'seed': seed, 'evidence_kind': spec['evidence_kind'],
                'created_at': datetime.now(timezone.utc).isoformat(),
                'environment': {'python': sys.version, 'platform': platform.platform()},
                'iteration': manifest['iteration'], 'artifact_hashes': artifacts,
                'result': {'status': status, 'returncode': code, 'reason': reason, 'metrics': metrics,
                           'elapsed_sec': time.monotonic() - started}}
            write_json(summary_path, summary)
            state['seeds'][str(seed)] = status
            write_json(state_path, state)
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
