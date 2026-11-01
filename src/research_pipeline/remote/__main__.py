"""python -m src.research_pipeline.remote --help"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .bundle import generated_path, prepare, repository_path, verify_transport
from .evaluation import compare, evaluate
from .kaggle import fetch, require_approval, status, submit
from .runtime import run_bundle, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded PIKAN research infrastructure')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Inspect local dependencies without credentials or network')
    build = commands.add_parser('prepare', help='Create private Kaggle/Colab portable bundle offline')
    build.add_argument('--manifest', required=True)
    build.add_argument('--out', required=True)
    build.add_argument('--kernel', required=True)
    build.add_argument('--file', action='append', required=True, dest='files')
    build.add_argument('--timeout', type=int, default=300)
    build.add_argument('--evidence-kind', choices=['research', 'infrastructure'], default='research')
    build.add_argument('--gpu', action='store_true')
    comparison = commands.add_parser('compare')
    for name in ['baseline-bundle', 'baseline-runs', 'candidate-bundle', 'candidate-runs']:
        comparison.add_argument('--' + name, required=True)
    comparison.add_argument('--out')
    for name in ['run', 'submit', 'status', 'fetch', 'evaluate']:
        sub = commands.add_parser(name)
        sub.add_argument('--bundle', required=True)
        if name in ['run', 'fetch']:
            sub.add_argument('--out', required=True)
        if name in ['run', 'submit']:
            sub.add_argument('--approval')
        if name == 'run':
            sub.add_argument('--retry-failed', action='store_true')
        if name == 'evaluate':
            sub.add_argument('--runs', required=True)
            sub.add_argument('--out')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            from .diagnostics import doctor
            result = doctor()
        elif args.command == 'prepare':
            result = prepare(args.manifest, args.out, args.kernel, args.files, args.timeout,
                             args.evidence_kind, args.gpu)
        elif args.command == 'compare':
            result = compare(*[repository_path(path) for path in (
                args.baseline_bundle, args.baseline_runs, args.candidate_bundle, args.candidate_runs)])
            if args.out:
                destination = generated_path(args.out)
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_json(destination, result)
        else:
            bundle = repository_path(args.bundle)
            if args.command == 'run':
                spec = verify_transport(bundle)
                if spec['evidence_kind'] == 'research':
                    require_approval(spec, args.approval)
                result = run_bundle(bundle, generated_path(args.out), args.retry_failed)
            elif args.command == 'submit':
                result = submit(bundle, args.approval)
            elif args.command in ['status', 'fetch']:
                spec = verify_transport(bundle)
                print('Read-only remote access: https://www.kaggle.com/code/' + spec['kernel'], file=sys.stderr)
                result = status(bundle) if args.command == 'status' else fetch(bundle, args.out)
            else:
                result = evaluate(bundle, repository_path(args.runs))
                if args.out:
                    destination = generated_path(args.out)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    write_json(destination, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == 'run':
            return 0 if all(v == 'completed' for v in result['seeds'].values()) else 1
        if args.command in ['evaluate', 'fetch']:
            return 0 if result['execution_valid'] else 1
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
