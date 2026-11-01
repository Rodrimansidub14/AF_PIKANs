"""Provisional evidence checks. Never grants scientific acceptance."""
from __future__ import annotations

import statistics
from pathlib import Path

from src.research_pipeline.pipeline import evaluate_gates
from .runtime import contained, digest, finite_number, read_json, verify_bundle


def evaluate(bundle, runs):
    bundle, runs = Path(bundle), Path(runs)
    spec = verify_bundle(bundle)
    manifest = read_json(bundle / 'manifest.json')
    entries, failures, metrics = [], [], {}
    for seed in spec['seeds']:
        run = contained(runs, f'seed-{seed}')
        try:
            summary = read_json(run / 'run_summary.json')
            if summary.get('bundle_id') != spec['bundle_id'] or summary.get('seed') != seed:
                raise ValueError('Provenance mismatch')
            if summary.get('evidence_kind') != spec['evidence_kind']:
                raise ValueError('Evidence kind mismatch')
            if summary['result']['status'] != 'completed':
                raise ValueError('Run status: ' + summary['result']['status'])
            hashes = summary.get('artifact_hashes', {})
            for name in set(spec['required_artifacts']) | {'manifest_snapshot.json'}:
                target = contained(run, name)
                if name not in hashes or not target.is_file() or digest(target.read_bytes()) != hashes[name]:
                    raise ValueError('Missing/modified artifact: ' + name)
            snapshot = read_json(run / 'manifest_snapshot.json')
            expected = read_json(bundle / 'manifest.json')
            expected['training']['seeds'], expected['training']['seed'] = [seed], seed
            if snapshot != expected:
                raise ValueError('Manifest snapshot mismatch')
            values = read_json(run / 'metrics.json')
            if values != summary['result']['metrics'] or not all(finite_number(v) for v in values.values()):
                raise ValueError('Invalid/inconsistent metrics')
            gate = evaluate_gates(manifest, summary)
            entries.append({'seed': seed, 'gate': gate})
            for name, value in values.items():
                metrics.setdefault(name, []).append(value)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({'seed': seed, 'reason': str(exc)})
    execution_valid = not failures and len(entries) == len(spec['seeds'])
    thresholds_pass = execution_valid and all(e['gate']['technical_pass'] for e in entries)
    pending = ['Human interpretation and scientific_claim gate']
    if spec['evidence_kind'] != 'research':
        pending.append('Infrastructure probe is not research evidence')
    gates = manifest.get('acceptance_gates', {})
    # Multiple RNG seeds do not establish multiple independent physical configurations.
    if gates.get('min_independent_configurations', 1) > 1:
        pending.append('Cross-configuration confirmation not supplied by a single bundle')
    if gates.get('max_cost_multiplier_vs_baseline') is not None:
        pending.append('Matched baseline cost comparison must be reviewed separately')
    if not thresholds_pass:
        pending.append('Per-run numerical thresholds have not all passed')
    return {
        'bundle_id': spec['bundle_id'], 'evidence_kind': spec['evidence_kind'],
        'execution_valid': execution_valid, 'thresholds_pass': thresholds_pass,
        'eligible_for_review': thresholds_pass and spec['evidence_kind'] == 'research',
        'scientific_accepted': False, 'requires_human_review': True,
        'pending_checks': pending, 'failures': failures, 'runs': entries,
        'metrics': {name: {'n': len(values), 'mean': statistics.mean(values),
                           'std': statistics.stdev(values) if len(values) > 1 else None,
                           'min': min(values), 'max': max(values)}
                    for name, values in metrics.items()}}


def compare(baseline_bundle, baseline_runs, candidate_bundle, candidate_runs):
    """Paired differences, with comparability limitations explicitly reported."""
    baseline = evaluate(baseline_bundle, baseline_runs)
    candidate = evaluate(candidate_bundle, candidate_runs)
    if not baseline['execution_valid'] or not candidate['execution_valid']:
        raise ValueError('Both campaigns must have valid, complete artifacts')
    left, right = verify_bundle(baseline_bundle), verify_bundle(candidate_bundle)
    lm, rm = read_json(Path(baseline_bundle) / 'manifest.json'), read_json(Path(candidate_bundle) / 'manifest.json')
    if set(left['seeds']) != set(right['seeds']):
        raise ValueError('Comparison requires matched seeds')
    if lm['problem'] != rm['problem'] or lm['iteration']['problem_id'] != rm['iteration']['problem_id']:
        raise ValueError('Comparison requires the same physical problem and reference')
    if lm['training'] != rm['training'] or left['timeout_sec'] != right['timeout_sec']:
        raise ValueError('Comparison requires matched training configuration and time budgets')
    metric_specs = lm['acceptance_gates']['metrics']
    if metric_specs != rm['acceptance_gates']['metrics']:
        raise ValueError('Comparison requires a fixed evaluation contract')
    pairs = [(read_json(Path(baseline_runs) / f'seed-{seed}/run_summary.json'),
              read_json(Path(candidate_runs) / f'seed-{seed}/run_summary.json')) for seed in left['seeds']]
    output = {}
    for name, metric_spec in metric_specs.items():
        if any(name not in s['result']['metrics'] for pair in pairs for s in pair):
            continue
        direction = metric_spec.get('direction')
        if direction not in ('minimize', 'maximize'):
            raise ValueError('Metric direction must be minimize or maximize: ' + name)
        sign = 1 if direction == 'minimize' else -1
        differences = [sign * (a['result']['metrics'][name] - b['result']['metrics'][name]) for a, b in pairs]
        output[name] = {'baseline_mean': baseline['metrics'][name]['mean'],
                        'candidate_mean': candidate['metrics'][name]['mean'],
                        'mean_paired_improvement': statistics.mean(differences),
                        'std_paired_improvement': statistics.stdev(differences) if len(differences) > 1 else None,
                        'improved_seeds': sum(d > 0 for d in differences), 'n': len(differences)}
    base_time = sum(a['result']['elapsed_sec'] for a, _ in pairs)
    candidate_time = sum(b['result']['elapsed_sec'] for _, b in pairs)
    ratio = candidate_time / base_time if base_time > 0 else None
    # Python/platform alone cannot prove matching GPUs or contention. Keep cost provisional.
    return {'baseline_bundle_id': left['bundle_id'], 'candidate_bundle_id': right['bundle_id'],
            'metrics': output, 'observed_runtime_ratio': ratio,
            'cost_limit': rm['acceptance_gates'].get('max_cost_multiplier_vs_baseline'),
            'cost_gate_pass': None,
            'eligible_for_review': baseline['eligible_for_review'] and candidate['eligible_for_review'],
            'scientific_accepted': False, 'requires_human_review': True,
            'limitations': ['Runtime ratio requires matched hardware/load before scientific interpretation.',
                            'Seeds are repeated trials, not independent physical configurations.',
                            'No statistical significance or SOTA claim is inferred.']}
