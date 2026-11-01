from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RemoteResearchTests(unittest.TestCase):
    def setUp(self):
        # All test writes stay inside this repository.
        base = ROOT / 'var' / 'agent_runs' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        import yaml
        self.manifest = yaml.safe_load((ROOT / 'src/harness_loop/experiment_manifest.yaml').read_text())
        self.manifest['training']['seeds'] = [11, 17]
        self.manifest['artifacts']['expected'] = ['metrics.json', 'decision_note.md']
        self.manifest['acceptance_gates']['metrics'] = {
            'error': {'required': True, 'direction': 'minimize', 'threshold': 0.1}}
        self.manifest['model']['implementation'] = 'src/research_pipeline/examples/infra_smoke.py'
        self.manifest_path = self.work / 'manifest.json'
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'src.research_pipeline.remote', *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)

    def prepare(self, destination='bundle', **extra):
        args = ['prepare', '--manifest', self.manifest_path, '--out', self.work / destination,
                          '--kernel', 'researcher/pikan-smoke', '--timeout', extra.get('timeout', 10),
                          '--evidence-kind', 'infrastructure', '--file',
                          'src/research_pipeline/examples/infra_smoke.py']
        if extra.get('gpu'):
            args.append('--gpu')
        result = self.cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return self.work / destination

    def test_cli_is_available(self):
        result = self.cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('prepare', result.stdout)

    def test_atomic_json_replace_retries_transient_permission_error(self):
        from src.research_pipeline.remote.runtime import write_json
        original=Path.replace
        calls=[]
        def transient(path,target):
            calls.append(1)
            if len(calls)<3: raise PermissionError('transient scanner lock')
            return original(path,target)
        with patch.object(Path,'replace',transient), patch('src.research_pipeline.remote.runtime.time.sleep'):
            write_json(self.work/'state.json',{'ok':True})
        self.assertEqual(len(calls),3)
        self.assertEqual(json.loads((self.work/'state.json').read_text()),{'ok':True})

    def test_pinned_dependencies_are_bound_to_private_notebook(self):
        from src.research_pipeline.remote.bundle import verify_transport
        self.manifest['remote']={'enable_internet':True,'runtime_dependencies':['pina-mathlab==0.3.2.post2609']}
        self.save_manifest()
        bundle=self.prepare()
        spec=verify_transport(bundle)
        self.assertEqual(spec['runtime_dependencies'],['pina-mathlab==0.3.2.post2609'])
        nb=json.loads((bundle/'experiment.ipynb').read_text())
        code=''.join(next(cell for cell in nb['cells'] if cell['id']=='dependencies')['source'])
        compile(code,'<notebook>','exec')
        self.assertIn('pip',code)
        metadata=json.loads((bundle/'kernel-metadata.json').read_text())
        self.assertTrue(metadata['enable_internet'])
        self.assertTrue(metadata['is_private'])

    def test_development_notebook_exposes_scientific_stages_and_t4x2_metadata(self):
        self.manifest['remote'] = {
            'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4',
            'max_parallel_seeds': 2,
            'runtime_dependencies': ['pina-mathlab==0.3.2.post2609'],
        }
        self.save_manifest()
        bundle = self.prepare(gpu=True)
        metadata = json.loads((bundle / 'kernel-metadata.json').read_text())
        self.assertEqual(metadata['machine_shape'], 'NvidiaTeslaT4')
        notebook = json.loads((bundle / 'experiment.ipynb').read_text())
        self.assertGreaterEqual(len(notebook['cells']), 12)
        self.assertEqual(notebook['metadata']['research']['role'], 'exploration')
        markdown = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                              if cell['cell_type'] == 'markdown')
        self.assertIn('Entrenamiento', markdown)
        self.assertIn('W&B', markdown)
        self.assertIn('Plotly', markdown)
        self.assertIn('PyVista', markdown)
        self.assertIn('Panel', markdown)
        cell_ids = {cell['id'] for cell in notebook['cells']}
        self.assertTrue({'gpu-preflight', 'sampling-signals', 'parallel-run',
                         'quantitative-signals', 'visualizations', 'closure'} <= cell_ids)
        gpu_preflight = ''.join(next(cell['source'] for cell in notebook['cells']
                                     if cell['id'] == 'gpu-preflight'))
        parallel_run = ''.join(next(cell['source'] for cell in notebook['cells']
                                    if cell['id'] == 'parallel-run'))
        self.assertIn('visible_gpus < 2', gpu_preflight)
        self.assertIn('max_parallel=MAX_PARALLEL_SEEDS', parallel_run)

    def test_gpu_slot_fallback_is_explicit_and_template_is_bundle_bound(self):
        import hashlib

        self.manifest['remote'] = {
            'machine_shape': 'NvidiaTeslaT4',
            'max_parallel_seeds': 2,
            'allow_gpu_count_fallback': True,
        }
        self.save_manifest()
        bundle = self.prepare(gpu=True)
        from src.research_pipeline.remote.bundle import verify_transport
        spec = verify_transport(bundle)
        self.assertTrue(spec['allow_gpu_count_fallback'])
        template = ROOT / 'src/research_pipeline/remote/notebook.py'
        self.assertEqual(spec['notebook_template_sha256'],
                         hashlib.sha256(template.read_bytes()).hexdigest())
        notebook = json.loads((bundle / 'experiment.ipynb').read_text())
        gpu_preflight = ''.join(next(cell['source'] for cell in notebook['cells']
                                     if cell['id'] == 'gpu-preflight'))
        parallel_run = ''.join(next(cell['source'] for cell in notebook['cells']
                                    if cell['id'] == 'parallel-run'))
        self.assertIn('min(SPEC[\'max_parallel_seeds\'], visible_gpus)', gpu_preflight)
        self.assertIn('max_parallel=len(GPU_SLOTS)', parallel_run)

    def test_mixed_heat_allen_cahn_sampling_handles_problem_specific_pool_keys(self):
        self.manifest['problem']['sequence'] = ['heat', 'allen_cahn']
        self.manifest['iteration']['experiment_id'] = 'EXP-0003'
        self.save_manifest()
        bundle = self.prepare()
        notebook = json.loads((bundle / 'experiment.ipynb').read_text())
        sampling = ''.join(next(cell['source'] for cell in notebook['cells']
                                if cell['id'] == 'sampling-signals'))

        compile(sampling, '<sampling-signals>', 'exec')
        self.assertIn("len(data['boundary_probe']) if 'boundary_probe' in data", sampling)
        self.assertIn("len(data['left']) + len(data['right'])", sampling)
        self.assertIn("len(data['boundary_probe'])", sampling)
        self.manifest['training']['interior_points'] = 8
        self.manifest['training']['boundary_points'] = 4
        self.manifest['training']['heldout_points'] = 16
        self.save_manifest()
        (bundle / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')
        from src.sciml.sampling import pools
        import numpy as np
        import pandas as pd
        import plotly.express as px
        from scipy.spatial import cKDTree
        from scipy.stats import qmc
        analysis = self.work / 'analysis'
        analysis.mkdir()
        executable_sampling = sampling.replace('from IPython.display import display\n', '')
        exec(executable_sampling, {
            'BUNDLE_DIR': bundle, 'RUNS_DIR': self.work / 'runs',
            'ANALYSIS_DIR': analysis, 'display': lambda _value: None,
            'pools': pools, 'np': np, 'pd': pd, 'px': px,
            'cKDTree': cKDTree, 'qmc': qmc,
        })
        table = pd.read_csv(analysis / 'sampling-signals.csv')
        self.assertEqual(set(table['problem']), {'heat', 'allen_cahn'})
        self.assertEqual(len(table), 4)

    def test_dfg_notebook_uses_problem_specific_tables_and_residual_visualization(self):
        self.manifest['iteration']['problem_id'] = 'dfg_2d_1_steady_re20'
        self.manifest['problem']['sequence'] = ['dfg_2d_1_steady_re20']
        self.manifest['parameter_decisions_markdown'] = '# DFG parameter decisions'
        self.save_manifest()
        bundle = self.prepare()
        notebook = json.loads((bundle / 'experiment.ipynb').read_text())
        cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
        cell_order = [cell['id'] for cell in notebook['cells']]

        for cell in ('sampling-signals', 'quantitative-signals', 'visualizations', 'field-maps'):
            compile(cells[cell], f'<{cell}>', 'exec')
        self.assertIn('dfg_training_pools.npz', cells['sampling-signals'])
        self.assertGreater(cell_order.index('sampling-signals'), cell_order.index('parallel-run'))
        self.assertIn('dfg_training_losses.html', cells['visualizations'])
        self.assertIn('pyvista as pv', cells['field-maps'])
        self.assertIn('dfg_heldout_residuals.npz', cells['field-maps'])
        self.assertNotIn('field_figures(RUNS_DIR)', cells['field-maps'])

    def test_notebook_transport_is_byte_deterministic_across_clock_ticks(self):
        from src.research_pipeline.remote.bundle import verify_transport
        bundle = self.prepare()
        original = (bundle / 'experiment.ipynb').read_bytes()
        time.sleep(1.1)
        verify_transport(bundle)
        self.assertEqual((bundle / 'experiment.ipynb').read_bytes(), original)

    def test_dependency_options_and_unpinned_versions_rejected(self):
        from src.research_pipeline.remote.bundle import prepare
        for dep in ('--index-url=https://example.com','torch','torch>=2','https://example.com/a.whl'):
            self.manifest['remote']={'enable_internet':True,'runtime_dependencies':[dep]}
            self.save_manifest()
            with self.assertRaises(ValueError):
                prepare(self.manifest_path,self.work/'bad','researcher/test',
                        ['src/research_pipeline/examples/infra_smoke.py'],evidence_kind='infrastructure')

    def test_bundle_rejects_omitted_local_evaluator_imports(self):
        from src.research_pipeline.remote.bundle import prepare
        source_root = ROOT / 'evals'
        omitted = 'evals/problems/burgers_pinnacle_1d/reference.py'
        files = [
            str(path.relative_to(ROOT)).replace('\\', '/')
            for path in source_root.rglob('*.py')
            if str(path.relative_to(ROOT)).replace('\\', '/') != omitted
        ]
        self.manifest['model']['implementation'] = 'evals/checkpoint_validation.py'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'evals/problems/burgers_pinnacle_1d/reference.py'):
            prepare(self.manifest_path, self.work/'missing-evaluator-import',
                    'researcher/missing-evaluator-import', files, evidence_kind='infrastructure')

    def test_package_is_portable_private_and_content_addressed(self):
        bundle = self.prepare()
        other = self.prepare('other')
        spec = json.loads((bundle / 'bundle.json').read_text())
        self.assertEqual(spec['bundle_id'], json.loads((other / 'bundle.json').read_text())['bundle_id'])
        metadata = json.loads((bundle / 'kernel-metadata.json').read_text())
        self.assertTrue(metadata['is_private'])
        self.assertFalse(metadata['enable_internet'])
        nb = json.loads((bundle / 'experiment.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']), '<'+cell['id']+'>', 'exec')
        self.assertNotIn('PIKAN-Vault', json.dumps(spec))

    def test_local_round_trip_and_resume(self):
        bundle = self.prepare()
        out = self.work / 'runs'
        result = self.cli('run', '--bundle', bundle, '--out', out)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        summaries = list(out.glob('seed-*/run_summary.json'))
        self.assertEqual(len(summaries), 2)
        before = {p: p.read_bytes() for p in summaries}
        self.assertEqual(self.cli('run', '--bundle', bundle, '--out', out).returncode, 0)
        self.assertEqual(before, {p: p.read_bytes() for p in summaries})
        report = self.cli('evaluate', '--bundle', bundle, '--runs', out)
        self.assertEqual(report.returncode, 0, report.stderr)
        parsed = json.loads(report.stdout)
        self.assertFalse(parsed['scientific_accepted'])
        self.assertFalse(parsed['eligible_for_review'])
        self.assertTrue(parsed['execution_valid'])

    def test_parallel_seed_runtime_assigns_distinct_gpu_slots_and_preserves_records(self):
        from src.research_pipeline.remote.runtime import read_json, run_bundle
        self.manifest['remote'] = {'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
                                   'max_parallel_seeds': 2}
        self.save_manifest()
        bundle = self.runtime_bundle('''import json, os, pathlib, time
run = pathlib.Path(os.environ['HARNESS_RUN_DIR'])
(run / 'metrics.json').write_text(json.dumps({'error': 0.01}))
(run / 'decision_note.md').write_text('exploratory')
(run / 'gpu_slot.txt').write_text(os.environ['HARNESS_GPU_SLOT'])
time.sleep(0.1)
''', gpu=True)
        runs = self.work / 'runs-parallel'
        state = run_bundle(bundle, runs, max_parallel=2, gpu_slots=['0', '1'])
        self.assertEqual(set(state['seeds'].values()), {'completed'})
        slots = {read_json(path)['seed']: read_json(path)['environment']['gpu_slot']
                 for path in runs.glob('seed-*/run_summary.json')}
        self.assertEqual(slots, {11: '0', 17: '1'})

    def test_artifact_hashing_ignores_wandb_symlink_outside_run_directory(self):
        import os
        from src.research_pipeline.remote.runtime import read_json, run_bundle

        external_log = self.work / 'wandb-debug-core.log'
        external_log.write_text('W&B debugger output', encoding='utf-8')
        probe_dir = self.work / 'symlink-capability-probe'
        probe_dir.mkdir()
        try:
            os.symlink(external_log, probe_dir / 'probe.log')
        except OSError as exc:
            self.skipTest(f'Cannot create a symlink on this host: {exc}')

        code = f'''import json, os, pathlib
run = pathlib.Path(os.environ['HARNESS_RUN_DIR'])
(run / 'metrics.json').write_text(json.dumps({{'error': 0.01}}))
(run / 'decision_note.md').write_text('completed')
(run / 'wandb').mkdir()
os.symlink({str(external_log)!r}, run / 'wandb' / 'debug-core.log')
'''
        bundle = self.runtime_bundle(code)
        runs = self.work / 'runs-with-wandb-symlink'
        state = run_bundle(bundle, runs)

        self.assertEqual(set(state['seeds'].values()), {'completed'})
        summary = read_json(runs / 'seed-11/run_summary.json')
        self.assertEqual(summary['result']['status'], 'completed')
        self.assertNotIn('wandb/debug-core.log', summary['artifact_hashes'])

    def test_modified_bundle_is_rejected(self):
        bundle = self.prepare()
        path = bundle / 'src/research_pipeline/examples/infra_smoke.py'
        path.write_text('print("tampered")')
        result = self.cli('run', '--bundle', bundle, '--out', self.work / 'runs')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('hash', result.stderr.lower())

    def test_external_output_and_duplicate_seeds_are_rejected(self):
        result = self.cli('prepare', '--manifest', self.manifest_path, '--out', ROOT.parent / 'forbidden',
                          '--kernel', 'researcher/test', '--file', 'src/research_pipeline/examples/infra_smoke.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('repository', result.stderr.lower())
        self.manifest['training']['seeds'] = [11, 11]
        self.save_manifest()
        result = self.cli('prepare', '--manifest', self.manifest_path, '--out', self.work / 'bad',
                          '--kernel', 'researcher/test', '--file', 'src/research_pipeline/examples/infra_smoke.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('seeds', result.stderr.lower())

    def test_submit_is_blocked_by_repository_policy_before_network(self):
        from src.research_pipeline.remote.kaggle import require_remote_policy
        policy_root = self.work / 'policy-test'
        (policy_root / 'context').mkdir(parents=True)
        (policy_root / 'context/agent-policy.toml').write_text('[scope]\nexternal_write_access="forbidden"\n')
        with patch('src.research_pipeline.remote.kaggle.ROOT', policy_root):
            with self.assertRaisesRegex(ValueError, 'external_write_access'):
                require_remote_policy()

    def test_kaggle_permission_does_not_grant_campaign_approval(self):
        bundle = self.prepare()
        result = self.cli('submit', '--bundle', bundle)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Human approval record required', result.stderr)
        self.assertFalse((bundle / 'submission.json').exists())

    def test_kaggle_output_download_accepts_extended_timeout(self):
        from types import SimpleNamespace
        from src.research_pipeline.remote import kaggle

        with patch.object(kaggle, 'executable_path', return_value=ROOT / 'var/agent_runs/remote-env/Scripts/python.exe'), \
             patch.object(kaggle.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='downloaded')) as run_command:
            output = kaggle.invoke(['kernels', 'output'], timeout_sec=1200)

        self.assertEqual(output, 'downloaded')
        self.assertEqual(run_command.call_args.kwargs['timeout'], 1200)

    def test_notebook_payload_executes_without_repository_imports(self):
        bundle = self.prepare()
        nb = json.loads((bundle / 'experiment.ipynb').read_text())
        script = self.work / 'notebook_cell.py'
        restore = next(cell for cell in nb['cells'] if cell['id']=='bundle-restore')
        script.write_text('import os\nos.environ["KAGGLE_WORKING_DIR"]='+repr(str(self.work/'kernel-working'))+
                          '\n'+''.join(restore['source']), encoding='utf-8')
        result = subprocess.run([sys.executable, str(script)], cwd=self.work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']), '<'+cell['id']+'>', 'exec')
        restored = next((self.work/'kernel-working').glob('pikan-*/bundle'))
        postprocess_script = self.work / 'import_remote_postprocess.py'
        postprocess_script.write_text(
            'import sys\nfrom pathlib import Path\n'
            'sys.path.insert(0,'+repr(str(restored))+')\n'
            'from src.research_pipeline.remote.postprocess import create_results_archive\n'
            'assert callable(create_results_archive)\n', encoding='utf-8')
        imported = subprocess.run([sys.executable, str(postprocess_script)], cwd=self.work,
                                  capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        runs = self.work/'runs-from-restored-notebook'
        run_script = self.work/'run_restored_bundle.py'
        run_script.write_text(
            'import sys\nfrom pathlib import Path\n'
            'bundle=Path('+repr(str(restored))+')\n'
            'sys.path.insert(0,str(bundle))\nfrom runtime import run_bundle\n'
            'print(run_bundle(bundle,Path('+repr(str(runs))+')))\n', encoding='utf-8')
        execution = subprocess.run([sys.executable, str(run_script)], cwd=self.work,
                                   capture_output=True, text=True)
        self.assertEqual(execution.returncode, 0, execution.stderr + execution.stdout)
        self.assertEqual(len(list(runs.glob('seed-*/run_summary.json'))), 2)

    def runtime_bundle(self, code, gpu=False, timeout=1):
        from src.research_pipeline.remote.runtime import canonical, digest, write_json
        bundle = self.prepare(timeout=timeout, gpu=gpu)
        spec = json.loads((bundle / 'bundle.json').read_text())
        source = bundle / spec['entrypoint']
        source.write_text(code, encoding='utf-8')
        spec['files'][spec['entrypoint']] = digest(source.read_bytes())
        del spec['bundle_id']
        spec['bundle_id'] = digest(canonical(spec))
        write_json(bundle / 'bundle.json', spec)
        return bundle

    def test_timeout_persists_failed_runs_and_does_not_retry_implicitly(self):
        from src.research_pipeline.remote.runtime import run_bundle
        bundle = self.runtime_bundle('import time; time.sleep(20)')
        runs = self.work / 'runs'
        result = run_bundle(bundle, runs)
        self.assertEqual(set(result['seeds'].values()), {'timed_out'})
        summary = runs / 'seed-11/run_summary.json'
        before = summary.read_bytes()
        run_bundle(bundle, runs)
        self.assertEqual(summary.read_bytes(), before)

    def test_missing_and_nonfinite_artifacts_are_failures(self):
        from src.research_pipeline.remote.runtime import run_bundle
        bundle = self.runtime_bundle('import os, pathlib; '
            'pathlib.Path(os.environ["HARNESS_RUN_DIR"], "metrics.json").write_text(\'{"error": NaN}\')', timeout=30)
        result = run_bundle(bundle, self.work / 'runs')
        self.assertEqual(set(result['seeds'].values()), {'invalid_artifacts'})

    def test_evaluation_and_resume_detect_modified_completed_artifacts(self):
        from src.research_pipeline.remote.runtime import run_bundle
        bundle = self.prepare()
        runs = self.work / 'runs'
        run_bundle(bundle, runs)
        (runs / 'seed-11/metrics.json').write_text('{"error": 0.099}')
        result = self.cli('evaluate', '--bundle', bundle, '--runs', runs)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)['execution_valid'])
        with self.assertRaisesRegex(ValueError, 'artifact'):
            run_bundle(bundle, runs)

    def test_kaggle_duplicate_submission_is_not_retried(self):
        from src.research_pipeline.remote.kaggle import submit
        bundle = self.prepare()
        approval = self.work / 'approval.json'
        spec = json.loads((bundle / 'bundle.json').read_text())
        approval.write_text(json.dumps({'bundle_id': spec['bundle_id'], 'approved_by': 'Test reviewer',
            'scope': 'approved', 'hypothesis': 'approved', 'code_surface': 'approved', 'remote_execution': True}))
        with patch('src.research_pipeline.remote.kaggle.require_remote_policy'), \
             patch('src.research_pipeline.remote.kaggle.invoke', side_effect=ValueError('uncertain response')):
            with self.assertRaisesRegex(ValueError, 'uncertain'):
                submit(bundle, approval)
        self.assertEqual(json.loads((bundle / 'submission.json').read_text())['status'], 'submission_unknown')
        with patch('src.research_pipeline.remote.kaggle.require_remote_policy'):
            with self.assertRaisesRegex(ValueError, 'already recorded'):
                submit(bundle, approval)

    def test_kaggle_download_verifies_identity_and_artifacts(self):
        import shutil
        from src.research_pipeline.remote.runtime import run_bundle
        from src.research_pipeline.remote.kaggle import fetch
        bundle = self.prepare()
        runs = self.work / 'runs'
        run_bundle(bundle, runs)
        download = self.work / 'download'
        def fake_download(arguments, **kwargs):
            self.assertEqual(arguments[:3], ['kernels', 'output', 'researcher/pikan-smoke'])
            shutil.copytree(runs, Path(arguments[-1]) / 'pikan-test/runs')
            return ''
        with patch('src.research_pipeline.remote.kaggle.invoke', side_effect=fake_download):
            result = fetch(bundle, download)
        self.assertTrue(result['execution_valid'])
        self.assertTrue(Path(result['report']).is_file())

    def test_kaggle_download_can_validate_complete_outputs_after_cli_error(self):
        import shutil
        from src.research_pipeline.remote.runtime import run_bundle
        from src.research_pipeline.remote.kaggle import fetch
        bundle = self.prepare()
        runs = self.work / 'runs'
        run_bundle(bundle, runs)
        download = self.work / 'download-after-error'
        def partially_failed_download(arguments, **kwargs):
            shutil.copytree(runs, Path(arguments[-1]) / 'pikan-test/runs')
            raise ValueError('Kaggle CLI failed (exit 1)')
        with patch('src.research_pipeline.remote.kaggle.invoke', side_effect=partially_failed_download):
            result = fetch(bundle, download)
        self.assertTrue(result['execution_valid'])
        self.assertIn('transport_warning', result)
        self.assertTrue(Path(result['report']).is_file())

    def test_comparison_reports_paired_metrics_but_never_accepts_claim(self):
        bundle = self.prepare()
        runs = self.work / 'runs'
        self.assertEqual(self.cli('run', '--bundle', bundle, '--out', runs).returncode, 0)
        result = self.cli('compare', '--baseline-bundle', bundle, '--baseline-runs', runs,
                          '--candidate-bundle', bundle, '--candidate-runs', runs)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['metrics']['error']['mean_paired_improvement'], 0)
        self.assertFalse(report['scientific_accepted'])
        self.assertFalse(report['eligible_for_review'])

    def test_transport_cannot_be_changed_by_refreshing_hash_index(self):
        from src.research_pipeline.remote.bundle import verify_transport
        from src.research_pipeline.remote.runtime import digest
        bundle = self.prepare()
        metadata = json.loads((bundle / 'kernel-metadata.json').read_text())
        metadata['is_private'] = False
        (bundle / 'kernel-metadata.json').write_text(json.dumps(metadata))
        index = {name: digest((bundle / name).read_bytes())
                 for name in ['experiment.ipynb', 'kernel-metadata.json']}
        (bundle / 'transport-hashes.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'Transport'):
            verify_transport(bundle)

    def test_notebook_code_is_bound_to_the_bundle_even_with_new_transport_hashes(self):
        from src.research_pipeline.remote.bundle import verify_transport
        from src.research_pipeline.remote.runtime import digest
        bundle = self.prepare()
        notebook = json.loads((bundle / 'experiment.ipynb').read_text())
        notebook['cells'][1]['source'].append('\nprint("unreviewed code")\n')
        (bundle / 'experiment.ipynb').write_text(json.dumps(notebook))
        index = {name: digest((bundle / name).read_bytes())
                 for name in ['experiment.ipynb', 'kernel-metadata.json']}
        (bundle / 'transport-hashes.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'Transport notebook'):
            verify_transport(bundle)
        (bundle / 'transport-hashes.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Transport'):
            verify_transport(bundle)

    def test_retry_does_not_reuse_nested_artifacts_from_failed_attempt(self):
        from src.research_pipeline.remote.runtime import run_bundle
        self.manifest['artifacts']['expected'].append('checkpoints/model.pt')
        self.save_manifest()
        code = '''import json, os, pathlib, sys
run = pathlib.Path(os.environ['HARNESS_RUN_DIR'])
if not (run / 'attempt-1').exists():
    (run / 'checkpoints').mkdir()
    (run / 'checkpoints/model.pt').write_text('stale')
    sys.exit(1)
(run / 'metrics.json').write_text(json.dumps({'error': 0.0}))
(run / 'decision_note.md').write_text('retry omits checkpoint')
'''
        bundle = self.runtime_bundle(code)
        runs = self.work / 'runs'
        run_bundle(bundle, runs)
        result = run_bundle(bundle, runs, retry_failed=True)
        self.assertEqual(set(result['seeds'].values()), {'invalid_artifacts'})
        self.assertTrue((runs / 'seed-11/attempt-1/checkpoints/model.pt').is_file())


class ScientificGateTests(unittest.TestCase):
    def test_presence_does_not_accept_science(self):
        from src.research_pipeline.pipeline import evaluate_gates
        result = evaluate_gates({'acceptance_gates': {'metrics': {'error': {'required': True}}}},
                                {'result': {'status': 'completed', 'metrics': {'error': 99}}})
        self.assertFalse(result['accepted'])
        self.assertTrue(result['requires_human_review'])

    def test_threshold_and_nonfinite_metrics(self):
        from src.research_pipeline.pipeline import evaluate_gates
        manifest = {'acceptance_gates': {'metrics': {
            'error': {'required': True, 'direction': 'minimize', 'threshold': 0.1}}}}
        for value in [0.2, float('nan'), float('inf'), True, '0.01']:
            result = evaluate_gates(manifest, {'result': {'status': 'completed', 'metrics': {'error': value}}})
            self.assertFalse(result['checks'][0]['passed'], str(value))


if __name__ == '__main__':
    unittest.main()
