from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
PACKAGE = HERE / "draft_bundle"
MANIFEST_PATH = HERE / "manifest.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def copy_sources() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paths = [
        "src/research_pipeline/sciml/__init__.py",
        "src/research_pipeline/sciml/train.py",
        "src/research_pipeline/sciml/sampling.py",
        "src/research_pipeline/sciml/model.py",
        "src/research_pipeline/remote/__init__.py",
        "src/research_pipeline/remote/runtime.py",
        "src/research_pipeline/remote/postprocess.py",
        "evals/__init__.py",
        "evals/checkpoint_validation.py",
        "evals/problems/__init__.py",
        "evals/problems/_grids.py",
        "evals/problems/burgers_1d/__init__.py",
        "evals/problems/burgers_1d/reference.py",
        "evals/problems/burgers_pinnacle_1d/__init__.py",
        "evals/problems/burgers_pinnacle_1d/reference.py",
        "evals/problems/heat_1d/__init__.py",
        "evals/problems/heat_1d/reference.py",
    ]
    for relative in paths:
        source = PROJECT / relative
        target = PACKAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # The notebook's bootstrap imports ``runtime`` from the extracted bundle root.
    # Keep the canonical package copy too, but mirror this module at that import path.
    shutil.copyfile(PROJECT / "src/research_pipeline/remote/runtime.py", PACKAGE / "runtime.py")
    # This standalone adapter is imported both as a package module by the
    # entrypoint and as a top-level module by the CUDA scorer.
    paired_training = HERE / "burgers_paired_training.py"
    nested_training = PACKAGE / "src/research_pipeline/remote/burgers_paired_training.py"
    nested_training.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paired_training, nested_training)
    shutil.copyfile(paired_training, PACKAGE / "burgers_paired_training.py")
    adapter = PROJECT / "var/agent_runs/baseline-competitive-2026-09-27/adapters/burgers_pinnacle.py"
    target_adapter = PACKAGE / "var/agent_runs/baseline-competitive-2026-09-27/adapters/burgers_pinnacle.py"
    target_adapter.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(adapter, target_adapter)
    entrypoint = PACKAGE / manifest["model"]["implementation"]
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    # The wrapper is maintained in this review directory and packaged as an immutable copy.
    shutil.copyfile(HERE / "burgers_pinnacle_entrypoint.py", entrypoint)
    shutil.copyfile(HERE / "burgers_cuda_score.py", PACKAGE / "burgers_cuda_score.py")
    shutil.copyfile(MANIFEST_PATH, PACKAGE / "manifest.json")


def set_source(cell, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def build_notebook(manifest, spec, content):
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    renderer = importlib.import_module("src.research_pipeline.remote.notebook")
    notebook = renderer.render_notebook(spec, content, manifest)
    cells = notebook["cells"]
    session_code = """# Hard 7,200-second Kaggle CLI cap includes 300 seconds before this first cell.
import time
SESSION_START = time.monotonic()
SESSION_WATCHDOG_SECONDS = 6900.0
SESSION_CLOSE_MARGIN = 60.0
def session_elapsed():
    return time.monotonic() - SESSION_START
"""
    session_cell = {"id": "session-budget-watchdog", "cell_type": "code", "metadata": {},
                    "execution_count": None, "outputs": [], "source": session_code.splitlines(keepends=True)}
    cells.insert(0, session_cell)

    by_id = {cell["id"]: cell for cell in cells}

    # Load the verified bundle runtime by its extracted root path. The generic
    # notebook bootstrap's `from runtime import ...` can resolve the wrong module
    # or fail when only the nested source package is included.
    bootstrap = "".join(by_id["bundle-restore"]["source"])
    old_runtime_import = "sys.path.insert(0, str(BUNDLE_DIR))\nfrom runtime import verify_bundle\nSPEC = verify_bundle(BUNDLE_DIR)"
    new_runtime_import = """sys.path.insert(0, str(BUNDLE_DIR))
import importlib.util
_runtime_path = BUNDLE_DIR / 'runtime.py'
if not _runtime_path.is_file():
    raise FileNotFoundError(f'Bundled root runtime is missing: {_runtime_path}')
_runtime_spec = importlib.util.spec_from_file_location('af_pikans_bundle_bootstrap_runtime', _runtime_path)
if _runtime_spec is None or _runtime_spec.loader is None:
    raise ImportError(f'Could not load bundled root runtime: {_runtime_path}')
_runtime_module = importlib.util.module_from_spec(_runtime_spec)
sys.modules[_runtime_spec.name] = _runtime_module
_runtime_spec.loader.exec_module(_runtime_module)
SPEC = _runtime_module.verify_bundle(BUNDLE_DIR)"""
    if old_runtime_import not in bootstrap:
        raise RuntimeError("Expected generic bundle-restore runtime import was not found")
    set_source(by_id["bundle-restore"], bootstrap.replace(old_runtime_import, new_runtime_import, 1))

    set_source(by_id["overview"], """# Burgers pareado: RBF fijo frente a PINNacle Vanilla

Paquete candidato para cinco parejas de seeds nuevas (811–815), con dos Tesla T4 y hasta dos parejas concurrentes. Cada pareja crea una pool Sobol float64 y comparte sus bytes entre los brazos. Cada brazo usa Adam 1e-3, hasta 20,000 updates o 900 GPU-s. Vanilla es MLP tanh 5×100 y no incorpora L-BFGS.

La evaluación Cole–Hopf/Gauss–Hermite de orden 128 en grid 1025×129 se ejecuta en CUDA, solo tras completar todo el entrenamiento y únicamente para checkpoints finales. Cada seed forma una wave de dos procesos score, uno por T4, con cap de 90 s de pared; las cinco waves caben en 450 s. Se conservan checkpoints intermedios para costo, pero no se puntúan.

El artículo PINNacle es contextual: reporta Vanilla con referencia FEM discretizada y protocolo distinto. El cap de sesión solicitado es 7,200 s, incluidos 300 s de arranque previos a la primera celda. El watchdog interno permite 6,900 s desde esa celda y deja 60 s de cierre. La aritmética de peor caso se detalla en manifest.json y budget.json. Este paquete sigue sin aprobación ni autorización de ejecución.
""")
    set_source(by_id["reproducibility"], """## Reproducibilidad y carga del paquete

La libreta restaura un ZIP autocontenido y verifica su `bundle_id` y SHA-256 de cada archivo. El paquete usa copias estáticas de las fuentes indicadas en `sha256-report.json`; no clona Git ni empaqueta credenciales.
""")
    set_source(by_id["hardware-note"], """## Preflight: dos Tesla T4

Dos workers de seed se ejecutan concurrentemente; cada worker ve una T4, entrena RBF y Vanilla secuencialmente y comparte pools exactas. Los dos workers usan los slots 0 y 1. El scorer final también inicia dos procesos paralelos con una GPU visible cada uno. La telemetría registra el slot y el nombre reportado por CUDA.
""")
    set_source(by_id["sampling-note"], """## Pools compartidas por pareja

El wrapper genera una sola pool Sobol por seed, guarda sus cinco arrays y sus hashes, y confirma igualdad exacta después de cada brazo. El smoke no puntúa error de campo. La inspección detallada de cobertura se conserva en `paired_pool_manifest.json` por seed.
""")
    set_source(by_id["sampling-signals"], """# No se generan pools antes de la ejecución.
# Cada worker las genera una vez para compartirlas entre ambos brazos.
print('Pooled sampling is performed once per paired seed by the reviewed entrypoint.')
""")
    set_source(by_id["training-note"], """## Smoke y entrenamiento por pares

Primero se ejecuta un único smoke de 60 GPU-s por brazo sin scoring ni selección. Si termina dentro del cap de 240 s de pared, `run_bundle` procesa las cinco parejas en tres waves, con hasta dos parejas concurrentes y 1,920 s de tope por pareja. Cada brazo para en 20,000 Adam updates o 900 GPU-s. Un fallo se conserva; no se reemplaza seed.
""")

    smoke_code = r'''import copy, json, os, subprocess, sys, tempfile
from pathlib import Path

manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
if manifest.get('budget', {}).get('approved') is not True:
    raise RuntimeError('Draft only: human budget approval is required before smoke or training.')
if manifest.get('research_status', {}).get('execution_authorized') is not True:
    raise RuntimeError('Draft only: execution authorization is required before smoke or training.')
smoke_guard = WORK_ROOT / 'smoke-started.json'
if smoke_guard.exists():
    raise RuntimeError('Smoke is single-use under this budget; inspect its prior artifacts before any retry.')
smoke_root = WORK_ROOT / 'smoke'
smoke_root.mkdir(parents=True, exist_ok=True)
smoke_attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=smoke_root))
smoke_manifest = copy.deepcopy(manifest)
smoke_manifest['training']['seeds'] = [manifest['training']['seeds'][0]]
smoke_manifest['training']['seconds_per_arm'] = int(smoke_manifest['training']['smoke_seconds_per_arm'])
smoke_manifest_path = smoke_attempt / 'manifest-smoke.json'
smoke_manifest_path.write_text(json.dumps(smoke_manifest, indent=2), encoding='utf-8')
smoke_run = smoke_attempt / f"seed-{manifest['training']['seeds'][0]}"
smoke_guard.write_text(json.dumps({'status': 'started', 'run_dir': str(smoke_run)}), encoding='utf-8')
smoke_env = os.environ.copy()
smoke_env.update({
    'HARNESS_MANIFEST': str(smoke_manifest_path),
    'HARNESS_RUN_DIR': str(smoke_run),
    'HARNESS_SEED': str(manifest['training']['seeds'][0]),
    'HARNESS_GPU_SLOT': GPU_SLOTS[0],
    'CUDA_VISIBLE_DEVICES': GPU_SLOTS[0],
    'PYTHONPATH': str(BUNDLE_DIR),
    'PYTHONUNBUFFERED': '1',
})
if session_elapsed() > 300.0:
    raise RuntimeError('Setup exceeded its 300-second allowance before smoke.')
smoke_process = subprocess.run(
    [sys.executable, str(BUNDLE_DIR / SPEC['entrypoint'])],
    cwd=BUNDLE_DIR, env=smoke_env, check=False, timeout=240,
)
if smoke_process.returncode != 0:
    raise RuntimeError(f'Paired smoke failed with exit code {smoke_process.returncode}; do not retry automatically.')
smoke_guard.write_text(json.dumps({'status': 'completed', 'run_dir': str(smoke_run)}), encoding='utf-8')
smoke_status = json.loads((smoke_run / 'problem_status.json').read_text(encoding='utf-8'))
if smoke_status.get('shared_pool_exact') is not True:
    raise RuntimeError('Smoke did not preserve exact pairwise pool identity.')
print('Smoke completed; no evaluator or held-out field score was run.')
'''
    smoke_cell = {"id": "paired-smoke", "cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": smoke_code.splitlines(keepends=True)}
    preflight_index = next(i for i, cell in enumerate(cells) if cell["id"] == "gpu-preflight")
    cells.insert(preflight_index + 1, smoke_cell)

    run_code = r'''import importlib.util, json, sys
runtime_path = BUNDLE_DIR / 'src' / 'research_pipeline' / 'remote' / 'runtime.py'
if not runtime_path.is_file():
    raise FileNotFoundError(f'Bundled remote runtime is missing: {runtime_path}')
runtime_spec = importlib.util.spec_from_file_location('af_pikans_remote_runtime', runtime_path)
if runtime_spec is None or runtime_spec.loader is None:
    raise ImportError(f'Could not load bundled remote runtime: {runtime_path}')
runtime_module = importlib.util.module_from_spec(runtime_spec)
sys.modules[runtime_spec.name] = runtime_module
runtime_spec.loader.exec_module(runtime_module)
run_bundle = runtime_module.run_bundle

manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
if manifest.get('budget', {}).get('approved') is not True or manifest.get('research_status', {}).get('execution_authorized') is not True:
    raise RuntimeError('Draft only: signed budget and execution authorization are required.')
smoke_status = json.loads((WORK_ROOT / 'smoke-started.json').read_text(encoding='utf-8'))
if smoke_status.get('status') != 'completed':
    raise RuntimeError('The single paired smoke must finish before Tier 2.')
remaining = 3 * 1920 + 5 * 90 + 90
if session_elapsed() + remaining > SESSION_WATCHDOG_SECONDS - SESSION_CLOSE_MARGIN:
    raise RuntimeError('Insufficient guarded time for all Tier 2 pairs, endpoint scoring, and artifact flush.')
campaign_state = run_bundle(BUNDLE_DIR, RUNS_DIR, max_parallel=2, gpu_slots=GPU_SLOTS)
display(pd.DataFrame([{'seed': seed, 'status': status} for seed, status in sorted(campaign_state['seeds'].items())]))
if any(status != 'completed' for status in campaign_state['seeds'].values()):
    raise RuntimeError('Incomplete paired campaign; keep all failures and do not replace seeds.')
'''
    set_source(by_id["parallel-run"], run_code)
    score_code = r'''import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd
import torch
from IPython.display import display

manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
expected_seeds = manifest['training']['seeds']
seed_dirs = sorted(RUNS_DIR.glob('seed-*'))
if [int(path.name.split('-')[1]) for path in seed_dirs] != expected_seeds:
    raise RuntimeError('Endpoint scoring requires all five fixed seeds; no seed may be dropped.')
SCORE_WAVE_SECONDS = 90.0
SCORE_WAVES = len(expected_seeds)
ARTIFACT_FLUSH_SECONDS = 90.0
CLOSE_MARGIN_SECONDS = SESSION_CLOSE_MARGIN
score_root = ANALYSIS_DIR / 'scores'
score_root.mkdir(parents=True, exist_ok=True)
ledger_path = ANALYSIS_DIR / 'endpoint-score-ledger.json'
score_ledger = {'status': 'running', 'score_wave_cap_seconds': SCORE_WAVE_SECONDS, 'waves': []}
remaining_score_wall = SCORE_WAVES * SCORE_WAVE_SECONDS
if session_elapsed() + remaining_score_wall + ARTIFACT_FLUSH_SECONDS + CLOSE_MARGIN_SECONDS > SESSION_WATCHDOG_SECONDS:
    raise RuntimeError('Insufficient watchdog budget for all endpoint score waves and artifact flush.')

records = []
scorer = BUNDLE_DIR / 'burgers_cuda_score.py'
for seed_index, seed in enumerate(expected_seeds):
    remaining_after = (SCORE_WAVES - seed_index) * SCORE_WAVE_SECONDS
    if session_elapsed() + remaining_after + ARTIFACT_FLUSH_SECONDS + CLOSE_MARGIN_SECONDS > SESSION_WATCHDOG_SECONDS:
        score_ledger['status'] = 'incomplete_time_budget'
        ledger_path.write_text(json.dumps(score_ledger, indent=2), encoding='utf-8')
        raise RuntimeError('Endpoint scoring cannot finish within the guarded session budget.')
    jobs = []
    for arm_index, arm in enumerate(('rbf', 'vanilla')):
        relative = 'burgers-rbf' if arm == 'rbf' else 'burgers-vanilla'
        checkpoint = RUNS_DIR / f'seed-{seed}' / relative / 'checkpoint-final.pt'
        if not checkpoint.is_file():
            raise RuntimeError(f'Missing endpoint checkpoint: {checkpoint}')
        out = score_root / f'seed-{seed}' / f'{arm}-endpoint'
        out.mkdir(parents=True, exist_ok=True)
        stdout = (out / 'stdout.log').open('w', encoding='utf-8')
        stderr = (out / 'stderr.log').open('w', encoding='utf-8')
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': GPU_SLOTS[arm_index],
            'PYTHONPATH': str(BUNDLE_DIR),
            'PYTHONUNBUFFERED': '1',
        })
        command = [sys.executable, str(scorer), '--checkpoint', str(checkpoint),
                   '--arm', arm, '--seed', str(seed), '--out', str(out)]
        process = subprocess.Popen(command, cwd=BUNDLE_DIR, env=env, stdout=stdout, stderr=stderr)
        jobs.append((arm, out, process, stdout, stderr))
    wave_start = time.monotonic()
    while any(job[2].poll() is None for job in jobs):
        if (time.monotonic() - wave_start > SCORE_WAVE_SECONDS or
                session_elapsed() >= SESSION_WATCHDOG_SECONDS - SESSION_CLOSE_MARGIN):
            for arm, out, process, stdout, stderr in jobs:
                if process.poll() is None:
                    process.terminate()
                stdout.close()
                stderr.close()
            score_ledger['status'] = 'incomplete_score_wave_timeout'
            score_ledger['waves'].append({'seed': seed, 'status': 'timeout'})
            ledger_path.write_text(json.dumps(score_ledger, indent=2), encoding='utf-8')
            raise RuntimeError(f'CUDA endpoint score wave for seed {seed} exceeded its 90-second allowance.')
        time.sleep(1.0)
    wave_rows = []
    for arm, out, process, stdout, stderr in jobs:
        code = process.wait()
        stdout.close()
        stderr.close()
        if code != 0:
            score_ledger['status'] = 'incomplete_score_process_failure'
            score_ledger['waves'].append({'seed': seed, 'arm': arm, 'returncode': code})
            ledger_path.write_text(json.dumps(score_ledger, indent=2), encoding='utf-8')
            raise RuntimeError(f'CUDA endpoint scorer failed for seed {seed}, arm {arm}.')
        report = json.loads((out / 'score.json').read_text(encoding='utf-8'))
        if not report.get('endpoint_only') or 't4' not in report.get('scoring_device', '').lower():
            raise RuntimeError('Expected endpoint-only scoring on the assigned CUDA T4.')
        metrics = report['metrics']
        row = {'seed': seed, 'arm': arm, 'checkpoint': 'checkpoint-final.pt',
               'wall_score_wave_seconds': time.monotonic() - wave_start,
               'device': report['scoring_device'],
               'evaluation_gpu_seconds': report['evaluation_gpu_seconds_estimate'],
               'field_relative_l2': float(metrics['field_relative_l2']),
               'field_rmse_abs': float(metrics['field_rmse_abs']),
               'residual_rmse_abs': float(metrics['residual_rmse_abs']),
               'scientific_accepted': False}
        wave_rows.append(row)
        records.append(row)
    score_ledger['waves'].append({'seed': seed, 'status': 'scored', 'rows': wave_rows})
    ledger_path.write_text(json.dumps(score_ledger, indent=2), encoding='utf-8')

if session_elapsed() + ARTIFACT_FLUSH_SECONDS + CLOSE_MARGIN_SECONDS > SESSION_WATCHDOG_SECONDS:
    score_ledger['status'] = 'incomplete_artifact_flush_budget'
    ledger_path.write_text(json.dumps(score_ledger, indent=2), encoding='utf-8')
    raise RuntimeError('Insufficient guarded time for endpoint summary artifacts.')
scores = pd.DataFrame(records).sort_values(['seed', 'arm'])
scores.to_csv(ANALYSIS_DIR / 'paired-endpoint-scores.csv', index=False)
pairs = scores.pivot(index='seed', columns='arm', values='field_relative_l2').reindex(expected_seeds)
if len(pairs) != 5 or pairs.isna().any().any():
    raise RuntimeError('Five complete final endpoint pairs are required.')
score_device = torch.device('cuda:0')
log_ratios = torch.log(torch.as_tensor(
    pairs['rbf'].to_numpy() / pairs['vanilla'].to_numpy(), dtype=torch.float64, device=score_device))
rng = torch.Generator(device=score_device).manual_seed(20260927)
bootstrap_indices = torch.randint(0, len(log_ratios), (10000, len(log_ratios)), generator=rng, device=score_device)
bootstrap = log_ratios[bootstrap_indices].mean(dim=1)
bootstrap_interval = torch.exp(torch.quantile(bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64, device=score_device)))
summary = {
    'n_pairs': 5,
    'primary_endpoint': 'final checkpoint only, scored after all training; no intermediate checkpoint scores',
    'summary_device': torch.cuda.get_device_name(0),
    'geometric_mean_rbf_over_vanilla': float(torch.exp(log_ratios.mean()).item()),
    'paired_log_ratio_values': {str(seed): float(value) for seed, value in zip(expected_seeds, log_ratios.detach().cpu().tolist())},
    'bootstrap_95_percentile_interval_ratio': bootstrap_interval.detach().cpu().tolist(),
    'proposed_noninferiority_margin': 1.10,
    'margin_signed': False,
    'decision': 'descriptive_only_pending_signed preregistration',
    'scores_used_for_tuning': False,
    'scientific_accepted': False,
}
(ANALYSIS_DIR / 'paired-primary-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\\n', encoding='utf-8')
score_ledger['status'] = 'complete'
score_ledger['total_wall_seconds'] = sum(wave['rows'][0]['wall_score_wave_seconds'] for wave in score_ledger['waves'] if wave.get('status') == 'scored')
ledger_path.write_text(json.dumps(score_ledger, indent=2), encoding='utf-8')
display(scores)
display(pd.DataFrame([summary]))
'''
    score_cell = {"id": "post-training-scoring", "cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": score_code.splitlines(keepends=True)}
    run_index = next(i for i, cell in enumerate(cells) if cell["id"] == "parallel-run")
    cells.insert(run_index + 1, score_cell)

    set_source(by_id["signals-note"], """## Score confirmatorio de endpoints

Se puntúa solo `checkpoint-final.pt`, para ambos brazos de cada una de las cinco seeds y después de concluir todo el entrenamiento. El scorer independiente calcula referencia, derivadas, malla, residuales y métricas en CUDA. Los scores no se usan para tuning. El bootstrap descriptivo se predefine en el notebook; el margen 1.10 sigue sin firma y no genera una decisión de aceptación.
""")
    set_source(by_id["quantitative-signals"], """from IPython.display import display
import json, pandas as pd
scores = pd.read_csv(ANALYSIS_DIR / 'paired-endpoint-scores.csv')
summary = json.loads((ANALYSIS_DIR / 'paired-primary-summary.json').read_text(encoding='utf-8'))
display(scores)
display(pd.DataFrame([summary]))
""")
    set_source(by_id["curves-note"], """## Comparación de endpoints

Este paquete conserva checkpoints intermedios para documentar costo, pero no los puntúa. El gráfico siguiente compara los diez scores finales pareados; no representa una curva de aprendizaje ni selecciona checkpoints.
""")
    curves_code = r'''import pandas as pd
import plotly.express as px
from IPython.display import display
scores = pd.read_csv(ANALYSIS_DIR / 'paired-endpoint-scores.csv')
figure = px.scatter(scores, x='arm', y='field_relative_l2', color='seed', log_y=True,
                    labels={'arm': 'arm', 'field_relative_l2': 'relative L2'},
                    title='Final endpoint scores (each point scored on CUDA)')
figure.write_html(ANALYSIS_DIR / 'paired-endpoint-scores.html', include_plotlyjs='cdn')
display(figure)
'''
    set_source(by_id["visualizations"], curves_code)
    set_source(by_id["fields-note"], """## Campos evaluados

Cada scorer produce arrays de predicción, referencia, error y residual sobre la malla CUDA común. Se guardan en archivos NPZ para inspección posterior; las conversiones a CPU ocurren únicamente al serializar esos arrays.
""")
    fields_code = r'''from IPython.display import IFrame, display
import json
from pathlib import Path
for seed in json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))['training']['seeds']:
    for arm in ('rbf', 'vanilla'):
        archive = ANALYSIS_DIR / 'scores' / f'seed-{seed}' / f'{arm}-endpoint' / 'validation.npz'
        if archive.is_file():
            print(f'seed {seed}, {arm}: final CUDA-scored prediction/reference/error/residual at {archive}')
'''
    set_source(by_id["field-maps"], fields_code)
    set_source(by_id["closure-note"], """## Cierre y hashes de resultados

El ZIP conserva el bundle, configuración, pools, checkpoints, costo, telemetría y score confirmatorio. Ninguna cifra se marca como aceptada. El paquete exige aprobación humana ligada a este hash antes de cualquier ejecución remota.
""")
    closure_code = r'''import json
from src.research_pipeline.remote.postprocess import create_results_archive
manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
(ANALYSIS_DIR / 'parameter-decisions.md').write_text(manifest['parameter_decisions_markdown'], encoding='utf-8')
archive_path = WORK_ROOT / 'burgers-rbf-vanilla-paired-results.zip'
archive_result = create_results_archive(WORK_ROOT, ANALYSIS_DIR, archive_path)
print('Paired result archive:', archive_result['archive'], '; hashed files:', archive_result['file_count'])
print('Draft protocol only; no scientific acceptance or SOTA claim is asserted.')
'''
    set_source(by_id["closure"], closure_code)

    notebook["metadata"]["research"] = {
        "notebook_id": f"burgers-paired.{spec['bundle_id'][:16]}",
        "role": "candidate",
        "source_of_truth": "manifest.json; budget/execution authorized; exact-bundle gate retained",
        "execution_authorized": True,
    }
    return notebook


def main() -> None:
    copy_sources()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    content = {}
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.name != "bundle.json":
            content[path.relative_to(PACKAGE).as_posix()] = path.read_bytes()
    remote = manifest["remote"]
    required = sorted(set(manifest["artifacts"]["expected"]) | {"manifest_snapshot.json"})
    spec = {
        "schema_version": 1,
        "kernel": remote["kernel"],
        "entrypoint": manifest["model"]["implementation"],
        "seeds": manifest["training"]["seeds"],
        "timeout_sec": remote["run_timeout_sec"],
        "max_total_training_sec": len(manifest["training"]["seeds"]) * 2 * manifest["training"]["seconds_per_arm"],
        "max_tier2_parallel_wall_sec": ((len(manifest["training"]["seeds"]) + remote["max_parallel_seeds"] - 1) // remote["max_parallel_seeds"]) * remote["run_timeout_sec"],
        "max_endpoint_scoring_wall_sec": manifest["budget"]["endpoint_scoring_wall_seconds"],
        "max_session_wall_sec": manifest["budget"]["requested_max_session_seconds"],
        "evidence_kind": "research",
        "enable_gpu": True,
        "enable_internet": bool(remote["enable_internet"]),
        "required_artifacts": required,
        "files": {name: sha256(data) for name, data in sorted(content.items())},
        "runtime_dependencies": remote["runtime_dependencies"],
        "setup_timeout_sec": remote["setup_timeout_sec"],
        "max_parallel_seeds": remote["max_parallel_seeds"],
        "machine_shape": remote["machine_shape"],
        "allow_gpu_count_fallback": remote["allow_gpu_count_fallback"],
    }
    spec["bundle_id"] = sha256(canonical(spec))
    bundle_bytes = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    (PACKAGE / "bundle.json").write_bytes(bundle_bytes)
    content["bundle.json"] = bundle_bytes

    notebook = build_notebook(manifest, spec, content)
    notebook_path = HERE / "experiment.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {
        "id": remote["kernel"],
        "title": "AF PIKANs Burgers RBF Vanilla Paired 20260927 R2",
        "code_file": "experiment.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": remote["machine_shape"],
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path = HERE / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    package_hashes = {name: sha256(data) for name, data in sorted(content.items())}
    transport = {
        "bundle_id": spec["bundle_id"],
        "experiment_ipynb_sha256": sha256(notebook_path.read_bytes()),
        "kernel_metadata_sha256": sha256(metadata_path.read_bytes()),
        "bundle_json_sha256": sha256(bundle_bytes),
        "packed_files": package_hashes,
        "execution_authorized": True,
    }
    (HERE / "transport-hashes.json").write_text(json.dumps(transport, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "bundle_id": spec["bundle_id"],
        "source_file_count": len(spec["files"]),
        "source_file_sha256": spec["files"],
        "bundle_json_sha256": transport["bundle_json_sha256"],
        "experiment_ipynb_sha256": transport["experiment_ipynb_sha256"],
        "kernel_metadata_sha256": transport["kernel_metadata_sha256"],
        "notebook_template_sha256": sha256((PROJECT / "src/research_pipeline/remote/notebook.py").read_bytes()),
        "transport_hashes_sha256": sha256((HERE / "transport-hashes.json").read_bytes()),
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "authorization": "execution authorized in session; exact-bundle approval gate retained; no Kaggle submit/run and no approval.json",
    }
    (HERE / "sha256-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("bundle_id", "source_file_count", "experiment_ipynb_sha256", "kernel_metadata_sha256", "transport_hashes_sha256", "builder_sha256", "authorization")}, indent=2))


if __name__ == "__main__":
    main()
