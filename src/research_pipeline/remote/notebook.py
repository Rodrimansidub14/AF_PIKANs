"""Structured, reproducible development notebook for remote experiments."""
from __future__ import annotations

import base64
import io
import zipfile
import zlib


def render_notebook(spec, content, manifest):
    model = manifest.get('model', {})
    model_label = model.get('label', model.get('id', manifest['iteration']['model_id']))
    problem_sequence = manifest.get('problem', {}).get('sequence', ['heat', 'burgers'])
    problem_labels = [name.replace('_', ' ').title() for name in problem_sequence]
    campaign_label = ' + '.join(problem_labels)
    is_allen_cahn_exploration = (
        problem_sequence == ['allen_cahn'] and manifest['iteration'].get('experiment_id') == 'EXP-0003'
    )
    is_dfg_baseline = manifest['iteration'].get('problem_id') == 'dfg_2d_1_steady_re20'
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    encoded = base64.b64encode(zlib.compress(packed.getvalue(), level=9)).decode("ascii")
    chunks = "\n".join("    " + repr(encoded[i:i + 96])
                       for i in range(0, len(encoded), 96))

    restore = '''import base64, io, os, pathlib, sys, zipfile, zlib

BUNDLE_ARCHIVE_B64 = (
__ARCHIVE_CHUNKS__
)
BUNDLE_ID = '__BUNDLE_ID__'
default_work = pathlib.Path('/kaggle/working') if pathlib.Path('/kaggle/working').exists() else pathlib.Path.cwd()
WORK_ROOT = pathlib.Path(os.environ.get('KAGGLE_WORKING_DIR', str(default_work))) / ('pikan-' + BUNDLE_ID)
BUNDLE_DIR = WORK_ROOT / 'bundle'
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
archive_bytes = zlib.decompress(base64.b64decode(BUNDLE_ARCHIVE_B64))
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    for item in archive.infolist():
        relative = pathlib.PurePosixPath(item.filename)
        if relative.is_absolute() or '..' in relative.parts or '\\\\' in item.filename:
            raise ValueError('Unsafe bundle path: ' + item.filename)
        target = (BUNDLE_DIR / pathlib.Path(*relative.parts)).resolve()
        if not target.is_relative_to(BUNDLE_DIR.resolve()):
            raise ValueError('Bundle path escaped destination: ' + item.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(item))
sys.path.insert(0, str(BUNDLE_DIR))
from runtime import verify_bundle
SPEC = verify_bundle(BUNDLE_DIR)
assert SPEC['bundle_id'].startswith(BUNDLE_ID)
RUNS_DIR = WORK_ROOT / 'runs'
ANALYSIS_DIR = WORK_ROOT / 'analysis'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
print('Bundle verified:', SPEC['bundle_id'], 'source files:', len(SPEC['files']))
'''.replace("__ARCHIVE_CHUNKS__", chunks).replace(
        "__BUNDLE_ID__", spec["bundle_id"][:16])

    install = '''import importlib.metadata, subprocess, sys

DEPENDENCIES = __DEPENDENCIES__
missing_or_different = []
for requirement in DEPENDENCIES:
    name, expected = requirement.split('==', 1)
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != expected:
        missing_or_different.append(requirement)
if missing_or_different:
    command = [sys.executable, '-m', 'pip', 'install',
               '--disable-pip-version-check', *missing_or_different]
    print('Installing pinned dependencies:', ', '.join(missing_or_different), flush=True)
    subprocess.run(command, check=True, timeout=__SETUP_TIMEOUT__)
else:
    print('Pinned experiment dependencies already match the manifest.')
'''.replace("__DEPENDENCIES__", repr(spec.get("runtime_dependencies", []))).replace(
        "__SETUP_TIMEOUT__", str(spec.get("setup_timeout_sec", 900)))

    preflight = '''import platform, subprocess, torch
import pandas as pd
from IPython.display import display

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; the approved T4x2 campaign did not start.')
visible_gpus = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(index) for index in range(visible_gpus)]
if visible_gpus < 2:
    raise RuntimeError(f'T4x2 was requested, but only {visible_gpus} CUDA devices are visible: {gpu_names}')
if any('T4' not in name.upper() for name in gpu_names[:2]):
    raise RuntimeError(f'The requested accelerator is 2xT4; visible devices are {gpu_names[:2]}')
GPU_SLOTS = ['0', '1']
hardware = pd.DataFrame([
    {'logical_gpu': i, 'name': gpu_names[i],
     'memory_total_mib': round(torch.cuda.get_device_properties(i).total_memory / 2**20),
     'parallel_seed_slot': GPU_SLOTS[i]} for i in range(2)
])
display(hardware)
display(pd.DataFrame([{'python': platform.python_version(), 'pytorch': torch.__version__,
                       'cuda_runtime': torch.version.cuda, 'visible_gpu_count': visible_gpus}]))
subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv'],
               check=True, text=True)
'''

    if spec.get('allow_gpu_count_fallback') is True:
        preflight = '''import platform, subprocess, torch
import pandas as pd
from IPython.display import display

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; this campaign requires at least one GPU.')
visible_gpus = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(index) for index in range(visible_gpus)]
GPU_SLOTS = [str(index) for index in range(min(SPEC['max_parallel_seeds'], visible_gpus))]
if not GPU_SLOTS:
    raise RuntimeError(f'No usable CUDA device is visible: {gpu_names}')
if SPEC.get('machine_shape') == 'NvidiaTeslaT4' and any('T4' not in name.upper() for name in gpu_names[:len(GPU_SLOTS)]):
    raise RuntimeError(f'The requested accelerator is NvidiaTeslaT4; visible devices are {gpu_names[:len(GPU_SLOTS)]}')
hardware = pd.DataFrame([
    {'logical_gpu': i, 'name': gpu_names[i],
     'memory_total_mib': round(torch.cuda.get_device_properties(i).total_memory / 2**20),
     'parallel_seed_slot': GPU_SLOTS[i]} for i in range(len(GPU_SLOTS))
])
display(hardware)
display(pd.DataFrame([{'python': platform.python_version(), 'pytorch': torch.__version__,
                       'cuda_runtime': torch.version.cuda, 'visible_gpu_count': visible_gpus,
                       'active_gpu_count': len(GPU_SLOTS)}]))
subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv'],
               check=True, text=True)
'''

    sampling = '''import json, numpy as np, pandas as pd, plotly.express as px
from scipy.spatial import cKDTree
from scipy.stats import qmc
from IPython.display import display
from src.research_pipeline.sciml.sampling import pools

manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
cfg = manifest['training']
sampling_rows = []
sampling_points = {}
for seed in cfg['seeds']:
    for problem in manifest['problem']['sequence']:
        effective = {**cfg, **cfg.get('problem_overrides', {}).get(problem, {})}
        data = pools(problem, seed, effective['interior_points'], effective['boundary_points'])
        points = data['interior'].copy()
        points[:, 0] = (points[:, 0] - (0.0 if problem == 'heat' else -1.0)) / (1.0 if problem == 'heat' else 2.0)
        neighbors = cKDTree(points).query(points, k=2)[0][:, 1]
        sampling_rows.append({
            'problem': problem, 'seed': seed, 'interior_n': len(data['interior']),
            'diagnostic_n': len(data['diagnostic']),
            'boundary_n': (len(data['boundary_probe']) if 'boundary_probe' in data
                           else len(data['left']) + len(data['right'])),
            'initial_n': (len(data['initial_probe']) if 'initial_probe' in data
                          else len(data['initial'])),
            'sobol_centered_discrepancy': qmc.discrepancy(points, method='CD'),
            'nn_min': np.min(neighbors), 'nn_median': np.median(neighbors),
            'nn_p95': np.quantile(neighbors, .95), 'nn_max': np.max(neighbors),
        })
        if seed == cfg['seeds'][0]:
            sampling_points[problem] = data['interior']
sampling_table = pd.DataFrame(sampling_rows)
display(sampling_table.style.format(precision=6))
sampling_table.to_csv(ANALYSIS_DIR / 'sampling-signals.csv', index=False)
for problem, points in sampling_points.items():
    figure = px.scatter(x=points[::4, 0], y=points[::4, 1], opacity=.55,
                        labels={'x': 'space coordinate x', 'y': 'time coordinate t'},
                        title=f'{problem.title()} interior Sobol pool, seed {cfg["seeds"][0]}')
    figure.write_html(ANALYSIS_DIR / f'{problem}-sampling.html', include_plotlyjs='cdn')
    display(figure)
'''

    if is_allen_cahn_exploration:
        sampling = '''import json, numpy as np, pandas as pd, plotly.express as px
from IPython.display import display
from src.research_pipeline.sciml.sampling import pools
from src.research_pipeline.sciml.train import sampling_signals

manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
cfg = manifest['training']
sampling_rows = []
sampling_points = None
for seed in cfg['seeds']:
    effective = {**cfg, **cfg.get('problem_overrides', {}).get('allen_cahn', {})}
    data = pools('allen_cahn', seed, effective['interior_points'],
                 effective['boundary_points'], heldout=effective['heldout_points'])
    sampling_rows.append({'problem': 'allen_cahn', 'seed': seed,
                          **sampling_signals('allen_cahn', data)})
    if seed == cfg['seeds'][0]:
        sampling_points = data
sampling_table = pd.DataFrame(sampling_rows)
display(sampling_table.style.format(precision=6))
sampling_table.to_csv(ANALYSIS_DIR / 'sampling-signals.csv', index=False)
points = pd.DataFrame({
    'x': np.r_[sampling_points['interior'][::16, 0], sampling_points['heldout'][::32, 0]],
    't': np.r_[sampling_points['interior'][::16, 1], sampling_points['heldout'][::32, 1]],
    'pool': (['train'] * len(sampling_points['interior'][::16])
             + ['heldout'] * len(sampling_points['heldout'][::32])),
})
figure = px.scatter(points, x='x', y='t', color='pool', opacity=.55,
                    labels={'x': 'space x', 't': 'time t'},
                    title=f'Allen–Cahn: independent training and held-out Sobol pools, seed {cfg["seeds"][0]}')
figure.write_html(ANALYSIS_DIR / 'allen-cahn-sampling.html', include_plotlyjs='cdn')
display(figure)
'''

    run = '''from runtime import run_bundle

MAX_PARALLEL_SEEDS = SPEC['max_parallel_seeds']
if len(GPU_SLOTS) != MAX_PARALLEL_SEEDS:
    raise RuntimeError('GPU slot assignment does not match approved parallelism.')
seed_order = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))['training']['seeds']
print('Seed schedule:', {seed: GPU_SLOTS[index] for index, seed in enumerate(seed_order[:MAX_PARALLEL_SEEDS])},
      '; later seeds use the first free T4.', flush=True)
campaign_state = run_bundle(BUNDLE_DIR, RUNS_DIR, max_parallel=MAX_PARALLEL_SEEDS,
                            gpu_slots=GPU_SLOTS)
display(pd.DataFrame([{'seed': seed, 'status': status}
                      for seed, status in sorted(campaign_state['seeds'].items())]))
if any(status != 'completed' for status in campaign_state['seeds'].values()):
    raise RuntimeError('Campaign ended with incomplete seeds: ' + repr(campaign_state['seeds']))
'''

    if spec.get('allow_gpu_count_fallback') is True:
        run = '''from runtime import run_bundle

MAX_PARALLEL_SEEDS = len(GPU_SLOTS)
seed_order = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))['training']['seeds']
print('Seed schedule:', {seed: GPU_SLOTS[index] for index, seed in enumerate(seed_order[:MAX_PARALLEL_SEEDS])},
      '; later seeds use the first free assigned GPU.', flush=True)
campaign_state = run_bundle(BUNDLE_DIR, RUNS_DIR, max_parallel=len(GPU_SLOTS),
                            gpu_slots=GPU_SLOTS)
display(pd.DataFrame([{'seed': seed, 'status': status}
                      for seed, status in sorted(campaign_state['seeds'].items())]))
if any(status != 'completed' for status in campaign_state['seeds'].values()):
    raise RuntimeError('Campaign ended with incomplete seeds: ' + repr(campaign_state['seeds']))
'''

    quantitative = '''from src.research_pipeline.remote.reporting import (
    aggregate_quantitative_table, long_quantitative_table, quantitative_table)
from IPython.display import display

signals = quantitative_table(RUNS_DIR)
long_signals = long_quantitative_table(signals)
for family in sorted(long_signals['family'].unique()):
    print(f'Measured {family} signals; each value is from a saved run artifact.')
    display(long_signals.loc[long_signals['family'] == family]
            .sort_values(['problem', 'seed', 'signal']).style.format({'value': '{:.6g}'}))
aggregated_signals = aggregate_quantitative_table(signals)
display(aggregated_signals.style.format(precision=6, na_rep='not measured'))
signals.to_csv(ANALYSIS_DIR / 'quantitative-signals.csv', index=False)
long_signals.to_csv(ANALYSIS_DIR / 'quantitative-signals-with-units.csv', index=False)
aggregated_signals.to_csv(ANALYSIS_DIR / 'signal-distributions.csv')
'''

    curves = '''from src.research_pipeline.remote.reporting import training_figures
from IPython.display import display

for name, figure in training_figures(RUNS_DIR).items():
    figure.write_html(ANALYSIS_DIR / f'{name}.html', include_plotlyjs='cdn')
    display(figure)
'''

    fields = '''from src.research_pipeline.remote.reporting import field_figures
from IPython.display import display

for name, figure in field_figures(RUNS_DIR).items():
    figure.write_html(ANALYSIS_DIR / f'{name}.html', include_plotlyjs='cdn')
    display(figure)
'''

    closure = '''import json
from src.research_pipeline.remote.postprocess import create_results_archive
from src.research_pipeline.remote.reporting import write_results_notebook

manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
(ANALYSIS_DIR / 'parameter-decisions.md').write_text(
    manifest['parameter_decisions_markdown'], encoding='utf-8')
results_notebook = write_results_notebook(RUNS_DIR, ANALYSIS_DIR, SPEC['bundle_id'])
archive_path = WORK_ROOT / 'heat-burgers-t4x2-results.zip'
archive_result = create_results_archive(WORK_ROOT, ANALYSIS_DIR, archive_path)
print('Run closed. Results archive:', archive_result['archive'],
      '; files hashed:', archive_result['file_count'])
print('Exploratory measurements only; no SOTA or scientific claim is established.')
'''
    campaign_slug = '-'.join(name.replace('_', '-') for name in problem_sequence)
    closure = closure.replace('heat-burgers-t4x2-results.zip', f'{campaign_slug}-t4x2-results.zip')

    def markdown(cell_id, source):
        return {'id': cell_id, 'cell_type': 'markdown', 'metadata': {},
                'source': source.splitlines(keepends=True)}

    def code(cell_id, source):
        return {'id': cell_id, 'cell_type': 'code', 'metadata': {},
                'execution_count': None, 'outputs': [], 'source': source.splitlines(keepends=True)}

    if is_allen_cahn_exploration:
        run = run.replace(
            'from runtime import run_bundle',
            "from runtime import run_bundle\n"
            "from src.research_pipeline.sciml.campaign import assert_exploratory_launch_ready\n"
            "assert_exploratory_launch_ready(json.loads((BUNDLE_DIR/'manifest.json').read_text(encoding='utf-8')))"
        )

    kind = spec['evidence_kind']
    notebook_role = 'exploration' if kind == 'infrastructure' or manifest['iteration'].get('protocol_phase') == 'exploratory' else 'candidate'
    overview = f'''# {campaign_label}: exploración instrumentada

Esta libreta reproduce EXP-0003 y separa preparación, preflight, muestreo,
entrenamiento, medición, visualización y cierre. El objetivo es observar señales
con RBF-PIKAN fija y Sobol fijo; el piloto no compara métodos ni afirma SOTA.

| Etapa | Elección |
| --- | --- |
| Entrenamiento | PINA + PyTorch, con {model_label} instrumentado |
| Seguimiento | W&B offline y artefactos locales reproducibles |
| Exploración de señales y curvas | Plotly |
| Campos, geometrías y mallas | PyVista para mallas; no aplica a esta PDE 1D |
| Aplicación de análisis posterior | Panel + Plotly/PyVista, posterior |

La segunda secuencia Sobol permanece separada del entrenamiento. Se recalculan
derivadas y residuo en los puntos held-out. Se mide energía y cumplimiento de las
trazas duras. El error de campo continúa bloqueado mientras la referencia BDF no
esté convergida. Kaggle recibe el notebook y las fuentes, nunca secretos; W&B se
sincroniza desde el repositorio al final.
''' if is_allen_cahn_exploration else f'''# {campaign_label}: comparación exploratoria instrumentada

Esta libreta reproduce el manifiesto y separa preparación, preflight, muestreo,
entrenamiento, medición, visualización y cierre. Cuantifica señales por problema
y semilla; no afirma SOTA.

| Etapa | Elección |
| --- | --- |
| Entrenamiento | PINA + PyTorch, con {model_label} instrumentado |
| Seguimiento | W&B y artefactos locales reproducibles |
| Exploración de señales y curvas | Plotly |
| Campos, geometrías y mallas | PyVista, con exportación sin ventana |
| Aplicación de análisis posterior | Panel + Plotly/PyVista |

W&B trabaja offline durante la corrida; el ZIP conserva registros para
sincronizarlos desde el repositorio con la credencial local, sin enviar secretos
a Kaggle.
'''
    hardware_note = '''## 4. Preflight: dos T4 y asignación de semillas

Se exige que Kaggle exponga dos T4 antes de entrenar. Cada semilla usa una sola
GPU lógica; los workers ocupan slots separados y la tercera semilla toma el primer
slot libre. La telemetría conserva la asignación real de cada worker.
'''
    sampling_note = '''## 5. Generación, separación y cobertura de puntos

Cada semilla genera 8.192 puntos Sobol para entrenamiento y 16.384 para evaluación
física held-out con streams distintos. Se guardan las coordenadas exactas,
discrepancia centrada, distancias entre vecinos, límites del dominio y conteos de
trazas diagnósticas. Inicial y frontera se imponen mediante transformación dura:
no hay pools de pérdida blanda.
'''
    fields_note = '''## 9. Campo predicho, residuo y energía

Plotly muestra el campo predicho y el logaritmo del valor absoluto del residuo
recalculado. La curva de energía es descriptiva. No se incluye campo de referencia,
error L2 ni máscara de interfaces: el error de campo permanece bloqueado.
'''
    if is_dfg_baseline:
        overview = f'''# DFG 2D-1 Re20: baseline exploratorio RBF-PIKAN

La libreta registra el problema canónico de flujo incompresible alrededor de un
cilindro, con parámetros y fuentes fijados en el manifiesto. Usa autodiferenciación
directa de PyTorch para Navier–Stokes, la tracción natural de salida y las fuerzas
integradas sobre el cilindro. No altera el evaluador ni afirma SOTA.

| Etapa | Elección |
| --- | --- |
| Entrenamiento | PyTorch directo + RBF-PIKAN fija [2,16,16,3] |
| Seguimiento | W&B offline y artefactos locales reproducibles |
| Muestreo / señales | Sobol aleatorizado fijo con rechazo geométrico; métricas por región |
| Campos y malla | Plotly para mapas; PyVista VTR sin ventana |
| Referencia canónica | FEATFLOW 2D-1 Re20 para Cd, Cl y Δp |

Los puntos held-out usan un stream Sobol independiente. La referencia FEATFLOW
aporta observables integrales, no un campo FEM; por eso el error L2 de campo queda
sin reportar hasta disponer de una solución numérica independiente y convergida.
'''
        hardware_note = '''## 4. GPU y asignación

La campaña requiere CUDA y usará hasta dos T4. Una semilla ocupa una GPU lógica;
si Kaggle expone una sola T4, el fallback autorizado reduce el paralelismo a una.
La telemetría guarda el dispositivo real por semilla.
'''
        sampling_note = '''## 5. Muestreo Sobol con geometría DFG

Se genera el pool interior Sobol aleatorizado por rechazo fuera del cilindro y
pools independientes para entrada, paredes, cilindro y salida. Cada pool y sus
seeds se guardan; held-out usa otro stream fijo. La tabla y la figura de cobertura
se leen de los pools realmente archivados por cada worker, inmediatamente después
de que el bloque de entrenamiento los genere y termine.
'''
        fields_note = '''## 9. Campos, fuerzas y límites de referencia

PyVista exporta (u,v,p) como VTR sin abrir una ventana; Plotly muestra contornos
con el cilindro enmascarado. Cd, Cl y Δp se contrastan con FEATFLOW. No se calcula
error L2 de campo porque no hay una solución FEM independiente empaquetada.
'''
        sampling = '''import json, numpy as np, pandas as pd, plotly.express as px
from IPython.display import display

manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
rows = []
for seed in manifest['training']['seeds']:
    run = RUNS_DIR / f'seed-{seed}'
    signal = json.loads((run / 'sampling_signals.json').read_text(encoding='utf-8'))
    rows.append({key: value for key, value in signal.items()
                 if type(value) in (int, float, str)})
sampling_table = pd.DataFrame(rows)
display(sampling_table.style.format(precision=6))
sampling_table.to_csv(ANALYSIS_DIR / 'sampling-signals.csv', index=False)
first_seed = manifest['training']['seeds'][0]
with np.load(RUNS_DIR / f'seed-{first_seed}' / 'dfg_training_pools.npz') as archive:
    points = archive['interior']
figure = px.scatter(x=points[::8, 0], y=points[::8, 1], opacity=.5,
                    labels={'x': 'x [m]', 'y': 'y [m]'},
                    title=f'DFG Re20 fixed scrambled-Sobol training pool, seed {first_seed}')
figure.add_shape(type='circle', x0=.15, y0=.15, x1=.25, y1=.25,
                 line={'color': 'black', 'width': 2})
figure.write_html(ANALYSIS_DIR / 'dfg-sampling.html', include_plotlyjs='cdn')
display(figure)
'''
        quantitative = '''import json, pandas as pd
from IPython.display import display

records = []
for seed_dir in sorted(RUNS_DIR.glob('seed-*')):
    seed = int(seed_dir.name.removeprefix('seed-'))
    metrics = json.loads((seed_dir / 'metrics.json').read_text(encoding='utf-8'))
    records.extend({'seed': seed, 'signal': name, 'value': value}
                   for name, value in metrics.items())
signal_table = pd.DataFrame(records)
display(signal_table.pivot(index='signal', columns='seed', values='value').style.format(precision=6))
signal_table.to_csv(ANALYSIS_DIR / 'quantitative-signals.csv', index=False)
aggregate = signal_table.groupby('signal')['value'].agg(['count', 'mean', 'std'])
display(aggregate.style.format(precision=6, na_rep='not measured'))
aggregate.to_csv(ANALYSIS_DIR / 'signal-distributions.csv')
'''
        curves = '''from IPython.display import IFrame, display

for seed_dir in sorted(RUNS_DIR.glob('seed-*')):
    curve = seed_dir / 'dfg_training_losses.html'
    print(seed_dir.name, 'optimizer steps:', len((seed_dir / 'training_history.csv').read_text().splitlines()) - 1)
    display(IFrame(src=str(curve), width='100%', height=560))
'''
        fields = '''from IPython.display import IFrame, display
import numpy as np
import plotly.express as px
import pyvista as pv

for seed_dir in sorted(RUNS_DIR.glob('seed-*')):
    field = seed_dir / 'dfg_fields.html'
    grid = pv.read(seed_dir / 'dfg_fields.vtr')
    print(seed_dir.name, 'PyVista rectilinear field:', grid, 'point arrays:', grid.array_names)
    display(IFrame(src=str(field), width='100%', height=560))
    with np.load(seed_dir / 'dfg_heldout_residuals.npz') as saved:
        points = saved['points']
        magnitude = np.sqrt(saved['momentum_x']**2 + saved['momentum_y']**2 + saved['continuity']**2)
        figure = px.scatter(x=points[::4, 0], y=points[::4, 1],
                            color=np.log10(magnitude[::4] + 1e-8), opacity=.6,
                            labels={'x': 'x [m]', 'y': 'y [m]', 'color': 'log10 |R|'},
                            title=f"{seed_dir.name}: independent held-out residuals")
    figure.add_shape(type='circle', x0=.15, y0=.15, x1=.25, y1=.25,
                     line={'color': 'black', 'width': 2})
    figure.write_html(ANALYSIS_DIR / f"{seed_dir.name}-heldout-residuals.html",
                      include_plotlyjs='cdn')
    display(figure)
'''
        closure = '''import json
from src.research_pipeline.remote.postprocess import create_results_archive

manifest = json.loads((BUNDLE_DIR / 'manifest.json').read_text(encoding='utf-8'))
(ANALYSIS_DIR / 'parameter-decisions.md').write_text(
    manifest['parameter_decisions_markdown'], encoding='utf-8')
archive = WORK_ROOT / 'dfg-re20-t4x2-results.zip'
result = create_results_archive(WORK_ROOT, ANALYSIS_DIR, archive)
print('DFG Re20 baseline archived:', result['archive'], '; hashed files:', result['file_count'])
print('FEATFLOW QoIs and held-out residuals are exploratory; no field error or SOTA claim is asserted.')
'''
    dependency_note = '''## 2. Dependencias y versiones

PyTorch/CUDA se comprueban contra Kaggle. Las dependencias restantes se instalan
solo si falta un pin o su versión difiere. Panel y PyVista no se instalan en el
entrenamiento porque estos dominios 1D no los necesitan.
'''
    training_stack = 'PINA + PyTorch'
    if is_dfg_baseline:
        dependency_note = '''## 2. Dependencias y versiones

PyTorch/CUDA provienen de la imagen GPU de Kaggle. SciPy, Plotly, PyVista/VTK,
Pandas y W&B se fijan por versión en el bundle. PyVista exporta VTR sin abrir
ventanas; no se necesita servidor gráfico.
'''
        training_stack = 'PyTorch directo con autodiferenciación de orden dos'
    cells = [
        markdown('overview', overview),
        markdown('reproducibility', '''## 1. Reproducibilidad y carga del paquete

La libreta incluye fuentes versionadas y sus hashes; no clona Git ni embebe secretos.
'''),
        code('bundle-restore', restore),
        markdown('dependencies-note', dependency_note),
        code('dependencies', install),
        markdown('protocol', '''## 3. Protocolo y decisiones de parámetros

El manifiesto es la fuente de verdad para condiciones, arquitectura, muestreo,
semillas, pasos, tiempo, diagnóstico y versiones. Las tablas reportan valores
calculados desde artefactos; las decisiones completas se conservan junto a resultados.
'''),
        markdown('hardware-note', hardware_note if is_allen_cahn_exploration or is_dfg_baseline else '''## 4. Preflight: dos T4 y asignación de semillas

Se exige que Kaggle exponga dos T4 antes de entrenar. Una semilla ocupa una GPU y
entrena Heat seguido de Burgers; la tercera comienza en la primera GPU libre. Cada
worker ve una sola GPU lógica mediante CUDA_VISIBLE_DEVICES.
'''),
        code('gpu-preflight', preflight),
        markdown('sampling-note', sampling_note if is_allen_cahn_exploration or is_dfg_baseline else '''## 5. Generación y cobertura de puntos

La tabla presenta conteos generados, discrepancia centrada de Sobol y distancias
al vecino más cercano en coordenadas normalizadas. La figura muestra los puntos
generados realmente para la primera semilla.
'''),
        code('sampling-signals', sampling),
        markdown('training-note', f'''## 6. Entrenamiento concurrente {training_stack} + {model_label}

Este es el único bloque prolongado. Usa el runtime reproducible del repositorio,
con workers por semilla y asignación explícita de GPU. La curva se registra cada
100 pasos, los diagnósticos físicos y propios de la arquitectura cada 250 y un
checkpoint final por problema.
El límite de pasos o tiempo no implica convergencia.
'''),
        code('parallel-run', run),
        markdown('signals-note', f'''## 7. Tabla de señales cuantificadas

Cada fila es un problema y una semilla. Las columnas informan mediciones de puntos,
pérdidas, gradientes, autodiferenciación, activaciones y señales específicas de
{model_label}, sensibilidad, validación y recursos. "not measured" significa ausente,
nunca cero.
'''),
        code('quantitative-signals', quantitative),
        markdown('curves-note', '''## 8. Curvas de aprendizaje y señales internas

Las curvas salen de history.jsonl y de diagnósticos periódicos guardados. Plotly
muestra pérdidas, norma de gradiente y señales de residual durante el entrenamiento.
'''),
        code('visualizations', curves),
        markdown('fields-note', fields_note if is_allen_cahn_exploration or is_dfg_baseline else '''## 9. Campos de validación independiente

Los mapas de campo y error usan los arrays de validación guardados. La malla es
independiente de los puntos Sobol empleados para entrenar.
'''),
        code('field-maps', fields),
        markdown('closure-note', '''## 10. Cierre, hashes y paquete de resultados

La celda final conserva decisiones, hashes, CSV y figuras en un archivo de resultados.
El verificador mantiene la autoridad sobre las métricas; ningún resultado queda marcado
como aceptado.
'''),
        code('closure', closure),
    ]
    if is_allen_cahn_exploration or (is_dfg_baseline and manifest.get('parameter_decisions_markdown')):
        cells.insert(6, code('parameter-table',
            "from IPython.display import Markdown, display\n"
            "import json, pandas as pd\n"
            "protocol=json.loads((BUNDLE_DIR/'manifest.json').read_text(encoding='utf-8'))\n"
            "display(Markdown(protocol['parameter_decisions_markdown']))\n"
            "display(pd.json_normalize(protocol['training']).T)\n"))
    elif manifest.get('parameter_decisions_markdown'):
        cells.insert(6, code('parameter-table',
            "from IPython.display import Markdown, display\n"
            "import json, pandas as pd\n"
            "protocol=json.loads((BUNDLE_DIR/'manifest.json').read_text(encoding='utf-8'))\n"
            "display(Markdown(protocol['parameter_decisions_markdown']))\n"
            "display(pd.json_normalize(protocol['training']).T)\n"))
    if is_dfg_baseline:
        sampling_cell = next(cell for cell in cells if cell['id'] == 'sampling-signals')
        cells.remove(sampling_cell)
        training_index = next(index for index, cell in enumerate(cells)
                              if cell['id'] == 'parallel-run')
        cells.insert(training_index + 1, sampling_cell)
    if manifest.get('comparison_design'):
        replacements={
            'overview': '# Burgers: Sobol frente a RAD, cinco semillas emparejadas\n\nPINA + PyTorch; RBF-PIKAN fija. Una tarea por T4, dos T4 en paralelo. Los identificadores 201–205 son controles y 1201–1205 son RAD; cada pareja utiliza la misma semilla real 201–205. W&B se conserva offline y se sincroniza desde el repositorio. Plotly muestra curvas logarítmicas y puntos adaptados. El test final permanece cerrado; no hay aceptación científica automática.',
            'hardware-note': '## 4. Dos T4 y diez tareas\n\nCada tarea ocupa una T4 y entrena únicamente Burgers. Las cinco parejas comparten inicialización, puntos iniciales y condiciones. La asignación real de GPU se registra; los IDs de tarea no son semillas independientes adicionales.',
            'sampling-note': '## 5. Puntos iniciales emparejados\n\nAmbos brazos comienzan con el mismo Sobol. RAD actualiza después solo el interior desde candidatos independientes usando el residual; los puntos de diagnóstico y evaluación no entran al muestreador.',
        }
        for cell in cells:
            if cell['id'] in replacements: cell['source']=replacements[cell['id']].splitlines(keepends=True)
            if cell['id']=='parallel-run':
                text=''.join(cell['source']).replace('from runtime import run_bundle',
                    "from runtime import run_bundle\nfrom src.research_pipeline.sciml.rad_campaign import assert_launch_ready\nassert_launch_ready(json.loads((BUNDLE_DIR/'manifest.json').read_text(encoding='utf-8')))")
                cell['source']=text.splitlines(keepends=True)
            if cell['id']=='sampling-signals':
                text=''.join(cell['source']).replace('from src.research_pipeline.sciml.sampling import pools',
                    'from src.research_pipeline.sciml.sampling import pools\nfrom src.research_pipeline.sciml.rad_campaign import job_spec')
                text=text.replace("data = pools(problem, seed,", "arm, paired_seed = job_spec(seed)\n        data = pools(problem, paired_seed,")
                text=text.replace("'problem': problem, 'seed': seed,", "'problem': problem, 'job_id': seed, 'paired_seed': paired_seed, 'arm': arm,")
                cell['source']=text.splitlines(keepends=True)
        cells.insert(7,code('parameter-table',"from IPython.display import Markdown, display\nimport json, pandas as pd\nprotocol=json.loads((BUNDLE_DIR/'manifest.json').read_text())\ndisplay(Markdown(protocol['parameter_decisions_markdown']))\ndisplay(pd.json_normalize(protocol['training']).T)\n"))
        adaptive='''import numpy as np, pandas as pd, plotly.express as px
adaptation_rows=[]
for job_dir in sorted(RUNS_DIR.glob('seed-*')):
    history=job_dir/'burgers/rad-history.jsonl'
    if not history.exists(): continue
    config=json.loads((job_dir/'config.json').read_text())['comparison']
    for line in history.read_text().splitlines(): adaptation_rows.append({**config,**json.loads(line)})
    final=sorted((job_dir/'burgers').glob('rad-*.npz'))[-1]
    with np.load(final) as saved:
        points=saved['points']
        figure=px.scatter(x=points[:,0],y=points[:,1],opacity=.3,
            labels={'x':'x','y':'t'},title=f"RAD final: paired seed {config['paired_seed']}")
        figure.write_html(ANALYSIS_DIR/f"rad-points-{config['paired_seed']}.html")
        display(figure)
adaptation_table=pd.DataFrame(adaptation_rows)
display(adaptation_table)
adaptation_table.to_csv(ANALYSIS_DIR/'adaptive-sampling.csv',index=False)
'''
        paired='''from scipy.stats import t as student_t
records=[]
for job_dir in sorted(RUNS_DIR.glob('seed-*')):
    config=json.loads((job_dir/'config.json').read_text())['comparison']
    metrics=json.loads((job_dir/'burgers/validation.json').read_text())['metrics']
    training=json.loads((job_dir/'burgers/training_summary.json').read_text())
    records.append({**config,**metrics,'seconds':training['training_and_diagnostics_sec'],
                    'steps':training['steps'],'stop_reason':training['stop_reason']})
paired_runs=pd.DataFrame(records)
display(paired_runs)
paired_runs.to_csv(ANALYSIS_DIR/'paired-runs.csv',index=False)
pairs=paired_runs.pivot(index='paired_seed',columns='arm',values='field_relative_l2')
if len(pairs)!=5 or pairs.isna().any().any():
    raise RuntimeError('Five complete pairs required; do not drop failures or replace seeds.')
if (pairs<=0).any().any(): raise RuntimeError('Nonpositive field error requires explicit analysis review.')
log_ratios=np.log(pairs['rad']/pairs['sobol'])
mean=log_ratios.mean(); margin=student_t.ppf(.975,4)*log_ratios.std(ddof=1)/np.sqrt(5)
summary=pd.DataFrame([{'independent_pairs':5,'geometric_mean_RAD_over_Sobol':np.exp(mean),
    'paired_t_95_lower':np.exp(mean-margin),'paired_t_95_upper':np.exp(mean+margin),
    'scientific_accepted':False}])
display(summary)
print('Small-sample parametric interval on paired log ratios; normality is uncertain at n=5. No automatic scientific acceptance.')
summary.to_csv(ANALYSIS_DIR/'paired-primary-summary.csv',index=False)
figure=px.line(paired_runs.sort_values(['paired_seed','arm']),x='arm',y='field_relative_l2',
    color='paired_seed',markers=True,log_y=True,title='Five paired seeds: independent validation L2')
figure.write_html(ANALYSIS_DIR/'paired-primary-log.html'); display(figure)
'''
        cells[-2:-2]=[markdown('adaptive-note','## 9a. Distribución adaptada y costo medido'),
                       code('adaptive-points',adaptive),markdown('paired-note','## 9b. Comparación por semillas emparejadas'),
                       code('paired-comparison',paired)]
    return {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            'language_info': {'name': 'python', 'version': '3'},
            'research': {
                'notebook_id': 'remote.' + spec['bundle_id'][:16],
                'role': notebook_role,
                'problem_id': manifest['iteration']['problem_id'],
                'model_id': manifest['iteration']['model_id'],
                'hypothesis_id': manifest['iteration']['hypothesis_id'],
                'source_of_truth': 'manifest',
            },
        },
        'cells': cells,
    }
