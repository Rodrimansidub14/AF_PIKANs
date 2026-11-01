"""Postprocess completed GPU runs. No training, tests, or artifact validators."""
from pathlib import Path
import json
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import t as student_t
import torch

OUT = Path(__file__).resolve().parent
CAMPAIGN = OUT.parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from src.research_pipeline.sciml.model import build_model
from evals.problems.heat_1d.reference import exact, eval_grid

RUNS = ROOT / 'var/agent_runs'
ARCH = RUNS / 'heat-burgers-architecture-screen/models'
EXT = CAMPAIGN / 'heat-ac-extension-r3/download/pikan-bcc43eb34dfe1060/runs'
AC = RUNS / 'allen-cahn-exp003/download/pikan-7d9fafd823f3e315/runs'
BURGERS = RUNS / 'burgers-rad-exp006-v4/download-attempts/attempt-01/pikan-4393e173686298ee'
R3 = CAMPAIGN / 'dfg-re20-gpu-r3/download-attempts/attempt-03/pikan-682b323105591987/runs'
R4 = CAMPAIGN / 'dfg-re20-gpu-r4/download-attempts/attempt-01/pikan-c349281ede3181b8/runs'
FEATFLOW = 'https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html'
DEEPXDE = 'https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html'
torch.set_num_threads(2)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def summarize(frame, group_cols, metrics):
    rows = []
    for keys, group in frame.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        labels = dict(zip(group_cols, keys))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors='coerce').dropna().to_numpy()
            n = len(values)
            if n == 0:
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else None
            half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
            rows.append({**labels, 'metric': metric, 'n': n, 'mean': mean,
                         'sample_sd': sd, 'ci95_t_low': mean-half if half is not None else None,
                         'ci95_t_high': mean+half if half is not None else None})
    return pd.DataFrame(rows)


def save_tables(stem, frame, summary, notes):
    frame.to_csv(OUT / (stem + '-by-seed.csv'), index=False)
    summary.to_csv(OUT / (stem + '-summary.csv'), index=False)
    write_json(stem + '-summary.json', {'scientific_accepted': False,
        'baseline_frozen': False, 'new_training': False, 'local_tests_run': False,
        'notes': notes, 'summary': records(summary), 'runs': records(frame)})


def load_model(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(payload['problem'], {'id': payload.get('model_id', 'rbf_fixed')}).double()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model, payload


def predict(model, x, time):
    points = np.column_stack((np.asarray(x).ravel(), np.asarray(time).ravel()))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(points), 2048):
            chunks.append(model(torch.as_tensor(points[start:start+2048], dtype=torch.float64)).numpy().reshape(-1))
    return np.concatenate(chunks)


def field_metrics(prediction, truth):
    wx = np.ones(prediction.shape[0]); wx[[0, -1]] = 0.5
    wt = np.ones(prediction.shape[1]); wt[[0, -1]] = 0.5
    weights = wx[:, None] * wt[None, :]
    error = prediction - truth
    return {'field_relative_l2': float(np.sqrt(np.sum(weights*error**2)/np.sum(weights*truth**2))),
            'field_relative_l2_unweighted': float(np.linalg.norm(error)/np.linalg.norm(truth)),
            'field_max_abs': float(np.max(np.abs(error))),
            'field_rmse_trapezoidal': float(np.sqrt(np.sum(weights*error**2)/np.sum(weights)))}


def save_figure(fig, filename):
    fig.update_layout(template='plotly_white', font=dict(family='Arial', size=14), margin=dict(t=75, b=65))
    fig.write_html(OUT / filename, include_plotlyjs=True, full_html=True)
    return fig


def markdown_table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.7g}'
        return str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def heat():
    xs, ts = eval_grid(nx=512, nt=256)
    truth = exact(xs, ts).reshape(512, 256)
    jobs = []
    for model_id in ['rbf_fixed', 'mlp_tanh', 'bspline_cubic']:
        for seed in [101, 102, 103]:
            jobs.append((model_id, 'EXP-0005', seed, ARCH/model_id/'download-extracted-bundle-logscale-v1/runs'/f'seed-{seed}'/'heat'))
    for seed in [104, 105]:
        jobs.append(('rbf_fixed', 'extension-r3', seed, EXT/f'seed-{seed}'/'heat'))
    rows = []
    for model_id, cohort, seed, folder in jobs:
        train = read(folder/'training_summary.json')
        checkpoint = folder/f"checkpoint-{train['steps']:06d}.pt"
        model, payload = load_model(checkpoint)
        pred = predict(model, xs, ts).reshape(512, 256)
        old = read(folder/'validation.json')['metrics']
        rows.append({'problem': 'Heat', 'model_id': model_id, 'cohort': cohort, 'seed': seed,
            'steps': payload['step'], 'interior_points': train['sampling_signals']['interior_count'],
            'nx': 512, 'nt': 256, **field_metrics(pred, truth),
            'original_grid_relative_l2': old['field_relative_l2'],
            'stop_reason': train['stop_reason'],
            'training_and_diagnostics_sec': train['training_and_diagnostics_sec'],
            'checkpoint': relative(checkpoint)})
        print('Heat', model_id, cohort, seed, rows[-1]['field_relative_l2'], flush=True)
    frame = pd.DataFrame(rows)
    summary = summarize(frame, ['model_id', 'cohort'], ['field_relative_l2','field_max_abs','training_and_diagnostics_sec'])
    notes = ('Error relativo L2 con cuadratura trapezoidal en la malla protegida 512×256, solución analítica, inferencia float64. '
        'Corrección del informe anterior: las dos cohortes Heat usaron 5,000 pasos y 4,096 puntos interiores; '
        'los 20,000 pasos eran el override de Burgers. Se conservan separadas: los hashes archivados de model.py, '
        'train.py y sampling.py difieren y esta entrega no ejecuta una auditoría de equivalencia ni validaciones locales. '
        'Los tres comparadores de EXP-0005 comparten los topes del protocolo, pero B-spline alcanza el tope temporal antes del de pasos; '
        'la tabla registra los pasos y segundos realmente realizados. Ningún entrenamiento nuevo.')
    save_tables('heat', frame, summary, notes)
    fig = go.Figure()
    for (model_id, cohort), group in frame.groupby(['model_id','cohort'], sort=False):
        fig.add_trace(go.Scatter(x=group.seed.astype(str), y=group.field_relative_l2, mode='markers',
            marker=dict(size=11), name=f'{model_id} · {cohort}'))
    fig.update_layout(title='Heat · error de campo en 512×256', xaxis_title='Semilla', yaxis_title='L2 relativo (trapezoidal)', yaxis_type='log')
    return {'by_seed': frame, 'summary': summary, 'notes': notes, 'figure': save_figure(fig, 'heat.html')}


def burgers():
    frame = pd.read_csv(BURGERS/'analysis/paired-runs.csv')
    frame['problem'] = 'Burgers'
    frame['source'] = relative(BURGERS/'analysis/paired-runs.csv')
    metrics = ['field_relative_l2', 'front_field_rmse_abs', 'complement_field_rmse_abs', 'residual_rmse_abs', 'seconds']
    summary = summarize(frame, ['arm'], metrics)
    paired = frame.pivot(index='paired_seed', columns='arm', values='field_relative_l2')
    paired['rad_minus_sobol'] = paired.rad - paired.sobol
    paired['rad_over_sobol'] = paired.rad / paired.sobol
    paired.to_csv(OUT/'burgers-pairs.csv')
    write_json('burgers-paired-effect.json', {'n': len(paired),
        'geometric_rad_over_sobol': float(np.exp(np.log(paired.rad_over_sobol).mean())),
        'pairs_rad_worse': int((paired.rad > paired.sobol).sum()),
        'mean_difference': float(paired.rad_minus_sobol.mean()), 'scientific_accepted': False})
    notes = ('Cinco pares existentes, 20,000 pasos por corrida; baseline Sobol fijo y RAD como contraste adverso. '
        'Se reutiliza la evaluación canónica Cole–Hopf de 1025×129 del experimento, sin reentrenamiento. '
        'RAD empeora en los cinco pares; esto solo descarta esta configuración, no toda focalización.')
    save_tables('burgers', frame, summary, notes)
    fig = go.Figure()
    for arm, group in frame.groupby('arm'):
        fig.add_trace(go.Scatter(x=group.paired_seed.astype(str), y=group.field_relative_l2, mode='lines+markers', name=arm))
    fig.update_layout(title='Burgers · cinco pares Sobol/RAD', xaxis_title='Semilla pareada', yaxis_title='L2 relativo', yaxis_type='log')
    return {'by_seed': frame, 'summary': summary, 'notes': notes, 'figure': save_figure(fig, 'burgers.html')}


def crossings(x, u, level, side):
    selection = x < 0 if side == 'left' else x > 0
    xx = x[selection]; yy = u[selection]-level
    idx = np.where(yy[:-1]*yy[1:] < 0)[0]
    roots = [float(xx[i]-yy[i]*(xx[i+1]-xx[i])/(yy[i+1]-yy[i])) for i in idx]
    roots.extend(float(xx[i]) for i in np.where(yy == 0)[0])
    return sorted(set(roots))


def interface(x, u, side):
    zero = crossings(x,u,0,side)
    low = crossings(x,u,-0.5,side); high = crossings(x,u,0.5,side)
    return {'zero_crossings':len(zero), 'position':zero[0] if len(zero)==1 else None,
        'width':abs(high[0]-low[0]) if len(low)==len(high)==1 else None}


def allen_cahn():
    mat = loadmat(CAMPAIGN/'references/Allen_Cahn.mat')
    x = mat['x'].ravel().astype(float); t = mat['t'].ravel().astype(float)
    truth = np.asarray(mat['u'], dtype=float)
    if truth.shape == (len(t),len(x)):
        truth = truth.T
    xx,tt = np.meshgrid(x,t,indexing='ij')
    ref_interp = RegularGridInterpolator((x,t),truth)
    rows=[]; interfaces=[]
    for seed in [101,102,103,104,105]:
        folder = (AC if seed<104 else EXT)/f'seed-{seed}'/'allen_cahn'
        checkpoint = folder/'checkpoint-020000.pt'
        model,payload = load_model(checkpoint)
        pred = predict(model,xx,tt).reshape(truth.shape)
        measured = field_metrics(pred,truth)
        original = read(folder/'validation.json')['metrics']
        with np.load(folder/'validation.npz') as old:
            ox,ot=np.meshgrid(old['x'],old['t'],indexing='ij')
            interpolated = ref_interp(np.column_stack((ox.ravel(),ot.ravel()))).reshape(ox.shape)
            old_l2 = field_metrics(old['prediction'],interpolated)['field_relative_l2']
        rows.append({'problem':'Allen–Cahn','seed':seed,'steps':payload['step'], 'nx':len(x),'nt':len(t),
            **measured, 'diagnostic_grid_l2_interpolated_reference':old_l2,
            'normalized_residual_mean':original['normalized_residual_mean'],
            'heldout_normalized_residual_rmse':original['heldout_normalized_residual_rmse'],
            'energy_decrease_fraction':-original['energy_relative_change'],
            'initial_trace_max_abs_error':original['initial_trace_max_abs_error'],
            'boundary_trace_max_abs_error':original['boundary_trace_max_abs_error'],
            'checkpoint':relative(checkpoint), 'reference_status':'published_provisional'})
        for time in [0.25,0.5,0.75,1.0]:
            tidx = int(np.argmin(abs(t-time)))
            for side in ['left','right']:
                a=interface(x,pred[:,tidx],side); b=interface(x,truth[:,tidx],side)
                interfaces.append({'seed':seed,'time':float(t[tidx]),'side':side,
                    **{'prediction_'+k:v for k,v in a.items()}, **{'reference_'+k:v for k,v in b.items()},
                    'position_abs_error':abs(a['position']-b['position']) if a['position'] is not None and b['position'] is not None else None,
                    'width_abs_error':abs(a['width']-b['width']) if a['width'] is not None and b['width'] is not None else None,
                    'reference_dx':float(x[1]-x[0]),
                    'width_definition':'distance between u=-0.5 and +0.5; null when levels not uniquely reached'})
        print('Allen–Cahn',seed,measured['field_relative_l2'],flush=True)
    frame=pd.DataFrame(rows); interface_frame=pd.DataFrame(interfaces)
    interface_frame.to_csv(OUT/'allen-cahn-interfaces.csv',index=False)
    summary=summarize(frame,['problem'],['field_relative_l2','field_relative_l2_unweighted',
        'diagnostic_grid_l2_interpolated_reference','field_max_abs','normalized_residual_mean',
        'heldout_normalized_residual_rmse','energy_decrease_fraction'])
    notes=('Cinco semillas. Error de campo provisional, calculado directamente con checkpoints en los 201×101 puntos publicados de DeepXDE. '
        'Se entregan L2 trapezoidal y L2 discreto sin pesos; el primero es el score mostrado. '
        'También se conserva el cálculo histórico en 257×129 con referencia interpolada, para no confundir mallas. '
        'Posición de interfaz: cruce u=0 por semiplano. Anchura: distancia entre cruces u=−0.5 y +0.5; '
        'un nivel no alcanzado o cruces múltiples producen null, no cero. Referencia con dx=0.01. '
        'La convergencia BDF y la procedencia del solver publicado siguen pendientes; no se desbloquea el evaluador protegido.')
    save_tables('allen-cahn',frame,summary,notes)
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Campo: L2 relativo provisional','Disminución de energía'))
    fig.add_trace(go.Scatter(x=frame.seed.astype(str),y=frame.field_relative_l2,mode='markers',name='L2',marker_size=11),row=1,col=1)
    fig.add_trace(go.Bar(x=frame.seed.astype(str),y=100*frame.energy_decrease_fraction,name='Energía (%)'),row=1,col=2)
    fig.update_yaxes(type='log',title_text='L2 relativo',row=1,col=1)
    fig.update_yaxes(title_text='Disminución (%)',range=[0,100],row=1,col=2)
    fig.update_layout(title='Allen–Cahn · cinco semillas · referencia provisional',showlegend=False)
    return {'by_seed':frame,'summary':summary,'notes':notes,'interfaces':interface_frame,'figure':save_figure(fig,'allen-cahn.html')}


def dfg():
    rows=[]
    provenance=read(CAMPAIGN/'dfg-re20-gpu-r4/analysis-five-seed/dfg_re20_five_seed_provenance.json')
    prior={r['seed']:r for r in provenance['run_records']}
    for seed in [301,302,303,304,305]:
        folder=(R3 if seed<303 else R4)/f'seed-{seed}'
        metrics=read(folder/'metrics.json')
        rows.append({'problem':'DFG Re20','seed':seed,'revision':'r3' if seed<303 else 'r4',
            **metrics,'wandb_url':prior[seed]['wandb_url'],'source':relative(folder/'metrics.json'),
            'harness_status':'posttraining_hash_failure' if seed<303 else 'completed',
            'gpu':prior[seed]['gpu'],'gpu_slot':prior[seed]['gpu_slot'],
            'gpu_utilization_mean_percent':prior[seed]['gpu_util_mean']})
    frame=pd.DataFrame(rows)
    metrics=['qoi/drag_coefficient','qoi/lift_coefficient','qoi/pressure_difference',
        'qoi_rel_error/drag_coefficient','qoi_abs_error/lift_coefficient','qoi_rel_error/pressure_difference',
        'heldout/global/continuity_rms','heldout/global/momentum_x_rms','heldout/global/momentum_y_rms',
        'heldout/near_cylinder/continuity_rms','heldout/near_cylinder/momentum_x_rms','heldout/near_cylinder/momentum_y_rms',
        'boundary/boundary/cylinder_velocity_rms_normalized','boundary/boundary/outlet_normal_traction_rms',
        'training/steps','training/seconds']
    summary=summarize(frame,['problem'],metrics)
    notes=('Semillas 301–302 recuperadas de r3 y 303–305 completadas en r4, 1,200 s nominales por semilla. '
        'Kaggle r4 terminó en ERROR al importar PyVista en la celda de visualización: '
        "ModuleNotFoundError: No module named 'IPython.core.guarded_eval'. "
        'El campaign.json registra tres semillas completed y la evaluación previamente archivada marca execution_valid=true. '
        'r3 no tiene cierre válido del harness; se preserva su condición de evidencia recuperada. '
        'No se repiten entrenamientos. El budget de r4 declara 2 semillas/3,000 GPU-s, contradictorio con training de 3 semillas/3,600 GPU-s; '
        'esta entrega conserva y señala ese defecto. El total nominal de la campaña es 6,000 GPU-s. '
        'No hay campo FEM independiente; el error de campo queda sin medir. Umbrales ausentes, scientific_accepted=false.')
    save_tables('dfg',frame,summary,notes)
    refs=[5.57953523384,0.010618948146,0.11752016697]
    qois=['qoi/drag_coefficient','qoi/lift_coefficient','qoi/pressure_difference']
    fig=make_subplots(rows=1,cols=3,subplot_titles=('Arrastre Cd','Sustentación Cl','Caída de presión Δp'))
    for i,(key,ref) in enumerate(zip(qois,refs),start=1):
        fig.add_trace(go.Scatter(x=frame.seed.astype(str),y=frame[key],mode='markers',marker_size=11,name=key,showlegend=False),row=1,col=i)
        fig.add_hline(y=ref,line_dash='dash',line_color='#a53832',annotation_text='FEATFLOW',row=1,col=i)
    fig.update_layout(title='DFG Re20 · cinco semillas recuperadas · QoIs y referencia')
    return {'by_seed':frame,'summary':summary,'notes':notes,'figure':save_figure(fig,'dfg.html')}


def consolidate():
    data={name:read(OUT/(name+'-summary.json')) for name in ['heat','burgers','allen-cahn','dfg']}
    entries=[]
    selections=[('heat','Heat / RBF · EXP-0005','field_relative_l2',{'model_id':'rbf_fixed','cohort':'EXP-0005'},'Analítica · 512×256'),
        ('heat','Heat / RBF · extensión','field_relative_l2',{'model_id':'rbf_fixed','cohort':'extension-r3'},'Analítica · 512×256'),
        ('burgers','Burgers / Sobol','field_relative_l2',{'arm':'sobol'},'Cole–Hopf · 1025×129'),
        ('allen-cahn','Allen–Cahn','field_relative_l2',{},'DeepXDE · 201×101 · provisional'),
        ('dfg','DFG / Cd','qoi/drag_coefficient',{},'FEATFLOW QoI'),
        ('dfg','DFG / Cl','qoi/lift_coefficient',{},'FEATFLOW QoI'),
        ('dfg','DFG / Δp','qoi/pressure_difference',{},'FEATFLOW QoI')]
    for source,label,metric,filters,ref in selections:
        row=next(s for s in data[source]['summary'] if s['metric']==metric and all(s.get(k)==v for k,v in filters.items()))
        entries.append({'baseline':label,'n':row['n'],'metric':metric,'mean':row['mean'],'sample_sd':row['sample_sd'],
            'ci95_t_low':row['ci95_t_low'],'ci95_t_high':row['ci95_t_high'],'reference':ref})
    table=pd.DataFrame(entries); table.to_csv(OUT/'baselines.csv',index=False)
    write_json('baselines.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scientific_accepted':False,'baseline_frozen':False,'local_validation_skipped':True,
        'gpu_training_this_turn':False,'results':records(table)})
    compact=table[['baseline','n','mean','sample_sd','reference']]
    report='# Baselines RBF-PIKAN — entrega de resultados\n\n'
    report+='Se recuperaron las corridas GPU y se ejecutaron cuatro notebooks de resultados. No se ejecutaron tests, smoke tests, auditorías de convergencia ni validadores de integridad locales en esta entrega. Las mediciones de campo son posprocesado de checkpoints existentes; no consumen entrenamiento GPU.\n\n'
    report+=markdown_table(compact)+'\n\nMedia ± desviación estándar muestral entre semillas. Los IC t del CSV son descriptivos, no gates de aceptación.\n\n'
    report+='## Qué cambió respecto al informe anterior\n\n'
    report+='1. **DFG ya no está en cola:** r4 produjo las tres semillas faltantes. El estado ERROR corresponde a la importación de PyVista/IPython en visualización, posterior al entrenamiento. Se recuperaron los resultados sin otro envío GPU.\n'
    report+='2. **Heat tenía un error documental:** ambas cohortes usaron 5,000 pasos y 4,096 puntos. La razón anterior para separarlas (20k frente a 5k) era incorrecta. Se mantienen separadas en esta entrega porque cambiaron versiones de los módulos de entrenamiento y no se auditó su equivalencia. Todos los checkpoints se puntuaron en 512×256.\n'
    report+='3. **Allen–Cahn ahora tiene campo e interfaz para cinco semillas:** el score usa directamente la malla publicada 201×101. El cálculo histórico interpolado se conserva como otra columna. La referencia sigue provisional.\n'
    report+='4. **Presupuesto r4 inconsistente:** budget declara dos semillas y 3,000 GPU-s; training y la autorización de continuación describen tres semillas y 3,600 GPU-s. Los archivos originales se preservan.\n\n'
    for name in ['heat','burgers','allen-cahn','dfg']:
        report+='## '+name+'\n\n'+data[name]['notes']+'\n\n'
        report+=f'[Notebook ejecutado]({name}-results.ipynb) · [Métricas por semilla]({name}-by-seed.csv) · [Resumen]({name}-summary.csv) · [Gráfica interactiva]({name}.html)\n\n'
    report+='## DFG frente a la referencia\n\n'
    ds=pd.DataFrame(data['dfg']['summary']).set_index('metric')
    report+=f"El error relativo de Cd es {100*ds.loc['qoi_rel_error/drag_coefficient','mean']:.2f}% y el de Δp {100*ds.loc['qoi_rel_error/pressure_difference','mean']:.2f}%. Para Cl se reporta el error absoluto medio {ds.loc['qoi_abs_error/lift_coefficient','mean']:.6f}, porque su referencia está cerca de cero. "
    report+=f'Las referencias Cd=5.57953523384, Cl=0.010618948146 y Δp=0.11752016697 proceden de [FEATFLOW]({FEATFLOW}). No es una reproducción precisa del benchmark.\n\n'
    report+=f"El tiempo de entrenamiento DFG observado suma {sum(r['training/seconds'] for r in data['dfg']['runs']):.3f} GPU-s (nominal: 6,000). Los 0.158 s aproximados adicionales corresponden al cierre de pasos; no se informa el nominal como tiempo medido.\n\n"
    report+='Las tablas conservan W&B y rutas de origen por semilla. Los hashes y veredictos históricos no se regeneraron. [Log de Kaggle](dfg-kaggle-logs.txt).\n\n'
    report+='## Alcance científico\n\n'
    report+='Entrega operativa de baselines exploratorios. `scientific_accepted=false`, `baseline_frozen=false`. Sigue pendiente una referencia convergida para Allen–Cahn y un campo FEM independiente para DFG; Heat conserva cohortes separadas y los comparadores de arquitectura tienen n=3. La evidencia no sostiene SOTA.\n\n'
    report+=f'La referencia de Allen–Cahn es el archivo publicado por [DeepXDE]({DEEPXDE}), archivado por la campaña con commit y hash; su documentación no especifica solver ni tolerancias de generación. No se modifica el evaluador para desbloquear métricas.\n'
    (OUT/'baselines-report.md').write_text(report,encoding='utf-8')
    (OUT/'analysis_summary.md').write_text(report,encoding='utf-8')
    shutil.copy2(CAMPAIGN/'dfg-re20-gpu-r4/kaggle-logs-retrieved.txt',OUT/'dfg-kaggle-logs.txt')
    html='<!doctype html><html lang="es"><meta charset="utf-8"><title>Baselines RBF-PIKAN</title><style>body{font-family:system-ui;margin:3rem auto;max-width:1150px;color:#162c3d}table{border-collapse:collapse;width:100%}td,th{padding:.7rem;border-bottom:1px solid #d4dde4;text-align:left}h1{font-size:2.3rem}a{color:#146c91}iframe{width:100%;height:500px;border:0}p{line-height:1.6}.note{background:#edf4f8;padding:1rem}</style><h1>Baselines RBF-PIKAN</h1><p>27 de septiembre de 2026 · Resultados de GPU recuperados y notebooks de análisis ejecutados.</p>'
    html+='<p class="note">Baselines exploratorios. Allen–Cahn usa una referencia provisional; DFG conserva los fallos de cierre y visualización. Media y desviación estándar entre semillas. Sin aceptación científica.</p>'
    html+=compact.to_html(index=False,float_format=lambda v:f'{v:.7g}',border=0)
    html+='<p><a href="baselines-report.md">Informe completo</a> · <a href="baselines.csv">CSV consolidado</a> · <a href="baselines-results.ipynb">Notebook consolidado</a></p>'
    for name in data:
        html+=f'<h2>{name}</h2><p>{data[name]["notes"]}</p><p><a href="{name}-results.ipynb">Notebook ejecutado</a> · <a href="{name}-by-seed.csv">CSV por semilla</a></p><iframe src="{name}.html" title="{name}"></iframe>'
    (OUT/'index.html').write_text(html+'</html>',encoding='utf-8')
    return table,report
