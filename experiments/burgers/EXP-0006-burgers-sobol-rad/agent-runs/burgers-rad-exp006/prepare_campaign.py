"""Prepare an unsigned, reviewable EXP-0006 bundle; never submit or sign it."""
import copy
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.research_pipeline.remote.bundle import prepare

BASE=ROOT/'var/agent_runs/heat-burgers-architecture-screen/models/rbf_fixed'
OUT=ROOT/'var/agent_runs/burgers-rad-exp006'
parser=argparse.ArgumentParser()
parser.add_argument('--bundle-name',default='bundle-v1')
args=parser.parse_args()
if Path(args.bundle_name).name!=args.bundle_name: raise ValueError('bundle-name must be a directory name')
EXP=ROOT/'experiments/EXP-0006-burgers-sobol-rad'
EXP.mkdir(parents=True,exist_ok=True)
metrics={}; resources=[]; hashes={}
for seed in (101,102,103):
    run=BASE/f'download-extracted-bundle-logscale-v1/runs/seed-{seed}/burgers'
    for name in ('validation.json','validation.npz','training_summary.json'):
        p=run/name; hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    report=json.loads((run/'validation.json').read_text())
    if report['final_test_used']: raise RuntimeError('unexpected final test access')
    metrics[str(seed)]=report['metrics']
    resources.append(json.loads((run/'training_summary.json').read_text()))
keys=['field_relative_l2','front_field_rmse_normalized','residual_rmse_abs',
      'boundary_max_abs_error','initial_max_abs_error']
aggregate={key:{'mean':float(np.mean([m[key] for m in metrics.values()])),
                'std':float(np.std([m[key] for m in metrics.values()],ddof=1)),
                'max':float(max(m[key] for m in metrics.values()))} for key in keys}
proposal={'status':'proposed_not_frozen','scope':'EXP-0005 canonical Burgers independent validation only',
          'baseline_id_proposed':'burgers_pinnacle_1d@rbf_fixed-exp005-validation-v1',
          'scientific_accepted':False,'final_test_used':False,'seeds':metrics,'aggregate':aggregate,
          'source_sha256':hashes,'proposed_physical_upper_envelope':{
              k:aggregate[k]['max'] for k in keys if k!='field_relative_l2'},
          'proposed_minimum_effect':{'relative_reduction':.2,
              'paired_geometric_error_ratio_upper':.8,
              'historical_mean_relative_l2_times_0_8':.8*aggregate['field_relative_l2']['mean'],
              'rationale':'Project proposal: 20% practical improvement. Factor is a human policy choice, not a literature optimum; absolute value derived from measured pilot mean. Not an active verifier threshold.'},
          'limitations':'n=3 pilot; maxima are descriptive guard proposals, not justified physical tolerances or frozen verifier thresholds'}
(OUT/'baseline-freeze-proposal.json').write_text(json.dumps(proposal,indent=2),encoding='utf-8')
review='# EXP-0006 — expediente concreto para firma científica\n\n'
review+='La ejecución del plan ya está autorizada. Pendiente: decidir congelar el snapshot de validación EXP-0005 y firmar este pre-registro; no se solicita aceptar un resultado futuro.\n\n'
review+='| Criterio propuesto | Valor | Alcance |\n|---|---:|---|\n'
review+='| Razón geométrica emparejada L2_RAD/L2_Sobol | <=0.8 | Mejora práctica20%, propuesta a firmar |\n'
review+=f"| Referencia absoluta0.8*media piloto | {aggregate['field_relative_l2']['mean']*.8:.8g} | Descriptiva; no segundo endpoint |\n"
for key,value in proposal['proposed_physical_upper_envelope'].items():
    review+=f'| {key} | <={value:.8g} | Máximo observado entre3 semillas piloto |\n'
review+='\nSe propone revisar estos guardas por semilla, sin compensar un fallo físico con otra semilla mejor. Son envolventes piloto, no tolerancias físicas universales. '
review+='Reporte principal: cinco razones individuales, media logarítmica y CI t95%; sin aceptación automática. Un intervalo inconcluso no demuestra equivalencia.\n\n'
review+='Semillas reales201–205; diez tareas; dos T4; máximo20000 pasos o1200s (control entre operaciones), timeout exterior2100s/tarea. Hasta3.17h nominales de ejecución, sin cola. No ampliar a10 pares automáticamente.\n\n'
review+='Fuente vinculante: AGENTS.md:33 (firmar el pre-registro es humano), context/agent-policy.toml:124 (solo el investigador congela baselines y mueve umbrales). Ni evals ni claims fueron modificados. La propuesta incluye hashes y los tres scorecards en baseline-freeze-proposal.json.\n'
(OUT/'approval-package.md').write_text(review,encoding='utf-8')

decisions='''# EXP-0006: decisiones de parámetros — propuesta previa a corrida

| Decisión | Valor | Fundamento |
|---|---|---|
| Problema | Burgers canónico, nu=0.01/pi, x en [-1,1], t en [0,1], IC=-sin(pi*x), BC=0 | Contrato y verificador intactos de EXP-0005. |
| Modelo | RBF-PIKAN [2,16,16,1], 3648 parámetros, centros/anchos fijos | Aislar política de muestreo; no modificar capacidad. |
| Entrenamiento | PINA/PyTorch float64, Adam 0.001, batch 1024 | Igual que el baseline. Estado Adam se conserva al adaptar. |
| Interior | 8192 puntos iniciales Sobol, mismo pool por pareja | Conteo fijo; diagnóstico, IC, BC y validación independientes. |
| Frontera e IC | 512 por lado; 1024 en IC | Igual que EXP-0005, no se adaptan. |
| Semillas reales | 201,202,203,204,205 | Nuevas; cinco pares, diez runs, no diez réplicas por brazo. |
| RAD | p_i=0.5/M+0.5*abs(r_i)^2/sum(abs(r)^2) | RAD k=2,c=1 sobre pool finito; 50% masa uniforme limita pérdida de cobertura. Si residual cero, distribución uniforme. |
| Candidatos | 32768 Sobol independientes, stream=10*seed+7 | Cuatro veces el pool activo, soporte reproducible y costo acotado. Propuesta operativa, no hiperparámetro óptimo publicado. |
| Refresco | desde paso 2000; cada 1000; hasta antes del paso 20000 | Warmup 10% del presupuesto; como máximo 18 refrescos. Medir costo, no asumir ganancia. |
| Selección | con reemplazo, RNG=10*seed+8 | Variante discreta explícita de RAD; duplicados codifican probabilidad. Registrar puntos únicos/ESS; no atribuir reproducción exacta de la implementación original. |
| Corrección de importancia | ninguna | RAD cambia el objetivo espacial; no se presenta como estimador insesgado de la integral uniforme. |
| AD de candidatos | chunks de 512, float64, gradientes solo para evaluar residual | No backprop a pesos ni uso de referencia; no cambia el RNG de entrenamiento. |
| Presupuesto | 20000 pasos o 1200 s por brazo, verificado entre operaciones | Igual límite operativo, costo adaptativo incluido; una operación en curso puede sobrepasarlo. Timeout exterior2100s. Si algún brazo no completa pasos, contraste de pasos iguales inconcluso; reportar tiempo y pasos reales. |
| GPU | dos T4, máximo dos tareas concurrentes | Una tarea por GPU; cinco parejas. IDs 201–205 control y 1201–1205 RAD solo para scheduler. Orden intercalado y alternado; slot real registrado. |
| Timeout exterior | 2100 s por tarea; setup <=900 s | Límite nominal de cinco olas=10500 s más setup (~3.17 h), sin cola del proveedor. No promesa de tiempo exacto. |
| Diagnósticos | cada250 pasos, 256 puntos y 32 sensibilidades; métricas cada100 | Conserva instrumentación baseline; RAD añade snapshots, probabilidades, puntos, ESS y costo cada refresh. |
| W&B | offline en Kaggle, sync local posterior | No embebe secretos; un run y artefacto por brazo/problema/semilla. |
| Métrica primaria | log(L2_RAD/L2_Sobol) por semilla; media y CI t95% | Unidad independiente: pareja. n=5 no garantiza potencia ni normalidad; publicar valores individuales. |
| Secundarias | frente, IC/BC, residual, costo total, ejemplos vistos, cobertura | Un error global menor no basta para afirmar mejora física local. |
| Delta práctico propuesto | reducción20%: razón geométrica RAD/Sobol <=0.8; referencia absoluta=0.8*media piloto | Decisión práctica propuesta, no óptimo publicado ni cálculo de potencia. Requiere firma; valores en baseline-freeze-proposal.json. |
| Umbrales | propuesta separada derivada del snapshot EXP-0005; pendientes de congelación humana | No modifica evals ni convierte un screening en baseline aceptado. |
| Cierre | hashes, notebook con tablas/Plotly log, archivo de resultados, reporte de fallos | El test final no se abre; no genera claim accepted. |

Fuentes: [Wu et al., RAD/RAR-D](https://arxiv.org/abs/2207.10289); [vRBA, 2026](https://www.nature.com/articles/s44387-026-00084-4). Los parámetros operativos anteriores son decisiones del proyecto, no una afirmación de óptimo SOTA.
'''
(EXP/'parameter-decisions.md').write_text(decisions,encoding='utf-8')
spec='''# EXP-0006 — Burgers: Sobol frente a RAD

Estado: preparado, pre-registro pendiente de firma científica. Ejecución del plan autorizada por el investigador en conversación; ningún agente firma el gate científico ni congela el baseline.

Pregunta: ¿la concentración residual observada en EXP-0005 permite reducir el error de campo y del frente bajo el mismo presupuesto?
Hipótesis: la media por pareja de log(L2_RAD/L2_Sobol) es negativa; se refuta una mejora observada si es >=0. Un intervalo que abarque cero permanece inconcluso.
Métrica primaria: field_relative_l2 de evals/checkpoint_validation.py, problem=burgers_pinnacle_1d. Malla y Cole–Hopf existentes, sin cambios. Comparador Sobol concurrente; no sustituirlo por los tres seeds históricos.
Cinco semillas emparejadas 201–205. No usar puntos espaciales como réplicas, no reemplazar fallos, no ajustar parámetros después de observar resultados, no reusar seeds 101–103 como confirmación.
Los brazos difieren únicamente en la actualización del interior. Máximo 20000 pasos/1200 s cada uno; estimación CPU/GPU separada en presupuesto. No atribuir a RAD una comparación a igual número de pasos si un brazo agotó tiempo antes.

Criterios propuestos, NO activos: reducción práctica de20% en razón geométrica RAD/Sobol (<=0.8), con media histórica*0.8 como referencia absoluta reportada, no segundo endpoint; frente, residual e IC/BC dentro de la envolvente máxima medida de EXP-0005. El20% es una elección práctica a firmar, no un óptimo de la literatura ni un cálculo de potencia. Los valores concretos y hashes están en var/agent_runs/burgers-rad-exp006/baseline-freeze-proposal.json. Esa envolvente de tres semillas no representa tolerancia física universal y requiere decisión humana antes de convertirse en gate.
Tamaño: n=5 pares iniciales. Una ampliación a10 se propone únicamente si el intervalo es inconcluso y requiere protocolo nuevo con control de análisis secuencial; no mirar cinco y declarar significancia con diez como si fuera un solo análisis fijo.
Sin aceptación automática; test final cerrado. El verificador actual sigue devolviendo threshold_not_defined mientras no exista baseline congelado. No firmar manifest.approved_by por cuenta del investigador.

Autorización pendiente concreta: congelación del snapshot de validación (sin abrir test final) y firma de criterios/parametrización. Tras ella: bundle con hashes exactos, smoke remoto técnico si se necesita, diez trabajos, descarga/sync W&B, análisis emparejado, decisión humana. No lanzar Allen–Cahn ni DFG por herencia de un resultado de Burgers.

Artefactos preparados: manifiesto, tabla de decisiones, audit.html/localization.csv/hotspot-reproducibility.csv/cost-and-basis.csv, baseline-freeze-proposal.json, notebook reproducible en bundle-v1.
'''
(EXP/'spec.md').write_text(spec,encoding='utf-8')

manifest=json.loads((BASE/'manifest.json').read_text())
manifest['iteration'].update(id='burgers-sobol-rad-exp006',experiment_id='EXP-0006',
    problem_id='burgers_pinnacle_1d',focalization_id='paired_sobol_rad',hypothesis_id='residual_localization',protocol_phase='exploratory')
manifest['hypothesis']={'statement':'Fixed RBF-PIKAN: compare fixed Sobol versus RAD, five paired seeds.',
    'expected_effect':'Lower independent field L2 and front error; quantify cost and physical guards.',
    'falsification_condition':'Nonnegative paired mean log error ratio refutes an observed benefit; nonfinite/missing runs invalidate confirmatory interpretation.'}
manifest['problem']={'sequence':['burgers'],'burgers':manifest['problem']['burgers']}
manifest['model']['implementation']='src/research_pipeline/sciml/rad_campaign.py'
jobs=[201,1201,1202,202,203,1203,1204,204,205,1205]
manifest['training'].update(seeds=jobs,sampling='paired_sobol_rad',problem_overrides={},rad={
    'candidate_points':32768,'chunk_size':512,'power':2,'uniform_fraction':.5,'warmup_steps':2000,'refresh_every':1000})
manifest['focalization']={'enabled':True,'bounded_edit_surface':['src/mechanisms/rad.py','src/mechanisms/rad_callback.py']}
manifest['comparison_design']={'paired_seeds':list(range(201,206)),'arms':['sobol','rad'],
    'job_ids_are_not_random_seeds':True,'primary':'paired_log_field_relative_l2_ratio',
    'preregistration_status':'awaiting_human_scientific_signature','launch_ready':False}
manifest['tracking']['group']='burgers-sobol-rad-EXP-0006'
manifest['artifacts']['expected']=[x for x in manifest['artifacts']['expected'] if not x.startswith('heat/')]
manifest['artifacts']['expected'].append('effective-manifest.json')
manifest['acceptance_gates']['metrics']={f'burgers/{k}':{'direction':'minimize','required':True} for k in keys}
manifest['parameter_decisions_markdown']=decisions
manifest['research_status']={'execution_authorized':True,'scientific_preregistration_signed':False,
    'baseline_frozen':False,'scientific_accepted':False,'submission_allowed':False}
manifest_path=EXP/'manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
source_spec=json.loads((BASE/'bundle-logscale-v1/bundle.json').read_text())
files=[p for p in source_spec['files'] if p.startswith(('src/','evals/'))]
files+=['src/research_pipeline/sciml/rad_campaign.py','src/mechanisms/__init__.py','src/mechanisms/base.py',
        'src/mechanisms/baseline_none.py','src/mechanisms/rad.py','src/mechanisms/rad_callback.py']
destination=OUT/args.bundle_name
result=prepare(manifest_path,destination,'rodrigomansilla/af-pikans-burgers-sobol-rad-exp006',sorted(set(files)),2100,'research',True)
budget={'historical_rbf_burgers_seconds':[r['training_and_diagnostics_sec'] for r in resources],
    'training_cap_seconds_per_arm':1200,'jobs':10,'parallel_gpus':2,'total_training_cap_gpu_hours':12000/3600,
    'max_worker_slot_hours_including_validation':21000/3600,'nominal_wall_upper_hours_excluding_queue':11400/3600,
    'rad_extra_residual_evaluations_per_job_max':18*32768,
    'note':'Wall upper assumes both T4s stay available; provider queue/install failures not predicted. GPU timing of RAD not yet measured.'}
(OUT/'budget.json').write_text(json.dumps(budget,indent=2),encoding='utf-8')
(OUT/'preparation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'bundle':str(destination),'manifest':str(manifest_path),'budget':budget,'launch_ready':False},indent=2))
