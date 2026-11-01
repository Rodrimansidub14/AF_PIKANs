"""Small local technical smoke; no scientific thresholds or remote run."""
import json
import os
from pathlib import Path
import sys
import uuid
import math
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from src.research_pipeline.sciml.rad_campaign import job_spec
from src.research_pipeline.sciml import campaign

parent=ROOT/'var/agent_runs/burgers-rad-exp006'/('cpu-pair-smoke-'+uuid.uuid4().hex[:8])
parent.mkdir()
manifest=json.loads((ROOT/'experiments/EXP-0006-burgers-sobol-rad/manifest.json').read_text())
manifest['training'].update(accelerator='cpu',interior_points=64,boundary_points=16,
    batch_size=16,steps_per_problem=8,seconds_per_problem=120,diagnostic_points=16,
    sensitivity_points=2,diagnostic_every=4,metric_log_every=2,rad={
    'candidate_points':128,'chunk_size':32,'warmup_steps':2,'refresh_every':2,'power':2,'uniform_fraction':.5})
manifest['iteration']['id']='EXP006-technical-cpu-smoke'
manifest['tracking']['group']='EXP006-technical-cpu-smoke-not-scientific'
source=parent/'smoke-manifest.json'; source.write_text(json.dumps(manifest))
os.environ['HARNESS_MANIFEST']=str(source)
results=[]
for job in (201,1201):
    # Invoke the ordinary CPU trainer for a bounded technical smoke, never the
    # scientifically gated GPU job adapter and never forge a signed manifest.
    arm,seed=job_spec(job)
    out=parent/f'seed-{job}'; out.mkdir()
    manifest['comparison']={'job_id':job,'arm':arm,'paired_seed':seed}
    manifest['training']['sampling']='rad' if arm=='rad' else 'scrambled_sobol_fixed'
    effective=out/'effective-manifest.json'; effective.write_text(json.dumps(manifest))
    os.environ['HARNESS_SEED']=str(seed)
    sys.argv=[sys.argv[0],'--manifest',str(effective),'--out',str(out)]
    campaign.main()
    cfg=json.loads((out/'config.json').read_text()); metrics=json.loads((out/'burgers/validation.json').read_text())
    results.append({'job':job,'comparison':cfg['comparison'],'steps':metrics['step'],
                    'validation_finite':all(isinstance(v,(int,float)) and math.isfinite(v) for v in metrics['metrics'].values()),
                    'log_figures_present':all((out/'burgers'/f).is_file() for f in ['training-losses-log.html','training-signals-log.html']),
                    'wandb_offline_files':len(list((out/'wandb').rglob('*.wandb'))),
                    'scientific_evidence':False})
(parent/'smoke-results.json').write_text(json.dumps(results,indent=2))
print(json.dumps({'output':str(parent),'results':results},indent=2))
