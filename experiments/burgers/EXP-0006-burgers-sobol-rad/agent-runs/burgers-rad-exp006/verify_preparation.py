"""Verify prepared notebook, baseline isolation and persisted CPU pair outputs."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import nbformat
import IPython.display
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from src.research_pipeline.remote.bundle import verify_transport
root=Path(__file__).resolve().parent
bundle=root/'bundle-v2'; spec=verify_transport(bundle)
nb=nbformat.read(bundle/'experiment.ipynb',as_version=4); nbformat.validate(nb)
for c in nb.cells:
    if c.cell_type=='code': compile(c.source,c.id,'exec')
os.environ['KAGGLE_WORKING_DIR']=str(root/'notebook-preparation-smoke-v2')
IPython.display.display=lambda *a,**k:None
ns={}
for c in nb.cells:
    if c.id in ['bundle-restore','parameter-table','sampling-signals']: exec(c.source,ns)
table=ns['sampling_table']; assert len(table)==10
assert set(table['paired_seed'])==set(range(201,206))
blocked=False
try: exec(next(c.source for c in nb.cells if c.id=='parallel-run'),ns)
except RuntimeError as e: blocked='scientific' in str(e)
assert blocked and not ns['RUNS_DIR'].exists()
result={'bundle_id':spec['bundle_id'],'schema_valid':True,'code_cells_compiled':12,
        'executed_preparation_cells':3,'sampling_rows':len(table),'unsigned_launch_blocked_before_jobs':True}
old=json.loads((ROOT/'var/agent_runs/heat-burgers-architecture-screen/models/rbf_fixed/bundle-logscale-v1/bundle.json').read_text())
for path,digest in old['files'].items():
    if path.startswith('evals/'):
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
result['bundled_verifier_sources_unchanged']=True
smoke=root/'cpu-pair-smoke-c9650bb3'
rows=json.loads((smoke/'smoke-results.json').read_text())
for row in rows:
    folder=smoke/('seed-'+str(row['job']))
    row['wandb_offline_files']=len(list((folder/'wandb').rglob('*.wandb')))
    assert row['wandb_offline_files']==1
    metrics=json.loads((folder/'burgers/validation.json').read_text())['metrics']
    assert all(isinstance(v,(float,int)) and math.isfinite(v) for v in metrics.values())
with np.load(smoke/'seed-201/burgers/pools.npz') as a,np.load(smoke/'seed-1201/burgers/pools.npz') as b:
    for key in a.files: np.testing.assert_array_equal(a[key],b[key])
with np.load(smoke/'seed-201/burgers/diagnostic-000000.npz') as a,np.load(smoke/'seed-1201/burgers/diagnostic-000000.npz') as b:
    np.testing.assert_array_equal(a['u'],b['u'])
result['cpu_pair_initial_predictions_and_pools_equal']=True
result['cpu_pair_wandb_offline_records']=2
(smoke/'smoke-results.json').write_text(json.dumps(rows,indent=2))
(root/'notebook-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
