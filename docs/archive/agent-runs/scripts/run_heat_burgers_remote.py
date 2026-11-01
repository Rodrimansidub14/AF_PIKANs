"""Local coordinator. Credentials are read into process memory, never bundled."""
from pathlib import Path
import sys,os,json,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
for line in (ROOT/'.env').read_text(encoding='utf-8-sig').splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        key,value=line.split('=',1)
        if key.strip() in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','WANDB_API_KEY','WANDB_ENTITY','WANDB_PROJECT'):
            os.environ[key.strip()]=value.strip().strip('\"\x27')
from src.research_pipeline.remote.bundle import prepare,verify_transport
from src.research_pipeline.remote.kaggle import submit,status,fetch,invoke
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','submit','status','fetch','check']);args=parser.parse_args()
base=ROOT/'var/agent_runs/heat-burgers-001'
bundle=base/'bundle'
if args.action=='check':
    reply=invoke(['kernels','list','--mine','--page-size','1','--csv'])
    print(json.dumps({'authenticated_listing': 'Authentication required' not in reply and ('ref' in reply or 'No kernels found' in reply)}))
elif args.action=='prepare':
    files=['src/research_pipeline/sciml/'+name+'.py' for name in ('__init__','model','sampling','train','campaign')]
    files+=['evals/__init__.py','evals/checkpoint_validation.py','evals/problems/__init__.py','evals/problems/_grids.py',
            'evals/problems/heat_1d/__init__.py','evals/problems/heat_1d/reference.py',
            'evals/problems/burgers_1d/__init__.py','evals/problems/burgers_1d/reference.py']
    spec=prepare(ROOT/'experiments/EXP-0002-heat-burgers-sobol/manifest.json',bundle,
        os.environ['KAGGLE_USERNAME']+'/af-pikans-heat-burgers-sobol-001',files,timeout=2100,gpu=True)
    # Records the user's explicit combined-run request; not a scientific acceptance.
    approval={'bundle_id':spec['bundle_id'],'approved_by':'Rodrigo Mansilla',
              'scope':'approved','hypothesis':'approved','code_surface':'approved','remote_execution':True,
              'authorization_source':'Conversation: Haz heat y Burguers en un mismo run; usa wandb; documentar toda decision de parametros.',
              'interpretation':'User requested point/step tradeoff avoiding 8+ hours. Budget v2: 3 seeds, Heat max5000 steps/300s, Burgers max20000 steps/1200s; max105min runtime +15min setup. No claim or baseline freeze.'}
    (base/'approval.json').write_text(json.dumps(approval,indent=2),encoding='utf-8')
    print(json.dumps({'bundle_id':spec['bundle_id'],'kernel':spec['kernel'],'files':len(files),'outer_timeout_sec_per_seed':2100}))
elif args.action=='submit': print(json.dumps(submit(bundle,base/'approval.json')))
elif args.action=='status': print(json.dumps(status(bundle)))
elif args.action=='fetch': print(json.dumps(fetch(bundle,base/'download')))
