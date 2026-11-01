"""Read-only status helper for the EXP-0006 Kaggle rerun."""
from pathlib import Path
import json, os, sys
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0,str(ROOT))
for line in (ROOT/'.env').read_text(encoding='utf-8-sig').splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        key,value=line.split('=',1); key=key.strip()
        if key in {'KAGGLE_API_TOKEN','KAGGLE_USERNAME'}:
            os.environ[key]=value.strip().strip('"\'')
os.environ['KAGGLE_CONFIG_DIR']=str(ROOT/'var/agent_runs/kaggle-config')
from src.research_pipeline.remote.bundle import verify_transport
from src.research_pipeline.remote.kaggle import status
BUNDLE=ROOT/'var/agent_runs/burgers-rad-exp006-v4/bundle-v6-gpu-20260926'
if len(sys.argv)!=2 or sys.argv[1]!='status': raise SystemExit('usage: status_exp006_v4.py status')
verify_transport(BUNDLE)
print(json.dumps(status(BUNDLE)))
