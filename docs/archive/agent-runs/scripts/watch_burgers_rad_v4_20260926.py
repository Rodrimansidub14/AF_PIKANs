"""Persistent EXP-0006 watcher: poll Kaggle, fetch with retries, validate, sync W&B."""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time
from datetime import datetime, timezone
ROOT=pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0,str(ROOT))
for line in (ROOT/'.env').read_text(encoding='utf-8-sig').splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        key,value=line.split('=',1); key=key.strip()
        if key in {'KAGGLE_API_TOKEN','KAGGLE_USERNAME','WANDB_API_KEY','WANDB_ENTITY','WANDB_PROJECT'}:
            os.environ[key]=value.strip().strip('"\'')
os.environ['KAGGLE_CONFIG_DIR']=str(ROOT/'var/agent_runs/kaggle-config')
from src.research_pipeline.remote.bundle import verify_transport
from src.research_pipeline.remote.evaluation import evaluate
from src.research_pipeline.remote.postprocess import sync_offline_wandb
from src.research_pipeline.remote.runtime import read_json, write_json
import src.research_pipeline.remote.kaggle as kaggle
BUNDLE=ROOT/'var/agent_runs/burgers-rad-exp006-v4/bundle-v6-gpu-20260926'
CAMPAIGN=ROOT/'var/agent_runs/burgers-rad-exp006-v4'
OUT=CAMPAIGN/'download-attempts'
STATE=CAMPAIGN/'watcher-status.json'
PYTHON=ROOT/'var/agent_runs/remote-env/Scripts/python.exe'


def stamp(): return datetime.now(timezone.utc).isoformat()
def save(payload):
    payload['updated_at']=stamp()
    write_json(STATE,payload)
def output_invoke(arguments,timeout_sec=90):
    # Kaggle's prior 20-minute output request timed out on a partial 151 MB transfer.
    budget=7200 if arguments[:2]==['kernels','output'] else timeout_sec
    return _original_invoke(arguments,timeout_sec=budget)
_original_invoke=kaggle.invoke
kaggle.invoke=output_invoke


def main():
    spec=verify_transport(BUNDLE)
    base={'bundle_id':spec['bundle_id'],'kernel':spec['kernel'],'poll_interval_sec':60}
    save({**base,'status':'watching'})
    consecutive_errors=0
    while True:
        try:
            provider=kaggle.status(BUNDLE)['provider_status']
            consecutive_errors=0
            save({**base,'status':'provider','provider_status':provider})
        except Exception as exc:
            consecutive_errors+=1
            save({**base,'status':'status_error','error_type':type(exc).__name__,
                  'consecutive_errors':consecutive_errors})
            time.sleep(60)
            continue
        upper=provider.upper()
        if any(term in upper for term in ('COMPLETE','ERROR','CANCEL','FAILED','STOPPED')):
            break
        time.sleep(60)
    save({**base,'status':'terminal','provider_status':provider})
    OUT.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,4):
        destination=OUT/f'attempt-{attempt:02d}'
        save({**base,'status':'fetching','provider_status':provider,
              'fetch_attempt':attempt,'fetch_timeout_sec':7200})
        try:
            result=kaggle.fetch(BUNDLE,destination)
            runs=pathlib.Path(result['runs'])
            sync=sync_offline_wandb(runs,python_executable=str(PYTHON),timeout_sec=1200)
            write_json(CAMPAIGN/'wandb-sync.json',sync)
            save({**base,'status':'fetched' if result['execution_valid'] else 'fetched_invalid',
                  'provider_status':provider,'fetch_attempt':attempt,**result,
                  'wandb_sync':{'status':sync.get('status'),'run_count':sync.get('run_count'),
                                'synced_count':sync.get('synced_count')}})
            return
        except Exception as exc:
            # A timeout may still have downloaded a complete tree; validate before retrying.
            if isinstance(exc,subprocess.TimeoutExpired):
                for campaign_file in destination.rglob('campaign.json') if destination.exists() else []:
                    try:
                        campaign_data=read_json(campaign_file)
                        if campaign_data.get('bundle_id')!=spec['bundle_id']: continue
                        report=evaluate(BUNDLE,campaign_file.parent)
                        if report.get('execution_valid'):
                            write_json(destination/'evaluation.json',report)
                            runs=str(campaign_file.parent)
                            sync=sync_offline_wandb(pathlib.Path(runs),python_executable=str(PYTHON),timeout_sec=1200)
                            write_json(CAMPAIGN/'wandb-sync.json',sync)
                            save({**base,'status':'fetched','provider_status':provider,
                                  'fetch_attempt':attempt,'runs':runs,'execution_valid':True,
                                  'report':str(destination/'evaluation.json'),
                                  'transport_warning':'CLI timed out after complete artifacts passed local validation',
                                  'wandb_sync':{'status':sync.get('status'),'run_count':sync.get('run_count'),
                                                'synced_count':sync.get('synced_count')}})
                            return
                    except Exception:
                        pass
            save({**base,'status':'fetch_retry','provider_status':provider,
                  'fetch_attempt':attempt,'error_type':type(exc).__name__})
            time.sleep(min(300,30*attempt))
    save({**base,'status':'fetch_error','provider_status':provider,
          'error':'Three download attempts failed; local partial artifacts were retained.'})

if __name__=='__main__': main()
