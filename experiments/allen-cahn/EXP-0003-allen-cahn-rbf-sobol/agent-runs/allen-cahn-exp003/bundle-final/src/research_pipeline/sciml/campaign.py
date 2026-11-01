"""Heat then Burgers, one campaign, separate W&B runs and artifacts."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import hashlib
import importlib.metadata
import argparse
import csv
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from src.research_pipeline.sciml.train import train,json_write
from src.research_pipeline.sciml.model import build_model
from src.research_pipeline.remote.reporting import log_wandb_figures,signal_unit
from src.research_pipeline.sciml.allen_cahn_evaluation import evaluate_allen_cahn


def assert_exploratory_launch_ready(manifest):
    """Fail closed unless an explicitly budgeted Allen–Cahn signal pilot is declared."""
    iteration = manifest.get('iteration', {})
    status = manifest.get('research_status', {})
    budget = manifest.get('budget', {})
    problem = manifest.get('problem', {}).get('allen_cahn', {})
    if iteration.get('experiment_id') != 'EXP-0003' or iteration.get('protocol_phase') != 'exploratory':
        raise RuntimeError('Only the preregistered EXP-0003 exploratory phase may use this runner')
    if status.get('execution_authorized') is not True:
        raise RuntimeError('Allen–Cahn exploratory execution is not authorized')
    if budget.get('approved') is not True:
        raise RuntimeError('Allen–Cahn exploratory budget is not approved')
    if status.get('scientific_accepted') is not False:
        raise RuntimeError('Exploratory Allen–Cahn runs cannot be scientifically accepted')
    if problem.get('field_error_status') != 'locked_pending_reference':
        raise RuntimeError('Allen–Cahn field error must remain locked pending reference convergence')


def hardware_profile(manifest):
    slot=os.environ.get('HARNESS_GPU_SLOT')
    profile={'accelerator_requested':manifest['training']['accelerator'],
             'gpu_slot':slot,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
             'torch_cuda_available':bool(torch.cuda.is_available()),
             'visible_device_count':int(torch.cuda.device_count()),
             'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda}
    if manifest['training']['accelerator']=='gpu':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Each seed worker must see exactly one assigned CUDA device.')
        name=torch.cuda.get_device_name(0)
        profile.update(gpu_name=name,gpu_logical_index=0,
                       gpu_total_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory))
        if manifest.get('remote',{}).get('machine_shape')=='NvidiaTeslaT4' and 'T4' not in name.upper():
            raise RuntimeError('T4x2 campaign received non-T4 device: '+name)
        if slot is None:
            raise RuntimeError('GPU slot assignment is missing from the remote scheduler.')
    else:
        profile.update(gpu_name=None,gpu_logical_index=None,gpu_total_memory_bytes=None)
    return profile


def flatten_numeric(prefix,value,output):
    if isinstance(value,dict):
        for key,item in value.items(): flatten_numeric(f'{prefix}/{key}' if prefix else key,item,output)
    elif type(value) in (int,float) and float(value)==float(value):
        output[prefix]=float(value)


def evaluate_candidate_checkpoint(checkpoint_path,problem,out_dir,model_config,problem_config=None):
    """Load an explicit candidate architecture and pass it to the read-only verifier."""
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or not isinstance(payload.get('state_dict'),dict):
        raise ValueError('candidate checkpoint must contain a state_dict')
    model=build_model(problem,model_config).double()
    if payload.get('problem')!=problem or payload.get('model_id')!=model.model_id:
        raise ValueError('candidate checkpoint model/problem identity mismatch')
    if isinstance(payload.get('step'),bool) or not isinstance(payload.get('step'),int) or payload['step']<0:
        raise ValueError('candidate checkpoint has invalid step metadata')
    model.load_state_dict(payload['state_dict'],strict=True)
    validation_problem=(problem_config or {}).get('evaluation_id',problem)
    if problem=='allen_cahn':
        pool_path=Path(out_dir)/'pools.npz'
        with np.load(pool_path) as archive:
            pool_data={name:archive[name].copy() for name in archive.files}
        metrics=evaluate_allen_cahn(model,problem_config,out_dir,pool_data)
    else:
        from evals.checkpoint_validation import evaluate_model
        metrics=evaluate_model(model,validation_problem,out_dir)
    report_path=Path(out_dir)/'validation.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    report.update({'step':payload['step'],'checkpoint':Path(checkpoint_path).name,
                   'model_id':model.model_id,'problem':validation_problem,'metrics':metrics})
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default=os.environ.get('HARNESS_MANIFEST'))
    parser.add_argument('--out',default=os.environ.get('HARNESS_RUN_DIR'))
    args=parser.parse_args()
    if not args.manifest or not args.out: parser.error('manifest and out required')
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    model_config=dict(manifest.get('model',{}))
    if 'id' not in model_config:
        declared=manifest.get('iteration',{}).get('model_id','')
        model_config['id']='rbf_fixed' if str(declared).startswith('rbf_fixed') else 'rbf_fixed'
    cfg=manifest['training'].copy(); cfg['seed']=int(os.environ.get('HARNESS_SEED',cfg['seeds'][0]))
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    for key,subdir in {'WANDB_DIR':'wandb','WANDB_CACHE_DIR':'wandb-cache','WANDB_CONFIG_DIR':'wandb-config','WANDB_DATA_DIR':'wandb-data'}.items():
        target=out/subdir; target.mkdir(exist_ok=True); os.environ[key]=str(target)
    os.environ.update(WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
    import wandb
    hardware=hardware_profile(manifest)
    json_write(out/'hardware.json',hardware)
    tracking=manifest['tracking']; mode=tracking['mode']
    if mode=='online' and not os.environ.get('WANDB_API_KEY'):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ['WANDB_API_KEY']=UserSecretsClient().get_secret('WANDB_API_KEY')
        except Exception:
            raise RuntimeError('W&B credential unavailable; no training started') from None
    versions={name:importlib.metadata.version(name) for name in ['torch','pina-mathlab','numpy','scipy','wandb','plotly']}
    json_write(out/'versions.json',versions)
    json_write(out/'config.json',manifest)
    bundle_file=Path(__file__).resolve().parents[3]/'bundle.json'
    provenance=json.loads(bundle_file.read_text()) if bundle_file.exists() else {'bundle_id':None,'scope':'local_integration'}
    json_write(out/'provenance.json',provenance)
    metrics={}; states={}
    for problem in manifest['problem']['sequence']:
        effective_cfg={**cfg,**cfg.get('problem_overrides',{}).get(problem,{})}
        effective_cfg['sampling_seeds']={name:10*cfg['seed']+offset for name,offset in cfg['seed_offsets'].items()}
        problem_config=manifest['problem'].get(problem,{})
        problem_out=out/problem
        exploratory = manifest.get('iteration', {}).get('protocol_phase') == 'exploratory'
        run_job_type = ('exploratory' if exploratory else
                        manifest.get('comparison', {}).get('arm', 'architecture-screen'))
        with wandb.init(entity=tracking['entity'],project=tracking['project'],
                        group=tracking.get('group',manifest['iteration']['id']),
                        name=f'{model_config["id"]}-{problem}-{cfg.get("sampling","sobol")}-seed-{cfg["seed"]}',
                        job_type=run_job_type,mode=mode,
                        config={'training':effective_cfg,'model':model_config,'problem':problem,
                                'physics':problem_config,'versions':versions,
                                'campaign_manifest':manifest,'provenance':provenance,'hardware':hardware},
                        dir=str(out/'wandb')) as run:
            checkpoint,summary=train(problem,effective_cfg,problem_out,run,model_config,problem_config)
            measured=evaluate_candidate_checkpoint(checkpoint,problem,problem_out,model_config,problem_config)
            sampling=json.loads((problem_out/'sampling_signals.json').read_text(encoding='utf-8'))
            diagnostic=json.loads((problem_out/'diagnostic_signals.json').read_text(encoding='utf-8'))
            with (problem_out/'history.jsonl').open(encoding='utf-8') as stream:
                history=[json.loads(line) for line in stream if line.strip()]
            log_wandb_figures(
                run,history,problem_out/'validation.npz',problem,output_dir=problem_out,
            )
            telemetry_path=out/'gpu_telemetry.csv'
            telemetry_rows=[]
            if telemetry_path.is_file():
                with telemetry_path.open(encoding='utf-8',newline='') as stream:
                    for row in csv.DictReader(stream):
                        telemetry_rows.append({key:float(value) for key,value in row.items()
                                               if value not in ('',None) and key!='elapsed_sec'})
            telemetry_signals={}
            for name in ('utilization_gpu_pct','memory_used_mib','power_draw_w','temperature_c'):
                values=[row[name] for row in telemetry_rows if name in row]
                if values:
                    telemetry_signals[f'{name}_mean']=float(sum(values)/len(values))
                    telemetry_signals[f'{name}_p95']=float(torch.tensor(values,dtype=torch.float64).quantile(.95))
                    telemetry_signals[f'{name}_max']=float(max(values))
            telemetry_signals['telemetry_sample_count']=len(telemetry_rows)
            loss_names=sorted({key for row in history for key in row if key.endswith('_loss')})
            loss_summary={}
            for key in loss_names:
                values=[row[key] for row in history if key in row]
                if values:
                    loss_summary[f'{key}_first_recorded']=float(values[0])
                    loss_summary[f'{key}_last_recorded']=float(values[-1])
                    loss_summary[f'{key}_relative_change']=float((values[-1]-values[0])/max(abs(values[0]),1e-300))
            diagnostic_groups={}
            for name,value in diagnostic.items():
                family,_,signal=name.partition('/')
                diagnostic_groups.setdefault(family,{})[signal]=value
            signals={'sampling':sampling,'training':{
                        'steps':summary['steps'],'training_and_diagnostics_sec':summary['training_and_diagnostics_sec'],
                        'steps_per_sec':summary['steps']/max(summary['training_and_diagnostics_sec'],1e-12),
                        'diagnostic_sec':summary['diagnostic_sec'],'parameters':summary['parameters'],
                        'peak_gpu_bytes':summary['peak_gpu_bytes'],'stop_reason_step_limit':int(summary['stop_reason']=='step_limit'),
                        'examples_seen':summary['examples_seen'],'losses':loss_summary},
                     **diagnostic_groups,'validation':measured,'hardware':hardware,
                     'resource_seed_campaign':telemetry_signals}
            if 'adaptive_sampling' in summary:
                signals['adaptive_sampling']=summary['adaptive_sampling']
            json_write(problem_out/'signals.json',signals)
            flattened={}
            flatten_numeric('',signals,flattened)
            metrics.update({f'{problem}/{name}':float(value) for name,value in measured.items()
                            if type(value) in (float,int)})
            metrics.update({f'{problem}/signals/{name}':value for name,value in flattened.items()})
            run.log({f'signals/{name}':value for name,value in flattened.items()})
            run.summary.update(measured)
            run.summary.update(summary)
            run.summary.update(flattened)
            rows=[]
            for key,value in flattened.items():
                family,_,signal=key.partition('/')
                unit=signal_unit(key,problem)
                rows.append([family,signal,value,unit])
            run.log({'signals/quantitative_table':wandb.Table(
                columns=['family','signal','value','unit'],data=rows)})
            run.summary['scientific_accepted']=False
            artifact_type = 'exploratory-results' if exploratory else 'baseline-evidence'
            artifact=wandb.Artifact(f'{problem}-{cfg.get("sampling","sobol")}-seed-{cfg["seed"]}',type=artifact_type)
            artifact.add_dir(str(problem_out))
            artifact.add_file(str(out/'config.json'))
            artifact.add_file(str(out/'provenance.json'))
            artifact.add_file(str(out/'hardware.json'))
            run.log_artifact(artifact)
            states[problem]={'status':'completed','wandb_url':run.url if mode=='online' else None,
                             'training':summary,'measured_signal_count':len(flattened)}
            json_write(out/'problem_status.json',states)
    json_write(out/'metrics.json',metrics)
    hashes={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for problem in manifest['problem']['sequence'] for p in (out/problem).rglob('*') if p.is_file()}
    json_write(out/'evidence_hashes.json',hashes)
    (out/'decision_note.md').write_text(
        f"Campaign using {model_config['id']} and {cfg.get('sampling','scrambled_sobol_fixed')}. "
        'Signals in signals.json are direct measurements from saved pools, histories, '
        'autodifferentiation snapshots, validation arrays and hardware telemetry. '
        'Final test unopened. No scientific claim accepted. See the preregistration '
        'for independent seed count, contrast and acceptance limitations.\n',encoding='utf-8')


if __name__=='__main__': main()
