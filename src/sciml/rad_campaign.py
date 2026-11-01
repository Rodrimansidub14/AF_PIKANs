"""EXP-0006 job adapter: ten jobs, five paired random seeds, two GPU slots."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))


def job_spec(job):
    if 201<=job<=205: return 'sobol',job
    if 1201<=job<=1205: return 'rad',job-1000
    raise ValueError('job ID outside preregistered five pairs')


def assert_launch_ready(manifest):
    status=manifest.get('research_status',{})
    design=manifest.get('comparison_design',{})
    if not all(status.get(k) is True for k in ('scientific_preregistration_signed','baseline_frozen','submission_allowed')) or design.get('launch_ready') is not True:
        raise RuntimeError('Missing scientific prerequisites: signed preregistration, frozen baseline and launch-ready manifest required')


def main():
    manifest=json.loads(Path(os.environ['HARNESS_MANIFEST']).read_text(encoding='utf-8'))
    assert_launch_ready(manifest)
    from src.sciml import campaign
    job=int(os.environ['HARNESS_SEED']); arm,seed=job_spec(job)
    out=Path(os.environ['HARNESS_RUN_DIR']); out.mkdir(parents=True,exist_ok=True)
    manifest['comparison']={'job_id':job,'arm':arm,'paired_seed':seed}
    manifest['training']['sampling']='rad' if arm=='rad' else 'scrambled_sobol_fixed'
    manifest['focalization']['enabled']=arm=='rad'
    manifest['iteration']['focalization_id']=arm
    path=out/'effective-manifest.json'
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    os.environ['HARNESS_SEED']=str(seed)
    sys.argv=[sys.argv[0],'--manifest',str(path),'--out',str(out)]
    campaign.main()
    if arm=='rad':
        stats=json.loads((out/'burgers/adaptive_sampling.json').read_text())
        if stats['refresh_count']==0 or stats['consumed_changed_batches']==0:
            raise RuntimeError('RAD arm never consumed adaptive collocation points')


if __name__=='__main__': main()
