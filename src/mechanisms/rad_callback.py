"""PINA integration for fixed-count residual-adaptive sampling."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch import Callback
from src.research_pipeline.sciml.model import derivatives
from src.research_pipeline.sciml.sampling import sobol
from .rad import sample_rad


class RADCallback(Callback):
    def __init__(self,model,problem,data,out,cfg,run,physics):
        self.model=model; self.problem=problem; self.data=data; self.out=Path(out); self.run=run
        settings=cfg.get('rad',{})
        self.warmup=int(settings.get('warmup_steps',2000))
        self.every=int(settings.get('refresh_every',1000))
        self.chunk=int(settings.get('chunk_size',512))
        self.max_steps=int(cfg['steps_per_problem'])
        if min(self.warmup,self.every,self.chunk)<1: raise ValueError('positive RAD intervals required')
        self.power=float(settings.get('power',2.)); self.uniform=float(settings.get('uniform_fraction',.5))
        self.nu=float(physics.get('nu',.01/np.pi))
        self.candidates=sobol(int(settings.get('candidate_points',32768)),2,10*cfg['seed']+7)
        self.candidates[:,0]=2*self.candidates[:,0]-1
        self.rng=np.random.default_rng(10*cfg['seed']+8)
        self.refresh_count=0; self.elapsed=0.; self.consumed_changed_batches=0; self.last_step=None
        self.pending=False; self.current_support=set()

    def refresh(self,step):
        start=time.perf_counter(); device=next(self.model.parameters()).device; values=[]
        with torch.enable_grad():
            for i in range(0,len(self.candidates),self.chunk):
                x=torch.tensor(self.candidates[i:i+self.chunk],device=device,dtype=torch.float64,requires_grad=True)
                u=self.model(x); ux,ut,uxx=derivatives(u,x)
                residual=ut+u*ux-self.nu*uxx
                values.append(residual.detach().cpu().numpy().ravel())
        residual=np.concatenate(values)
        result=sample_rad(self.candidates,residual,len(self.data['interior']),self.rng,self.power,self.uniform)
        # Condition stores one shared input tensor. Preserve identity for PINA's
        # existing subsets/dataloaders and Adam state; num_workers stays zero.
        target=self.problem.conditions['physics'].input
        with torch.no_grad(): target.copy_(torch.as_tensor(result['points'],dtype=target.dtype,device=target.device))
        self.data['interior']=result['points']
        self.current_support=set(map(tuple,result['points']))
        self.pending=True; self.last_step=step; self.refresh_count+=1
        np.savez_compressed(self.out/f'rad-{step:06d}.npz',candidates=self.candidates,residual=residual,
                            probabilities=result['probabilities'],indices=result['indices'],points=result['points'])
        seconds=time.perf_counter()-start; self.elapsed+=seconds
        row={'step':step,'sampling/rad/refresh_count':self.refresh_count,
             'sampling/rad/seconds_total':self.elapsed,'sampling/rad/refresh_seconds':seconds,
             'sampling/rad/effective_candidates':result['effective_candidates'],
             'sampling/rad/unique_points':result['unique_count'],
             'sampling/rad/residual_evaluations_total':len(self.candidates)*self.refresh_count,
             'sampling/rad/probability_min':float(result['probabilities'].min()),
             'sampling/rad/probability_max':float(result['probabilities'].max())}
        with (self.out/'rad-history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        self.run.log(row,step=step)

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        if self.pending and trainer.global_step>self.last_step:
            for name,values in batch:
                if name=='physics':
                    points=values['input'].detach().cpu().numpy()
                    if all(tuple(p) in self.current_support for p in points):
                        self.consumed_changed_batches+=1; self.pending=False
                    elif trainer.global_step>self.last_step+2:
                        raise RuntimeError('PINA did not consume updated RAD collocation points')

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        step=trainer.global_step
        if step>=self.warmup and (step-self.warmup)%self.every==0 and step<self.max_steps and not trainer.should_stop:
            self.refresh(step)

    def summary(self):
        return {'refresh_count':self.refresh_count,'seconds':self.elapsed,
                'consumed_changed_batches':self.consumed_changed_batches,
                'candidate_evaluations':len(self.candidates)*self.refresh_count,
                'candidate_points':len(self.candidates),'power':self.power,'uniform_fraction':self.uniform,
                'sampling_with_replacement':True,'loss_importance_correction':False}
