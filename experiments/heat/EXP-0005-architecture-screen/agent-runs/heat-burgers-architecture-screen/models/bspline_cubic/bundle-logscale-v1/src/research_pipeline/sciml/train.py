"""PINA training; this module never imports the verifier or interior truth."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch import Callback, seed_everything
from pina import Condition, LabelTensor, Trainer
from pina.equation import Equation
from pina.operator import grad
from pina.problem import SpatialProblem
from pina.domain import CartesianDomain
from pina.solver import PhysicsInformedSingleModelSolver
from pina.optim import TorchOptimizer
from .model import build_model, derivatives
from .sampling import pools
from scipy.spatial import cKDTree
from scipy.stats import qmc


def json_write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def physics(problem, problem_config=None):
    parameters = dict(problem_config or {})
    alpha = float(parameters.get('alpha', 0.1))
    nu = float(parameters.get('nu', 0.02))
    residual_scale = float(parameters.get('residual_scale', 1.1 if problem == 'heat' else .76))

    def equation(input_,output_):
        u=output_.extract(['u'])
        ux=grad(output_,input_,components=['u'],d=['x'])
        ut=grad(output_,input_,components=['u'],d=['t'])
        uxx=grad(ux,input_,d=['x'])
        if problem=='heat': return (ut-alpha*uxx)/residual_scale
        return (ut+u*ux-nu*uxx)/residual_scale
    return equation


def sampling_signals(problem, data):
    """Return measured Sobol coverage and point-count signals for stored pools."""
    points = np.asarray(data['interior'], dtype=np.float64).copy()
    low, high = (0.0, 1.0) if problem == 'heat' else (-1.0, 1.0)
    points[:, 0] = (points[:, 0] - low) / (high - low)
    nearest = cKDTree(points).query(points, k=2)[0][:, 1]
    signals = {
        'interior_count': len(data['interior']),
        'diagnostic_count': len(data['diagnostic']),
        'left_count': len(data['left']),
        'right_count': len(data['right']),
        'initial_count': len(data['initial']),
        'sobol_centered_discrepancy': float(qmc.discrepancy(points, method='CD')),
        'nearest_neighbor_min': float(nearest.min()),
        'nearest_neighbor_median': float(np.median(nearest)),
        'nearest_neighbor_p95': float(np.quantile(nearest, .95)),
        'nearest_neighbor_max': float(nearest.max()),
        'coordinate_x_min': float(data['interior'][:, 0].min()),
        'coordinate_x_max': float(data['interior'][:, 0].max()),
        'coordinate_t_min': float(data['interior'][:, 1].min()),
        'coordinate_t_max': float(data['interior'][:, 1].max()),
    }
    return signals


def burgers_trace(input_,output_,condition='initial',problem_config=None):
    """Boundary/initial traces for the declared Burgers variant."""
    parameters=dict(problem_config or {})
    x=input_.extract(['x'])
    output=output_.extract(['u'])
    definition=parameters.get('definition','stationary_tanh')
    if definition=='raissi_pinnacle_1d':
        target=-torch.sin(torch.pi*x) if condition=='initial' else torch.zeros_like(output)
        scale=float(parameters.get('field_scale',1.0))
    elif definition=='stationary_tanh':
        nu=float(parameters.get('nu',.02)); speed=float(parameters.get('c',0.))
        amplitude=float(parameters.get('A',.5)); center=float(parameters.get('x0',-.25))
        exact=speed-amplitude*torch.tanh(amplitude*(x-speed*input_.extract(['t'])-center)/(2*nu))
        target=exact
        scale=float(parameters.get('field_scale',amplitude+abs(speed)))
    else:
        raise ValueError(f'unknown Burgers definition: {definition!r}')
    return (output-target)/scale


def burgers_condition_equation(condition,problem_config):
    """Bind condition metadata while exposing PINA's required two-argument API."""
    parameters=dict(problem_config or {})
    def equation(input_,output_):
        return burgers_trace(input_,output_,condition,parameters)
    return equation


def make_problem(problem,data,problem_config=None):
    parameters=dict(problem_config or {})
    conditions={'physics':Condition(input=LabelTensor(torch.tensor(data['interior']),['x','t']),equation=Equation(physics(problem,parameters)))}
    if problem=='burgers':
        for name in ('left','right','initial'):
            conditions[name]=Condition(input=LabelTensor(torch.tensor(data[name]),['x','t']),
                equation=Equation(burgers_condition_equation(name,parameters)))
    domain=CartesianDomain({'x':[0.,1.] if problem=='heat' else [-1.,1.],'t':[0.,1.]})
    # CartesianDomain sorts variable names; explicitly preserve model order x,t.
    return type('BaselineProblem',(SpatialProblem,),{'output_variables':['u'],'spatial_variables':['x','t'],
                'spatial_domain':domain,'conditions':conditions})()


class Recorder(Callback):
    def __init__(self,model,data,out,cfg,run,problem_config=None):
        self.model=model; self.data=data; self.out=Path(out); self.cfg=cfg; self.run=run
        self.problem_config=dict(problem_config or {})
        self.residual_scale=float(self.problem_config.get(
            'residual_scale',1.1 if model.problem=='heat' else .76))
        self.alpha=float(self.problem_config.get('alpha',.1))
        self.nu=float(self.problem_config.get('nu',.02))
        self.rows=[]; self.started=time.perf_counter(); self.diagnostic_sec=0.; self.gradient_norm=0.
        self.examples_seen={}; self.latest_row={}; self.last_snapshot={}; self.gradient_signals={}

    def on_after_backward(self,trainer,pl_module):
        self.gradient_norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in self.model.parameters() if p.grad is not None)))
        if not np.isfinite(self.gradient_norm): raise FloatingPointError('nonfinite_gradient')
        signals={}
        for index,layer in enumerate(self.model.layers):
            for name,parameter in layer.named_parameters(recurse=False):
                signal_name={'coefficients':'basis_coefficients'}.get(name,name)
                if self.model.model_family=='rbf':
                    signal_name={'coefficients':'rbf','weight':'weight','bias':'bias'}.get(name,name)
                grad_norm=float(parameter.grad.detach().norm()) if parameter.grad is not None else 0.0
                if not np.isfinite(grad_norm): raise FloatingPointError('nonfinite_layer_gradient')
                signals[f'training/gradient/layer_{index}/{self.model.model_family}_{signal_name}_l2']=grad_norm
                signals[f'training/parameter/layer_{index}/{self.model.model_family}_{signal_name}_l2']=float(parameter.detach().norm())
        self.gradient_signals=signals

    def snapshot(self,step,checkpoint=False):
        start=time.perf_counter(); device=next(self.model.parameters()).device
        coordinates=self.data['diagnostic'][:self.cfg['diagnostic_points']]
        x=torch.tensor(coordinates,device=device,dtype=torch.float64,requires_grad=True)
        with torch.enable_grad():
            u=self.model(x); ux,ut,uxx=derivatives(u,x)
            res=ut-self.alpha*uxx if self.model.problem=='heat' else ut+u*ux-self.nu*uxx
            sensitivities=[]
            for i in range(min(self.cfg['sensitivity_points'],len(x))):
                g=torch.autograd.grad(u[i].sum(),tuple(self.model.parameters()),retain_graph=True)
                sensitivities.append(float(torch.sqrt(sum(v.square().sum() for v in g))))
        arrays={k:v.detach().cpu().numpy() for k,v in {'u':u,'ux':ux,'ut':ut,'uxx':uxx,'residual':res}.items()}
        arrays.update(coordinates=coordinates,sensitivity=np.asarray(sensitivities))
        stats={}
        with torch.no_grad():
            _, traces=self.model.forward_with_trace(x)
            for i,(layer,trace) in enumerate(zip(self.model.layers,traces)):
                activation=trace['activation'].detach()
                preactivation=trace['preactivation'].detach()
                if 'basis' in trace:
                    occupancy=trace['basis'].mean(0).cpu().numpy()
                    arrays[f'basis_occupancy_{i}']=occupancy
                    probabilities=occupancy / max(float(occupancy.sum()),1e-300)
                    entropy=-float(np.sum(probabilities*np.log(np.maximum(probabilities,1e-300))))
                    family=self.model.model_family
                    stats[f'{family}/layer_{i}/occupancy_min']=float(occupancy.min())
                    stats[f'{family}/layer_{i}/occupancy_mean']=float(occupancy.mean())
                    stats[f'{family}/layer_{i}/occupancy_p95']=float(np.quantile(occupancy,.95))
                    stats[f'{family}/layer_{i}/occupancy_below_0_01_count']=int(np.count_nonzero(occupancy<.01))
                    stats[f'{family}/layer_{i}/effective_basis_count']=float(np.exp(entropy))
                    stats[f'{family}/layer_{i}/basis_coefficients_l2']=float(layer.coefficients.norm())
                    stats[f'{family}/layer_{i}/base_parameter_l2']=float(layer.base.norm())
                    if family=='rbf':
                        stats[f'{family}/layer_{i}/coefficient_l2']=stats[f'{family}/layer_{i}/basis_coefficients_l2']
                arrays[f'activations_{i}']=activation.cpu().numpy()
                arrays[f'preactivations_{i}']=preactivation.cpu().numpy()
                stats[f'activation/layer_{i}/mean']=float(activation.mean())
                stats[f'activation/layer_{i}/rms']=float(torch.sqrt(torch.mean(activation.square())))
                stats[f'activation/layer_{i}/abs_p95']=float(torch.quantile(activation.abs().reshape(-1),.95))
                stats[f'activation/layer_{i}/abs_max']=float(activation.abs().max())
                stats[f'activation/layer_{i}/preactivation_rms']=float(torch.sqrt(torch.mean(preactivation.square())))
                if i<len(self.model.layers)-1:
                    saturation=float((activation.abs()>.99).double().mean())
                    stats[f'activation/layer_{i}/saturation_fraction_abs_gt_0_99']=saturation
                    if self.model.model_family=='rbf':
                        stats[f'rbf/layer_{i}/saturation_fraction_abs_gt_0_99']=saturation
        if not all(np.isfinite(v).all() for v in arrays.values()): raise FloatingPointError('nonfinite_diagnostic')
        np.savez_compressed(self.out/f'diagnostic-{step:06d}.npz',**arrays)
        if checkpoint:
            torch.save({'problem':self.model.problem,'model_id':self.model.model_id,
                        'model_family':self.model.model_family,'problem_parameters':self.problem_config,
                        'validation_problem':self.problem_config.get('evaluation_id',self.model.problem),
                        'state_dict':self.model.state_dict(),'step':step},self.out/f'checkpoint-{step:06d}.pt')
        self.diagnostic_sec+=time.perf_counter()-start
        for name in ('u','ux','ut','uxx','residual'):
            values=arrays[name].reshape(-1)
            prefix=f'autodiff/{name}' if name!='residual' else 'residual'
            stats[f'{prefix}/mean']=float(values.mean())
            stats[f'{prefix}/rms']=float(np.sqrt(np.mean(np.square(values))))
            stats[f'{prefix}/abs_p95']=float(np.quantile(np.abs(values),.95))
            stats[f'{prefix}/abs_max']=float(np.abs(values).max())
        stats['residual/normalized_rms']=stats['residual/rms']/self.residual_scale
        stats['residual/normalized_abs_p95']=stats['residual/abs_p95']/self.residual_scale
        if len(arrays['sensitivity']):
            stats['sensitivity/l2_median']=float(np.median(arrays['sensitivity']))
            stats['sensitivity/l2_p95']=float(np.quantile(arrays['sensitivity'],.95))
            stats['sensitivity/l2_max']=float(np.max(arrays['sensitivity']))
        stats['diagnostic/point_count']=len(coordinates)
        stats['diagnostic/elapsed_sec']=self.diagnostic_sec
        self.last_snapshot=stats
        return stats

    def on_train_start(self,trainer,pl_module):
        self.started=time.perf_counter()
        self.snapshot(0)

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        step=trainer.global_step
        row={'step':step,'elapsed_sec':time.perf_counter()-self.started,'gradient_norm':self.gradient_norm}
        for name,values in batch:
            points=values.get('input')
            if torch.is_tensor(points):
                self.examples_seen[name]=self.examples_seen.get(name,0)+len(points)
                row[f'examples_seen/{name}']=self.examples_seen[name]
        for key,value in trainer.callback_metrics.items():
            if torch.is_tensor(value) and value.numel()==1: row[str(key)]=float(value.detach())
        row.update(self.gradient_signals)
        if not all(np.isfinite(v) for v in row.values()): raise FloatingPointError('nonfinite_training_metric')
        if step%self.cfg['diagnostic_every']==0: row.update(self.snapshot(step))
        self.latest_row=row
        cadence=max(1,int(self.cfg.get('metric_log_every',100)))
        if step%cadence==0 or step%self.cfg['diagnostic_every']==0:
            self.rows.append(row)
            with (self.out/'history.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            self.run.log(row,step=step)
        if time.perf_counter()-self.started>=self.cfg['seconds_per_problem']: trainer.should_stop=True


def train(problem,cfg,out,run,model_config=None,problem_config=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    seed_everything(cfg['seed'],workers=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg['cpu_threads'])
    data=pools(problem,cfg['seed'],cfg['interior_points'],cfg['boundary_points'])
    np.savez_compressed(out/'pools.npz',**data)
    point_signals=sampling_signals(problem,data)
    json_write(out/'sampling_signals.json',point_signals)
    model=build_model(problem,model_config).double()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    solver=PhysicsInformedSingleModelSolver(make_problem(problem,data,problem_config),model,TorchOptimizer(torch.optim.Adam,lr=cfg['learning_rate']))
    recorder=Recorder(model,data,out,cfg,run,problem_config)
    trainer=Trainer(solver=solver,batch_size=cfg['batch_size'],max_steps=cfg['steps_per_problem'],max_epochs=-1,
                    precision='64-true',accelerator=cfg['accelerator'],devices=1,
                    logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
                    callbacks=[recorder],default_root_dir=str(out),num_sanity_val_steps=0,
                    deterministic=True,log_every_n_steps=max(1,int(cfg.get('metric_log_every',100))))
    started=time.perf_counter(); trainer.train()
    final_signals=recorder.snapshot(trainer.global_step,checkpoint=True)
    json_write(out/'diagnostic_signals.json',final_signals)
    if recorder.latest_row.get('step')!=trainer.global_step or trainer.global_step%max(1,int(cfg.get('metric_log_every',100)))!=0:
        final_row={**recorder.latest_row,**final_signals,'step':trainer.global_step,
                   'elapsed_sec':time.perf_counter()-recorder.started,'event':'final'}
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(final_row,allow_nan=False)+'\n')
        run.log(final_row,step=trainer.global_step)
    summary={'steps':trainer.global_step,'training_and_diagnostics_sec':time.perf_counter()-started,
             'diagnostic_sec':recorder.diagnostic_sec,'parameters':sum(p.numel() for p in model.parameters()),
             'model_id':model.model_id,'model_family':model.model_family,
             'examples_seen':recorder.examples_seen,
             'device':str(next(model.parameters()).device),'peak_gpu_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
             'stop_reason':'step_limit' if trainer.global_step>=cfg['steps_per_problem'] else 'time_limit',
             'sampling_signals':point_signals,'diagnostic_signals':final_signals}
    json_write(out/'training_summary.json',summary)
    return out/f'checkpoint-{trainer.global_step:06d}.pt',summary
