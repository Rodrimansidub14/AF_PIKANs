"""Strict exploratory metrics; no thresholds or scientific acceptance.

Arrays must come from a trusted evaluator on verifier coordinates. This module
does not authenticate candidate-supplied derivatives or reference arrays.
"""
import numpy as np
from evals.graders.core import Score, MINIMIZE


def _missing(name,scope,reason):
    return Score(name,None,MINIMIZE,scope,0,reason)


def residual_score(values,*,scale,equation):
    name='normalized_residual_mean'; scope='eval_grid:'+equation
    a=np.asarray(values,dtype=float)
    if a.size==0: return _missing(name,scope,'empty_input')
    if not np.isfinite(a).all(): return _missing(name,scope,'non_finite_input')
    if not np.isfinite(scale) or scale<=0: return _missing(name,scope,'invalid_scale')
    value=float(np.mean(np.abs(a/scale)))
    if not np.isfinite(value): return _missing(name,scope,'non_finite_result')
    return Score(name,value,MINIMIZE,scope,a.size)


def field_error(problem,prediction,reference=None,*,weights=None):
    name='relative_l2_error'; scope='eval_grid'
    if problem in ('allen_cahn_1d','dfg_2d'):
        return _missing(name,scope,'reference_not_verified')
    if problem!='burgers_1d': return _missing(name,scope,'problem_not_enabled')
    if reference is None: return _missing(name,scope,'reference_missing')
    p=np.asarray(prediction,dtype=float); r=np.asarray(reference,dtype=float)
    if p.shape!=r.shape: return _missing(name,scope,'shape_mismatch')
    if not p.size: return _missing(name,scope,'empty_input')
    w=np.ones_like(r) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=r.shape: return _missing(name,scope,'weight_shape_mismatch')
    if not all(np.isfinite(a).all() for a in (p,r,w)):
        return _missing(name,scope,'non_finite_input')
    if (w<0).any() or not (w>0).any(): return _missing(name,scope,'invalid_weights')
    with np.errstate(over='ignore',invalid='ignore'):
        denom=np.sum(w*r*r)
        numerator=np.sum(w*(p-r)**2)
    if not np.isfinite(denom) or not np.isfinite(numerator):
        return _missing(name,scope,'non_finite_intermediate')
    if denom==0: return _missing(name,scope,'reference_norm_zero')
    value=float(np.sqrt(numerator/denom))
    if not np.isfinite(value): return _missing(name,scope,'non_finite_result')
    return Score(name,value,MINIMIZE,scope,p.size)
