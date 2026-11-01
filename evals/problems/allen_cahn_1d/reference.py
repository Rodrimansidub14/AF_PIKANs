"""Physics/traces only. No verified numerical field reference is supplied."""
import numpy as np
from evals.problems._grids import tensor_grid

D=.001
RESIDUAL_SCALE=1.+D+5.
REFERENCE_STATUS='pending_convergence'


def initial(x):
    x=np.asarray(x)
    return x**2*np.cos(np.pi*x)


def transform(x,t,raw):
    x=np.asarray(x)
    return initial(x)+np.asarray(t)*(1-x*x)*np.asarray(raw)


def residual(u,ut,uxx):
    u=np.asarray(u)
    return np.asarray(ut)-D*np.asarray(uxx)-5*(u-u**3)


def eval_grid(nx=512,nt=256):
    return tensor_grid((-1.,1.),(0.,1.),nx,nt)
