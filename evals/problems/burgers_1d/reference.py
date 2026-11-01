"""Exact traveling wave; parameters and traces follow the thesis contract."""
import numpy as np
from evals.problems._grids import tensor_grid


def _check(nu,c,A,x0):
    if not all(np.isfinite(v) for v in (nu,c,A,x0)) or nu<=0 or A<=0:
        raise ValueError('finite parameters, positive viscosity and amplitude required')


def exact(x,t,nu=.02,c=0.,A=.5,x0=-.25):
    _check(nu,c,A,x0)
    return c-A*np.tanh(A*(np.asarray(x)-c*np.asarray(t)-x0)/(2*nu))


def residual(u,ut,ux,uxx,nu=.02):
    _check(nu,0.,.5,-.25)
    return np.asarray(ut)+np.asarray(u)*np.asarray(ux)-nu*np.asarray(uxx)


def residual_scale(nu=.02,c=0.,A=.5):
    _check(nu,c,A,-.25)
    U=A+abs(c)
    return U+U**2+nu*U


def boundary(side,t,**parameters):
    if side not in (-1,1): raise ValueError('side must be -1 or 1')
    return exact(np.full_like(np.asarray(t,dtype=float),side),t,**parameters)


def initial(x,**parameters):
    return exact(x,np.zeros_like(x),**parameters)


def critical_mask(x,t,nu=.02,c=0.,A=.5,x0=-.25):
    _check(nu,c,A,x0)
    return np.abs(np.asarray(x)-c*np.asarray(t)-x0)<=3*(2*nu/A)


def eval_grid(nx=512,nt=256):
    return tensor_grid((-1.,1.),(0.,1.),nx,nt)
