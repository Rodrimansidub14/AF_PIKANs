"""Laplacian-form incompressible Navier-Stokes, kinematic pressure, rho=1.

No FEM field is supplied. Manufactured fields test operators, not DFG accuracy.
"""
import numpy as np
from evals.problems._grids import tensor_grid

NU=.001
U=.2
L=.1
MOMENTUM_SCALE=U**2/L
CONTINUITY_SCALE=U/L
REFERENCE_STATUS='pending_fem_convergence'


def inlet(y):
    y=np.asarray(y)
    return 4*.3*y*(.41-y)/.41**2


def residual(u,v,ux,uy,vx,vy,px,py,uxx,uyy,vxx,vyy):
    return {'momentum_x':u*ux+v*uy+px-NU*(uxx+uyy),
            'momentum_y':u*vx+v*vy+py-NU*(vxx+vyy),
            'continuity':ux+vy}


def outlet(ux,vx,p):
    # n=(1,0); nu * grad(velocity) n - p n = 0, no extra p=0.
    return NU*np.asarray(ux)-np.asarray(p),NU*np.asarray(vx)


def eval_grid(nx=440,ny=82):
    x,y=tensor_grid((0.,2.2),(0.,.41),nx,ny,interior=True)
    fluid=(x-.2)**2+(y-.2)**2>.05**2
    return x[fluid],y[fluid]


def boundary_points(n=512):
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<2:
        raise ValueError('n must be an integer >=2')
    y=np.linspace(0.,.41,n); x=np.linspace(0.,2.2,n)
    theta=np.linspace(0.,2*np.pi,n,endpoint=False)
    return {'inlet':(np.zeros(n),y),'outlet':(np.full(n,2.2),y),
            'lower_wall':(x,np.zeros(n)),'upper_wall':(x,np.full(n,.41)),
            'cylinder':(.2+.05*np.cos(theta),.2+.05*np.sin(theta))}


def critical_mask(x,y,region):
    x=np.asarray(x); y=np.asarray(y)
    r=np.sqrt((x-.2)**2+(y-.2)**2)
    fluid=(r>.05)&(x>=0)&(x<=2.2)&(y>=0)&(y<=.41)
    if region=='near_cylinder': return fluid&(r<.10)
    if region=='wake': return fluid&(x>=.25)&(x<=1.)&(np.abs(y-.2)<=.1)
    raise ValueError('unknown region')
