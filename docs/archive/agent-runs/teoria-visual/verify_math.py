"""Independent checks of formulas and declared plotting parameters."""
from pathlib import Path
import json
import numpy as np
out=Path(__file__).resolve().parent
results={}

# Centered Laplacian must converge quadratically to manufactured source.
errs=[]
for n in (101,201):
    x=np.linspace(0,1,n); h=x[1]-x[0]; X,Y=np.meshgrid(x,x)
    u=np.sin(4*np.pi*X)*np.sin(4*np.pi*Y)
    minuslap=-(u[2:,1:-1]+u[:-2,1:-1]+u[1:-1,2:]+u[1:-1,:-2]-4*u[1:-1,1:-1])/h**2
    f=2*(4*np.pi)**2*u[1:-1,1:-1]
    errs.append(float(np.max(np.abs(minuslap-f))))
assert 3.9<errs[0]/errs[1]<4.1
results['poisson_fd_errors']=errs
results['poisson_refinement_ratio']=errs[0]/errs[1]

worst=0.
for c in (0.,.5):
    for nu in (.02,.005):
        x=np.linspace(-1,1,2001); A=.5
        for t in (0.,.5,1.):
            q=A*(x-c*t+.25)/(2*nu); T=np.tanh(q); S=1-T*T
            u=c-A*T; ux=-A*A*S/(2*nu); ut=-c*ux; uxx=A**3*S*T/(2*nu**2)
            worst=max(worst,float(np.max(np.abs(ut+u*ux-nu*uxx))))
assert worst<1e-12
results['burgers_max_residual_all_contract_parameters']=worst

d=.001; ell=np.sqrt(2*d/5); q=np.linspace(-6,6,2001); u=np.tanh(q)
uxx=-2*u*(1-u*u)/ell**2
res=d*uxx+5*(u-u**3)
assert np.max(np.abs(res))<1e-12
results['allen_cahn_ideal_kink_max_residual']=float(np.max(np.abs(res)))
results['allen_cahn_kink_scope']='Stationary infinite-line mechanism, not the finite-domain initial boundary value solution.'

y=np.linspace(0,.41,10001); inlet=4*.3*y*(.41-y)/.41**2
mean=float(np.trapezoid(inlet,y)/.41)
assert abs(mean-.2)<1e-8 and abs(max(inlet)-.3)<1e-12
results['dfg_inlet_mean']=mean; results['dfg_reynolds']=.2*.1/.001

C=np.array([[2.1836e9,.6332e9],[.6332e9,2.1836e9]])
etop=np.array([-.02904,-.05157]); gradphi=1e5
strain=np.linalg.solve(C,-etop*gradphi)
assert strain[0]>0 and strain[1]>0
results['pvdf_upper_free_strain_illustrative']=strain.tolist()
results['pvdf_scope']='Locally stress-free tendency under assumed uniform potential gradient; not bonded cantilever solution.'

J=-1/(.5/1+.5/2)
assert abs(-1*(4/3)-J)<1e-14 and abs(-2*(2/3)-J)<1e-14
assert abs(.5*(4/3)+.5*(2/3)-1)<1e-14
results['dielectric_interface_flux']=J
results['status']='PASS'
(out/'math-verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
