"""Reproducible solution graphics for P1-P3; analytical and numerical references.

Only writes below this directory. No neural-training results are implied.
"""
from pathlib import Path
import os, json, time
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
OUT=ROOT/'solutions'; OUT.mkdir(exist_ok=True)
TEAL,ORANGE,INK='#137C80','#D66A35','#222C30'
CM=LinearSegmentedColormap.from_list('teal-white-orange',[TEAL,'#FAFAF6',ORANGE])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#BBC3C4',
 'axes.titlesize':12,'figure.facecolor':'white','savefig.facecolor':'white',
 'pdf.fonttype':42,'svg.fonttype':'none','lines.linewidth':2})
def save(fig,name):
    for ext in ('pdf','png','svg'): fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=180)
    plt.close(fig)
def heading(ax,letter,text): ax.set_title(f'{letter}  {text}',loc='left',pad=10)
evidence={'numpy':np.__version__,'scipy':scipy.__version__}

# Poisson: actual sparse finite-difference solves versus known exact solution.
records=[]; k=4
for n in (33,65,129):
    x=np.linspace(0,1,n); h=1/(n-1); X,Y=np.meshgrid(x[1:-1],x[1:-1])
    exact=np.sin(k*np.pi*X)*np.sin(k*np.pi*Y); f=2*(k*np.pi)**2*exact
    m=n-2; D=diags([-np.ones(m-1),2*np.ones(m),-np.ones(m-1)],[-1,0,1])/h**2
    K=(kron(eye(m),D)+kron(D,eye(m))).tocsr()
    u=spsolve(K,f.ravel()).reshape((m,m))
    err=np.linalg.norm(u-exact)/np.linalg.norm(exact)
    residual=np.linalg.norm(K@u.ravel()-f.ravel())/np.linalg.norm(f)
    records.append({'nodes_per_axis':n,'unknowns':m*m,'h':h,'relative_l2_grid_error':float(err),'relative_algebraic_residual':float(residual)})
    full=np.zeros((n,n)); full[1:-1,1:-1]=u
    np.savez_compressed(OUT/f'p1-fd-{n}.npz',x=x,u=full,exact=np.pad(exact,1))
ratios=[records[i]['relative_l2_grid_error']/records[i+1]['relative_l2_grid_error'] for i in range(2)]
assert all(3.9<r<4.2 for r in ratios)
assert max(r['relative_algebraic_residual'] for r in records)<1e-10
evidence['poisson']={'method':'Centered finite differences, second order; Dirichlet zero; exact manufactured reference','k':4,'runs':records,'error_ratios':ratios}
x=np.linspace(0,1,257); X,Y=np.meshgrid(x,x)
fig,ax=plt.subplots(2,2,figsize=(10,7.3),layout='constrained')
for a,k,letter in ((ax[0,0],1,'a'),(ax[0,1],4,'b')):
    im=a.pcolormesh(X,Y,np.sin(k*np.pi*X)*np.sin(k*np.pi*Y),cmap=CM,vmin=-1,vmax=1,rasterized=True)
    a.set(xlabel='$x$',ylabel='$y$'); a.set_aspect('equal'); heading(a,letter,f'Solución exacta · k = {k}')
fig.colorbar(im,ax=ax[0,:],label='$u$',ticks=[-1,0,1],shrink=.8)
for k,col in ((1,TEAL),(4,ORANGE)):
    ax[1,0].plot(x,np.sin(k*np.pi*x),color=col,label=f'$k={k}$')
ax[1,0].set(xlabel='$x$',ylabel=r'$u(x,1/(2k))$');ax[1,0].legend(frameon=False)
heading(ax[1,0],'c','Cortes de las soluciones exactas')
hs=np.array([r['h'] for r in records]); errs=np.array([r['relative_l2_grid_error'] for r in records])
ax[1,1].loglog(hs,errs,'o-',color=TEAL,label='Error FDM')
ax[1,1].loglog(hs,errs[-1]*(hs/hs[-1])**2,'--',color=ORANGE,label='$O(h^2)$')
ax[1,1].set(xlabel='$h$',ylabel='Error relativo discreto');ax[1,1].legend(frameon=False)
heading(ax[1,1],'d','Comprobación numérica · k = 4')
save(fig,'p1-solution')
print('P1 verified',flush=True)

# Burgers: exact finite-domain solution; all plotted traces satisfy the contract.
A=.5;c=.5;nu=.02;x0=-.25
x=np.linspace(-1,1,1001); t=np.linspace(0,1,301);X,T=np.meshgrid(x,t)
q=A*(X-c*T-x0)/(2*nu);U=c-A*np.tanh(q)
np.savez_compressed(OUT/'p2-exact.npz',x=x,t=t,u=U,A=A,c=c,nu=nu,x0=x0)
worst=0.
for visc in (.02,.005):
    for speed in (0.,.5):
        Q=A*(X-speed*T-x0)/(2*visc); tan=np.tanh(Q); S=1-tan*tan
        uu=speed-A*tan;ux=-A*A/(2*visc)*S;ut=-speed*ux;uxx=A**3/(2*visc**2)*S*tan
        worst=max(worst,float(np.max(np.abs(ut+uu*ux-visc*uxx))))
assert worst<1e-12
evidence['burgers']={'method':'Exact viscous traveling wave evaluated on finite-domain initial and boundary traces','A':A,'c':c,'nu':nu,'x0':x0,'max_analytic_residual_all_contract_parameters':worst}
fig,ax=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
im=ax[0].pcolormesh(x,t,U,cmap=CM,vmin=0,vmax=1,rasterized=True)
ax[0].plot(c*t+x0,t,'--',color=INK,lw=1.1);ax[0].set(xlabel='$x$',ylabel='$t$')
fig.colorbar(im,ax=ax[0],label='$u$',ticks=[0,.5,1]);heading(ax[0],'a','Solución exacta en espacio–tiempo')
for tt,col,ls in ((0,TEAL,'-'),(.5,INK,'--'),(1,ORANGE,'-')):
    ax[1].plot(x,c-A*np.tanh(A*(x-c*tt-x0)/(2*nu)),color=col,ls=ls,label=f'$t={tt:g}$')
ax[1].set(xlabel='$x$',ylabel='$u(x,t)$');ax[1].legend(frameon=False);heading(ax[1],'b','Perfiles de la misma solución')
save(fig,'p2-solution');print('P2 verified',flush=True)

# Allen-Cahn: method of lines, sparse analytic Jacobian, BDF time stepping.
t=np.linspace(0,1,301); d=.001
def solve_ac(n,rtol=1e-9,atol=1e-11):
    tic=time.perf_counter(); x=np.linspace(-1,1,n);h=x[1]-x[0];m=n-2
    lap=diags([np.ones(m-1),-2*np.ones(m),np.ones(m-1)],[-1,0,1],format='csc')/h**2
    forcing=np.zeros(m);forcing[0]-=d/h**2;forcing[-1]-=d/h**2
    def fun(time,u): return d*(lap@u)+forcing+5*(u-u**3)
    def jac(time,u): return d*lap+diags(5*(1-3*u*u),format='csc')
    sol=solve_ivp(fun,(0,1),x[1:-1]**2*np.cos(np.pi*x[1:-1]),method='BDF',jac=jac,t_eval=t,rtol=rtol,atol=atol)
    if not sol.success: raise RuntimeError(sol.message)
    U=np.full((len(t),n),-1.);U[:,1:-1]=sol.y.T
    energy=d/(2*h)*np.sum(np.diff(U,axis=1)**2,axis=1)+h*np.sum(1.25*(U[:,1:-1]**2-1)**2,axis=1)
    assert np.max(np.diff(energy))<1e-8
    assert np.max(np.abs(U[0]-x*x*np.cos(np.pi*x)))<1e-13
    assert np.max(U)<=1+1e-8 and np.min(U)>=-1-1e-8
    record={'nodes':n,'h':float(h),'rtol':rtol,'atol':atol,'nfev':sol.nfev,'nlu':sol.nlu,'seconds':time.perf_counter()-tic,'max_energy_increment':float(np.max(np.diff(energy)))}
    np.savez_compressed(OUT/f'p3-bdf-{n}-{rtol:g}.npz',x=x,t=t,u=U,energy=energy)
    print('P3',record,flush=True)
    return x,U,energy,record
runs=[solve_ac(n) for n in (401,801,1601)]
coarse=np.max(np.abs(runs[0][1]-runs[1][1][:,::2]));fine=np.max(np.abs(runs[1][1]-runs[2][1][:,::2]))
temporal=solve_ac(1601,rtol=1e-11,atol=1e-13)
time_diff=float(np.max(np.abs(temporal[1]-runs[-1][1])))
assert fine<1e-3 and time_diff<fine/20
ratio=float(coarse/fine)
assert 3.5<ratio<4.5
x,U,E,_=temporal
evidence['allen_cahn']={'method':'Second-order centered differences + BDF, sparse exact Jacobian','d':d,'domain':[-1,1],'time':[0,1],'initial':'x^2 cos(pi x)','boundaries':'u(-1,t)=u(1,t)=-1','runs':[r[3] for r in runs]+[temporal[3]],'successive_mesh_linf_differences':[float(coarse),float(fine)],'observed_ratio':ratio,'temporal_tolerance_linf_difference':time_diff,'scope':'Self-convergence and temporal sensitivity; not a proof of absolute error. No PINN training.'}
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
im=ax[0,0].pcolormesh(x,t,U,cmap=CM,vmin=-1,vmax=1,rasterized=True)
ax[0,0].set(xlabel='$x$',ylabel='$t$');fig.colorbar(im,ax=ax[0,0],label='$u$',ticks=[-1,0,1]);heading(ax[0,0],'a','Solución numérica del caso P3')
for tt,col,ls in ((0,INK,':'),(.5,TEAL,'--'),(1,ORANGE,'-')):
    j=int(round(tt*300)); ax[0,1].plot(x,U[j],color=col,ls=ls,label=f'$t={tt:g}$')
ax[0,1].set(xlabel='$x$',ylabel='$u(x,t)$');ax[0,1].legend(frameon=False);heading(ax[0,1],'b','Evolución de las interfaces')
ax[1,0].plot(t,E,color=TEAL);ax[1,0].set(xlabel='$t$',ylabel='Energía discreta');heading(ax[1,0],'c','Disipación de energía')
ax[1,1].plot(x[::4],np.max(np.abs(runs[0][1]-runs[1][1][:,::2]),axis=0),color=ORANGE,label='401 → 801 nodos')
ax[1,1].plot(x[::2],np.max(np.abs(runs[1][1]-runs[2][1][:,::2]),axis=0),color=TEAL,label='801 → 1601 nodos')
ax[1,1].set(xlabel='$x$',ylabel=r'$\max_t |u_h-u_{h/2}|$');ax[1,1].legend(frameon=False,fontsize=9,loc='upper center');heading(ax[1,1],'d','Sensibilidad al refinamiento')
save(fig,'p3-solution')
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print('ALL SCALAR CHECKS PASS',flush=True)
