"""Recreaciones trazables; no entrena ni representa resultados del mecanismo."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from scipy.integrate import solve_ivp
from scipy.sparse import diags
ROOT=Path(__file__).resolve().parents[3]
RUN=Path(__file__).resolve().parent
OUT=ROOT/'figuras/revision2026'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    for ext in ['pdf','png']: fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=190)
    plt.close(fig)
def ac(n,tol):
    x=np.linspace(-1,1,n+1);dx=2/n;y0=x[1:-1]**2*np.cos(np.pi*x[1:-1])
    L=diags([np.ones(n-2),-2*np.ones(n-1),np.ones(n-2)],[-1,0,1],format='csc')*.001/dx**2
    b=np.zeros(n-1);b[[0,-1]]=-.001/dx**2
    def rhs(t,u):return L@u+b+5*(u-u**3)
    def jac(t,u):return L+diags(5-15*u*u,format='csc')
    sol=solve_ivp(rhs,[0,1],y0,method='BDF',jac=jac,rtol=tol,atol=tol/100,t_eval=np.linspace(0,1,201))
    assert sol.success
    u=np.vstack([-np.ones(201),sol.y,-np.ones(201)])
    return x,sol.t,u
levels=[ac(n,1e-9) for n in [256,512,1024]]
diff=[]
for lo,hi in zip(levels[:-1],levels[1:]):
    diff.append(float(np.linalg.norm(hi[2][::2]-lo[2])/np.linalg.norm(hi[2][::2])))
timeref=ac(1024,1e-10)
temporal=float(np.linalg.norm(timeref[2]-levels[-1][2])/np.linalg.norm(timeref[2]))
x,t,u=levels[-1]
np.savez_compressed(RUN/'allen_cahn_recreacion.npz',x=x,t=t,u=u)
fig,axs=plt.subplots(3,1,figsize=(9,9),gridspec_kw={'height_ratios':[1,1.3,.9]})
xB=np.linspace(-1,1,1200)
for time in [0,.5,1]:
    axs[0].plot(xB,.5-.5*np.tanh(.5*(xB-.5*time+.25)/(.01)),label=f't = {time:g}')
axs[0].set(xlabel='x',ylabel='u',title='Burgers: onda exacta, ν = 0.005, c = 0.5');axs[0].legend(ncol=3)
im=axs[1].pcolormesh(t,x,u,cmap='coolwarm',vmin=-1,vmax=1,shading='auto',rasterized=True)
fig.colorbar(im,ax=axs[1],label='u');axs[1].set(xlabel='t',ylabel='x',title='Allen–Cahn: integración BDF / diferencias finitas, d = 0.001')
ax=axs[2];ax.add_patch(Rectangle((0,0),2.2,.41,facecolor='#edf3f7',edgecolor='#254b73'))
ax.add_patch(Circle((.2,.2),.05,facecolor='white',edgecolor='#254b73',lw=2))
for y in [.07,.14,.205,.27,.34]:ax.annotate('',xy=(.09,y),xytext=(-.09,y),arrowprops={'arrowstyle':'->','color':'#247e77'})
ax.annotate('r = 0.05',xy=(.23,.23),xytext=(.45,.34),arrowprops={'arrowstyle':'->'})
ax.text(1.45,.19,'Salida natural',ha='center');ax.set(xlim=(-.12,2.23),ylim=(-.06,.48),xlabel='x (m)',ylabel='y (m)',title='DFG 2D-1: dominio del benchmark Re20 (sin campo simulado)');ax.set_aspect('equal')
fig.tight_layout(h_pad=2);save(fig,'problemas')
def box(ax,x,y,w,h,txt,color='#254b73'):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='none'))
    ax.text(x,y,txt,ha='center',va='center',color='white',fontsize=10)
def arrow(ax,x0,y0,x1,y1):ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops={'arrowstyle':'->','color':'#59717f','lw':1.5})
fig,ax=plt.subplots(figsize=(10,6));ax.set(xlim=(-.5,10),ylim=(-.5,6));ax.axis('off')
box(ax,1,5,2,1,'Coordenadas\ny material');box(ax,4.1,5,2.5,1,'Ramas KAN\npor campo / capa');box(ax,8,5,2.5,1,'Derivadas y\nresiduales físicos')
arrow(ax,2.1,5,2.8,5);arrow(ax,5.4,5,6.7,5)
box(ax,8,3,2.5,1,'Activaciones + señal\nProponer 2 acciones','#247e77');arrow(ax,8,4.4,8,3.6)
box(ax,4.1,3,2.5,1,'Transferir\nVerificar derivadas','#247e77');arrow(ax,6.7,3,5.4,3)
box(ax,1,3,2,1,'Estado previo\ncompleto');arrow(ax,1,4.4,1,3.6);arrow(ax,2.1,3,2.8,3)
box(ax,4.1,1,2.7,1,'Sondeos emparejados\nAcción / sin cambio','#247e77');arrow(ax,4.1,2.4,4.1,1.6)
box(ax,8,1,2.5,1,'Aceptar o revertir\nRegistrar costo');arrow(ax,5.5,1,6.7,1)
ax.text(.1,.3,'Referencia exacta / FEM:\nsolo evaluación externa',color='#8a5a13',fontsize=10)
ax.annotate('',xy=(9.4,5),xytext=(9.4,1),arrowprops={'arrowstyle':'->','color':'#59717f'});ax.text(9.6,3,'siguiente evento',rotation=90,va='center',fontsize=9)
save(fig,'mecanismo')
fig,ax=plt.subplots(figsize=(10,3.4));ax.set(xlim=(-.5,10),ylim=(-.3,3.3));ax.axis('off')
box(ax,1,1.5,2,1,'Coordenadas\nz = (x, t)');box(ax,4,1.5,2.2,1,'Red de solución\nuθ(z)')
box(ax,7,2.5,2.3,.8,'Operador / fuente\nr = N[uθ] − f','#247e77');box(ax,7,.5,2.3,.8,'Frontera / inicial\nB[uθ] − g','#247e77')
arrow(ax,2.1,1.5,2.8,1.5);arrow(ax,5.2,1.7,5.8,2.3);arrow(ax,5.2,1.3,5.8,.7)
box(ax,9.3,1.5,1.1,1,'Pérdida\nL');arrow(ax,8.2,2.3,8.7,1.8);arrow(ax,8.2,.7,8.7,1.2)
ax.text(4,3.05,'Derivadas por regla de la cadena',ha='center',fontsize=10)
save(fig,'perdida')
checks={'allen_cahn':{'n':[256,512,1024],'successive_relative_l2_space_time':diff,'time_tolerance_change':temporal,'observed_order':float(np.log2(diff[0]/diff[1])),'illustration_only':True},'mechanism_training_executed':False}
(RUN/'validacion-numerica.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
