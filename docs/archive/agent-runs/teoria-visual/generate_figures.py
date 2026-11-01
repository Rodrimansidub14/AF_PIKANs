"""Analytical teaching figures; no training runs or claimed FEM results.
Run with bundled Python. All generated files and caches stay beside this file.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle

OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
TEAL, ORANGE, INK, GREY = '#137C80', '#D66A35', '#222C30', '#738084'
CMAP = LinearSegmentedColormap.from_list('teal_white_orange', [TEAL, '#FAFAF6', ORANGE])
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.labelcolor':INK,
    'text.color':INK, 'xtick.color':GREY, 'ytick.color':GREY,
    'axes.edgecolor':'#C6CDCD', 'axes.titleweight':'medium',
    'axes.titlesize':12, 'figure.facecolor':'white', 'savefig.facecolor':'white',
    'pdf.fonttype':42, 'svg.fonttype':'none', 'lines.linewidth':2.2})

def finish(fig, name):
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

def title(ax, letter, text):
    ax.set_title(f'{letter}  {text}', loc='left', pad=12)

# P1: exact manufactured fields, identical amplitude and shared color scale.
x = np.linspace(0,1,241)
X,Y = np.meshgrid(x,x)
fig, axes = plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for ax,k in zip(axes[:2],(1,4)):
    im = ax.pcolormesh(X,Y,np.sin(k*np.pi*X)*np.sin(k*np.pi*Y),cmap=CMAP,vmin=-1,vmax=1,rasterized=True)
    ax.set_aspect('equal'); ax.set(xlabel='$x$',ylabel='$y$')
    title(ax, 'a' if k==1 else 'b', f'Campo exacto · k = {k}')
fig.colorbar(im,ax=axes[:2],label='$u_k^*$',shrink=.8,ticks=[-1,0,1])
for k,c in ((1,TEAL),(4,ORANGE)):
    axes[2].plot(x,np.sin(k*np.pi*x),color=c,label=f'$k={k}$')
axes[2].axhline(0,color='#D6DADA',lw=.8)
axes[2].set(xlabel='$x$',ylabel=r'$u_k^*(x,1/(2k))$',ylim=(-1.18,1.18))
axes[2].legend(frameon=False,loc='lower left'); title(axes[2],'c','Misma amplitud, más curvatura')
finish(fig,'p1-poisson')

# P2: closed-form Burgers wave and term cancellation.
x=np.linspace(-1,1,2401); A=.5; c=.5; x0=-.25
fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
for nu,col in ((.02,TEAL),(.005,ORANGE)):
    q=A*(x-x0)/(2*nu); u=c-A*np.tanh(q)
    axes[0,0].plot(x,u,color=col,label=fr'$\nu={nu}$')
    axes[0,1].plot(x,A*A/(2*nu)/np.cosh(q)**2,color=col)
axes[0,0].set(ylabel='$u(x,0)$',xlabel='$x$'); axes[0,0].legend(frameon=False)
axes[0,1].set(xlim=(-.45,-.05),xlabel='$x$',ylabel=r'$|u_x(x,0)|$')
title(axes[0,0],'a','El frente se estrecha'); title(axes[0,1],'b','La derivada se concentra')
t=np.linspace(0,1,241); xx=np.linspace(-1,1,501); XX,TT=np.meshgrid(xx,t)
im=axes[1,0].pcolormesh(xx,t,c-A*np.tanh(A*(XX-c*TT-x0)/(.04)),cmap=CMAP,vmin=0,vmax=1,rasterized=True)
axes[1,0].plot(c*t+x0,t,color=INK,ls='--',lw=1.2)
axes[1,0].set(xlabel='$x$',ylabel='$t$'); fig.colorbar(im,ax=axes[1,0],label='$u$',ticks=[0,.5,1])
title(axes[1,0],'c',r'Trayectoria del frente · $\nu=0.02$')
nu=.02; time=.5; q=A*(x-c*time-x0)/(2*nu); S=1/np.cosh(q)**2
u=c-A*np.tanh(q); ux=-A*A/(2*nu)*S; ut=-c*ux; uxx=A**3/(2*nu**2)*S*np.tanh(q)
axes[1,1].plot(x,ut,color=TEAL,label='$u_t$')
axes[1,1].plot(x,u*ux,color=ORANGE,label='$u u_x$')
axes[1,1].plot(x,-nu*uxx,color=INK,ls='--',label=r'$-\nu u_{xx}$')
axes[1,1].set(xlim=(-.3,.3),ylim=(-4.8,3.7),xlabel='$x$',ylabel='Términos de la ecuación')
axes[1,1].legend(frameon=False,ncol=3,fontsize=9,loc='lower center')
title(axes[1,1],'d',r'Balance exacto · $t=0.5$')
assert np.max(np.abs(ut+u*ux-nu*uxx))<1e-12
finish(fig,'p2-burgers')

# P3: mechanism, not a surrogate numerical evolution of the finite-domain case.
u=np.linspace(-1.25,1.25,501); d=.001
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
axes[0].plot(u,1.25*(u*u-1)**2,color=TEAL)
axes[0].scatter([-1,1],[0,0],c=TEAL,zorder=4); axes[0].scatter([0],[1.25],facecolor='white',edgecolor=ORANGE,zorder=4)
axes[0].set(xlabel='$u$',ylabel='$W(u)$'); title(axes[0],'a','Dos estados preferidos')
axes[1].plot(u,5*(u-u**3),color=ORANGE); axes[1].axhline(0,color=GREY,lw=.8)
for v in (-1,0,1): axes[1].scatter([v],[0],color=INK,s=15,zorder=4)
axes[1].set(xlabel='$u$',ylabel=r'$-W\prime(u)=5(u-u^3)$'); title(axes[1],'b','La reacción aleja de cero')
xi=np.linspace(-.12,.12,501); ell=np.sqrt(2*d/5)
axes[2].plot(xi,np.tanh(xi/ell),color=TEAL)
axes[2].axvspan(-ell,ell,color=TEAL,alpha=.09)
axes[2].set(xlabel=r'$x-x_c$',ylabel='$u$',ylim=(-1.15,1.15))
title(axes[2],'c','Interfaz ideal en la recta')
axes[2].text(.04,-.65,r'$\ell=\sqrt{2d/5}$',fontsize=10)
finish(fig,'p3-allen-cahn')

# P4: exact geometry and prescribed inlet. No fabricated streamlines.
fig=plt.figure(figsize=(10,4.8),layout='constrained'); gs=fig.add_gridspec(2,2,height_ratios=[1,1.25])
ax=fig.add_subplot(gs[0,:]); ax.add_patch(Rectangle((0,0),2.2,.41,fill=False,edgecolor=INK,lw=1.5))
ax.add_patch(Circle((.2,.2),.05,facecolor='#DAE7E7',edgecolor=TEAL,lw=2))
for y in np.linspace(.04,.37,8):
    vel=4*.3*y*(.41-y)/.41**2
    ax.arrow(-.25,y,.6*vel,0,width=.003,head_width=.018,length_includes_head=True,color=ORANGE)
ax.text(1.1,.48,r'Paredes: $\mathbf{u}=0$',ha='center')
ax.text(.38,.19,'Cilindro fijo',fontsize=9)
ax.text(2.23,.2,'Salida\nnatural',va='center',fontsize=9)
ax.set(xlim=(-.3,2.5),ylim=(-.06,.62),xlabel='$x$ [m]',ylabel='$y$ [m]'); ax.set_aspect('equal')
title(ax,'a','DFG 2D-1 · geometría y entrada prescrita')
ax=fig.add_subplot(gs[1,0]); y=np.linspace(0,.41,201); vel=4*.3*y*(.41-y)/.41**2
ax.plot(vel,y,color=ORANGE); ax.axvline(.2,color=TEAL,ls='--',lw=1.3,label='Media: 0.2 m/s')
ax.set(xlabel='$u_{in}$ [m/s]',ylabel='$y$ [m]',xlim=(0,.34)); ax.legend(frameon=False,fontsize=9)
title(ax,'b','Perfil parabólico de entrada')
ax=fig.add_subplot(gs[1,1]); ax.axis('off'); title(ax,'c','Qué equilibra cada término')
for yy,txt,col in [(.83,r'$(\mathbf{u}\cdot\nabla)\mathbf{u}$   transporte de momento',INK),(.61,r'$-\nabla p$   fuerza por presión',TEAL),(.39,r'$\nu\Delta\mathbf{u}$   difusión viscosa',ORANGE),(.12,r'$\nabla\cdot\mathbf{u}=0$   conservación de volumen',INK)]:
    ax.text(.02,yy,txt,color=col,transform=ax.transAxes,fontsize=11)
finish(fig,'p4-dfg')

# P5 geometry/bending: dimensionless drawing, aspect ratio explicitly exaggerated.
fig,axes=plt.subplots(2,1,figsize=(10,5.8),layout='constrained')
ax=axes[0]; ax.axis('off'); ax.set(xlim=(-.9,10.9),ylim=(-.8,3.5))
for y,col,label in [(0,TEAL,'Capa inferior\n'+r'$e_{21},e_{22}>0$'),(1,ORANGE,'Capa superior\n'+r'$e_{21},e_{22}<0$')]:
    ax.add_patch(Rectangle((0,y),10,1,facecolor=col,alpha=.12,edgecolor=col))
    ax.text(7,y+.5,label,color=col,va='center')
ax.add_patch(Rectangle((-.25,-.2),.25,2.4,facecolor='#DDDDDD',hatch='////',edgecolor=GREY))
ax.plot([0,10],[2,2],color=ORANGE,lw=3);ax.plot([0,10],[0,0],color=TEAL,lw=3)
ax.text(5,2.18,r'$\varphi=100$ V · electrodo superior',ha='center')
ax.text(5,-.38,r'$\varphi=0$ V · electrodo inferior',ha='center')
for x in (2,3.5,5): ax.annotate('',(x,.3),(x,1.7),arrowprops=dict(arrowstyle='->',color=INK,lw=1.5))
ax.text(1.2,1,r'$\mathbf{E}$',va='center'); ax.text(-.48,1,'Empotramiento',rotation=90,ha='center',va='center',fontsize=9)
ax.text(10.3,1,r'$\mathbf{D}\cdot n=0$',rotation=90,va='center',fontsize=10)
ax.text(5.8,3.05,r'Ambos extremos: $D_x=0$ · caras y punta: $\sigma n=0$',ha='center',fontsize=9)
ax.annotate('',(.7,2.95),(0,2.95),arrowprops=dict(arrowstyle='->',color=INK))
ax.annotate('',(0,3.4),(0,2.95),arrowprops=dict(arrowstyle='->',color=INK))
ax.text(.8,2.9,'$x$',fontsize=10); ax.text(-.18,3.4,'$y$',fontsize=10)
title(ax,'a','Bimorfo PVDF · altura exagerada · L = 0.1 m, H = 0.001 m')
ax=axes[1]; ax.axis('off'); ax.set(xlim=(-.9,10.9),ylim=(-2,1.5))
xx=np.linspace(0,10,201); bend=-.012*xx**2
ax.fill_between(xx,bend-.25,bend,color=TEAL,alpha=.18); ax.fill_between(xx,bend,bend+.25,color=ORANGE,alpha=.18)
for yy,col in [(-.25,TEAL),(0,INK),(.25,ORANGE)]: ax.plot(xx,bend+yy,color=col,lw=1.3)
ax.plot([0,10],[0,0],color=GREY,ls='--',lw=1)
ax.add_patch(Rectangle((-.25,-.45),.25,.9,facecolor='#DDDDDD',hatch='////',edgecolor=GREY))
ax.text(3,.65,'Tendencias libres opuestas + unión entre capas',ha='left')
ax.text(5,-1.8,'Curvatura ilustrativa: no es una solución FEM ni una flecha calculada.',ha='center',fontsize=10)
title(ax,'b','Compatibilidad mecánica: la unión transforma extensión en flexión')
finish(fig,'p5-bimorfo')

# Interface teaching counterexample: scalar dielectric transmission, not full PVDF.
y=np.linspace(0,1,501); k1=1.; k2=2.; J=-1/(.5/k1+.5/k2)
phi=np.where(y<=.5,-J*y/k1,-J*.5/k1-J*(y-.5)/k2)
grad=np.where(y<.5,-J/k1,-J/k2)
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for ax in axes:
    ax.axvspan(0,.5,color=TEAL,alpha=.06);ax.axvspan(.5,1,color=ORANGE,alpha=.06)
    ax.axvline(.5,color=GREY,ls='--',lw=1);ax.set(xlabel='$y/H$',xlim=(0,1))
axes[0].plot(y,phi,color=INK);axes[0].set(ylabel=r'$\varphi/V$');title(axes[0],'a','Potencial continuo')
axes[1].plot([0,.5],[4/3,4/3],color=TEAL);axes[1].plot([.5,1],[2/3,2/3],color=ORANGE)
axes[1].set(ylabel=r'$\partial_{(y/H)}(\varphi/V)$',ylim=(0,1.6));title(axes[1],'b','La pendiente puede saltar')
axes[2].plot(y,np.full_like(y,J),color=INK);axes[2].set(ylabel=r'$D_y/(\kappa_1 V/H)$',ylim=(-1.8,-.8));title(axes[2],'c','Flujo normal continuo')
assert abs(phi[0])<1e-14 and abs(phi[-1]-1)<1e-14
finish(fig,'p5-interfaz')
print('Generated 6 figures, each PDF/SVG/PNG. Analytic Burgers residual and interface values verified.')
