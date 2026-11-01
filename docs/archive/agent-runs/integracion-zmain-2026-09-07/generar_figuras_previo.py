"""Figuras matemáticas y demostración FEM; no ejecuta entrenamiento de PIKAN."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.tri import Triangulation
from scipy.interpolate import BSpline
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figuras'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                    'axes.spines.right': False, 'figure.dpi': 160})
BLUE, TEAL, GOLD = '#254b73', '#247e77', '#c18b32'
records = {}

def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=190, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

def box(ax, x, y, text, w=2.3, h=.75, color=BLUE):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.06',
                              facecolor=color,edgecolor='none'))
    ax.text(x,y,text,ha='center',va='center',color='white',fontsize=10)

def arrow(ax, a, b, text=None):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#657786','lw':1.5})
    if text: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,text,ha='center',fontsize=8)

def network():
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for ax,kind in zip(axes,['MLP','KAN']):
        coords=[[(0,y) for y in [-.6,.6]],[(1.7,y) for y in [-1,0,1]],[(3.4,0)]]
        for layer in range(2):
            for a in coords[layer]:
                for b in coords[layer+1]:
                    ax.plot([a[0],b[0]],[a[1],b[1]],color=BLUE if kind=='MLP' else TEAL,lw=1.1)
        for layer in coords:
            for x,y in layer: ax.add_patch(Circle((x,y),.17,facecolor='white',edgecolor=BLUE,lw=2))
        ax.text(1.7,1.55,'MLP: '+r'$h=\sigma(Wh+b)$' if kind=='MLP' else 'KAN: '+r'$h_j=\sum_i\varphi_{ji}(h_i)$',ha='center',fontsize=13)
        ax.text(1.7,-1.65,'Aristas: pesos aprendibles\nNodos: activación fija' if kind=='MLP' else 'Aristas: funciones aprendibles\nNodos: suma',ha='center')
        ax.set(xlim=(-.5,3.9),ylim=(-2,2)); ax.axis('off')
    save(fig,'arquitecturas')

def diagrams():
    fig,ax=plt.subplots(figsize=(10,3.8));ax.set(xlim=(-1.5,9),ylim=(-1.7,2.1));ax.axis('off')
    box(ax,0,0,'Coordenadas\ninterior / frontera',w=2.1)
    box(ax,2.8,0,'Aproximador\nMLP o KAN',w=2)
    box(ax,5.5,1,'Derivación automática\nresidual de PDE',w=2.5,color=TEAL)
    box(ax,5.5,-1,'Trazas / condición\ninicial y de frontera',w=2.5,color=TEAL)
    box(ax,8,0,'Pérdida\nadimensional',w=1.6)
    arrow(ax,(1.1,0),(1.7,0));arrow(ax,(3.9,.1),(4.2,.8));arrow(ax,(3.9,-.1),(4.2,-.8))
    arrow(ax,(6.8,1),(7.2,.25));arrow(ax,(6.8,-1),(7.2,-.25))
    ax.text(4,1.85,'Referencia exacta/FEM: solo evaluación independiente',ha='center',color=GOLD)
    save(fig,'perdida_pinn')
    fig,ax=plt.subplots(figsize=(10,4.2));ax.set(xlim=(-1.5,10.5),ylim=(-2.8,2));ax.axis('off')
    for x,t,c in [(0,'Estado actual\nparámetros y geometría',BLUE),(3,'Diagnóstico sin\nreferencia exacta',TEAL),(6,'Acción candidata\nutilidad / costo',TEAL),(9,'Transferencia\ny verificación',BLUE)]:
        box(ax,x,.8,t,w=2.45 if x==0 else 2.3,color=c)
    for a,b in [(1.3,1.8),(4.2,4.8),(7.2,7.8)]:arrow(ax,(a,.8),(b,.8))
    box(ax,6,-1.1,'Aceptar o revertir\nsegún restricciones',w=2.7,color=TEAL)
    box(ax,1.8,-1.1,'Registrar resultado\nContinuar entrenamiento',w=2.6,color=GOLD)
    arrow(ax,(9,.35),(7.4,-1));arrow(ax,(4.6,-1.1),(3.2,-1.1));arrow(ax,(.45,-1.1),(0,.35))
    ax.text(4,-2.3,'Entrenar → diagnosticar → proponer → comprobar → aceptar o revertir',ha='center')
    save(fig,'controlador')

def bases():
    fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    t=np.r_[np.zeros(4),np.linspace(0,1,9)[1:-1],np.ones(4)]
    x=np.linspace(0,1,801);n=len(t)-4
    basis=BSpline(t,np.eye(n),3)(x)
    records['spline_partition_error']=float(np.max(np.abs(basis.sum(1)-1)))
    assert records['spline_partition_error']<1e-12
    for i in range(n):axes[0].plot(x,basis[:,i],lw=1)
    axes[0].set_title('Bases cúbicas');axes[0].set_xlabel('Coordenada de arista')
    coeff=np.sin(np.linspace(0,2*np.pi,n))
    axes[1].plot(x,basis@coeff,color=BLUE,lw=2);axes[1].set_title('Suma ponderada');axes[1].set_xlabel('Coordenada de arista')
    for c in [.3,.65]:
        q=(x-c)/.22;y=np.zeros_like(x);mask=np.abs(q)<1;y[mask]=np.exp(-1/(1-q[mask]**2))
        axes[2].plot(x,y,label=f'Centro {c}')
    axes[2].set_title('Alternativa compacta');axes[2].legend(fontsize=8);axes[2].set_xlabel('Coordenada de arista')
    fig.tight_layout();save(fig,'bases')

def exact_solutions():
    fig,axes=plt.subplots(1,3,figsize=(10,3.3));x=np.linspace(-1,1,1501)
    axes[0].plot((x+1)/2,np.sin(8*np.pi*(x+1)/2),color=BLUE);axes[0].set_title('Poisson: corte oscilatorio')
    for nu in [.1,.05,.02]:axes[1].plot(x,-np.tanh(x/(2*nu)),label=f'ν={nu}')
    axes[1].set_title('Burgers estacionaria');axes[1].legend(fontsize=8)
    for time in [0,.5,1]:axes[2].plot(x,np.tanh((x-(-.5+time))/(np.sqrt(2)*.05)),label=f't={time}')
    axes[2].set_title('Allen–Cahn manufacturada');axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('x');ax.set_ylabel('u exacta');ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'soluciones_exactas')

def piezo():
    fig,ax=plt.subplots(figsize=(10,3.5));ax.set(xlim=(-.4,10),ylim=(-1.2,3));ax.axis('off')
    ax.add_patch(Rectangle((0,0),5,.7,facecolor='#bedee2',edgecolor=BLUE))
    ax.add_patch(Rectangle((0,.7),5,.7,facecolor='#ead5b0',edgecolor=BLUE))
    ax.plot([0,0],[-.2,1.6],color=BLUE,lw=6)
    ax.text(2.5,1.05,'Capa superior: e⁺',ha='center');ax.text(2.5,.3,'Capa inferior: e⁻ = −e⁺',ha='center')
    ax.text(2.5,1.9,'Potencial impuesto arriba y abajo',ha='center')
    ax.text(2.5,-.55,'Interfaz: continuidad de d, φ, σn y D·n',ha='center')
    arrow(ax,(5,.7),(5,1.8));ax.text(5.1,1.7,'Tracción',fontsize=9)
    box(ax,8,1.65,'d → deformación S',w=2.6)
    box(ax,8,.45,'Constitutivo C, e, κ\nσ y D',w=2.6,color=TEAL)
    box(ax,8,-.75,'φ → campo E = −∇φ',w=2.6)
    arrow(ax,(8,1.2),(8,.9));arrow(ax,(8,-.3),(8,0))
    save(fig,'piezo_esquema')

# Dunavant: quadrature normalized to unit total on each triangle.
BARY=np.array([[1/3]*3,[.05971587178977,.470142064105115,.470142064105115],
 [.470142064105115,.05971587178977,.470142064105115],[.470142064105115,.470142064105115,.05971587178977],
 [.797426985353087,.101286507323456,.101286507323456],[.101286507323456,.797426985353087,.101286507323456],
 [.101286507323456,.101286507323456,.797426985353087]])
W=np.array([.225]+[.132394152788506]*3+[.125939180544827]*3)

def fem(n):
    grid=np.linspace(0,1,n+1);xx,yy=np.meshgrid(grid,grid)
    xy=np.c_[xx.ravel(),yy.ravel()];tri=[]
    for j in range(n):
        for i in range(n):
            k=j*(n+1)+i;tri += [[k,k+1,k+n+2],[k,k+n+2,k+n+1]]
    tri=np.array(tri);rows=[];cols=[];vals=[];rhs=np.zeros(len(xy))
    for ids in tri:
        pts=xy[ids];M=np.c_[np.ones(3),pts];area=abs(np.linalg.det(M))/2
        grads=np.linalg.inv(M)[1:,:];ke=area*grads.T@grads
        qp=BARY@pts;fq=2*np.pi**2*np.sin(np.pi*qp[:,0])*np.sin(np.pi*qp[:,1])
        rhs[ids]+=area*(BARY.T@(W*fq))
        for a in range(3):
            for b in range(3):rows.append(ids[a]);cols.append(ids[b]);vals.append(ke[a,b])
    K=coo_matrix((vals,(rows,cols)),shape=(len(xy),len(xy))).tocsr()
    free=np.where((xy[:,0]>0)&(xy[:,0]<1)&(xy[:,1]>0)&(xy[:,1]<1))[0]
    u=np.zeros(len(xy));u[free]=spsolve(K[free][:,free],rhs[free])
    l2=h1=norm=0.
    for ids in tri:
        pts=xy[ids];M=np.c_[np.ones(3),pts];area=abs(np.linalg.det(M))/2;qp=BARY@pts
        exact=np.sin(np.pi*qp[:,0])*np.sin(np.pi*qp[:,1]);diff=BARY@u[ids]-exact
        grad_exact=np.c_[np.pi*np.cos(np.pi*qp[:,0])*np.sin(np.pi*qp[:,1]),np.pi*np.sin(np.pi*qp[:,0])*np.cos(np.pi*qp[:,1])]
        grad=np.linalg.inv(M)[1:,:]@u[ids]
        l2+=area*np.dot(W,diff**2);norm+=area*np.dot(W,exact**2)
        h1+=area*np.dot(W,np.sum((grad_exact-grad)**2,axis=1))
    res=float(np.linalg.norm((K@u-rhs)[free])/np.linalg.norm(rhs[free]))
    return xy,tri,u,{'n':n,'nodes':len(xy),'triangles':len(tri),'relative_l2':float(np.sqrt(l2/norm)),'h1_seminorm_error':float(np.sqrt(h1)),'linear_relative_residual':res}

def fem_figures():
    levels=[]
    for n in [8,16,32]:
        xy,tri,u,metrics=fem(n);levels.append(metrics)
    for a,b in zip(levels[:-1],levels[1:]):
        b['l2_order']=float(np.log2(a['relative_l2']/b['relative_l2']))
        b['h1_order']=float(np.log2(a['h1_seminorm_error']/b['h1_seminorm_error']))
    assert levels[-1]['l2_order']>1.8 and levels[-1]['h1_order']>.9
    assert max(r['linear_relative_residual'] for r in levels)<1e-10
    records['fem_poisson']=levels
    mesh=Triangulation(xy[:,0],xy[:,1],tri);exact=np.sin(np.pi*xy[:,0])*np.sin(np.pi*xy[:,1])
    fig,axes=plt.subplots(1,3,figsize=(10,3.3))
    axes[0].triplot(mesh,lw=.25,color=BLUE);axes[0].set_title('Malla P₁: 2048 triángulos')
    for ax,data,title in [(axes[1],u,'Solución FEM'),(axes[2],np.abs(u-exact),'Error absoluto nodal')]:
        h=ax.tripcolor(mesh,data,shading='gouraud',cmap='viridis');fig.colorbar(h,ax=ax,shrink=.75);ax.set_title(title)
    for ax in axes:ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y')
    fig.tight_layout();save(fig,'fem_poisson')
    fig,ax=plt.subplots(figsize=(5,3.4));hs=[1/r['n'] for r in levels]
    ax.loglog(hs,[r['relative_l2'] for r in levels],'o-',label='Error relativo L²')
    ax.loglog(hs,[r['h1_seminorm_error'] for r in levels],'s-',label='Error seminorma H¹')
    ax.set(xlabel='Tamaño h',ylabel='Error',title='Convergencia de la demostración FEM');ax.legend();ax.grid(alpha=.3)
    save(fig,'fem_convergencia')

if __name__=='__main__':
    network();diagrams();bases();exact_solutions();piezo();fem_figures()
    records['scope']='Illustrative mathematics and Poisson FEM only; no mechanism training'
    (ROOT/'validacion-figuras.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))
