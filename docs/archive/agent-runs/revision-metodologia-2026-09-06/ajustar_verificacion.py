from pathlib import Path
p=Path('verificar_documento.py');z=p.read_text(encoding='utf-8');z=z.replace('import sympy as s','import numpy as np')
a=z.index('x,y,t,nu,eps=');b=z.index("result['figure_calculations']",a)
z=z[:a]+'''rng=np.random.default_rng(617)
x=rng.uniform(.1,.9,200);y=rng.uniform(.1,.9,200);t=rng.uniform(.05,.95,200)
h=2e-4
# Fourth-order finite differences, independent of the formula derivations.
def d1(f,x): return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)
def d2(f,x): return (-f(x+2*h)+16*f(x+h)-30*f(x)+16*f(x-h)-f(x-2*h))/(12*h*h)
checks={}
u=lambda x,y:np.sin(4*np.pi*x)*np.sin(4*np.pi*y)
r=-d2(lambda z:u(z,y),x)-d2(lambda z:u(x,z),y)-2*(4*np.pi)**2*u(x,y)
checks['poisson_distribuido']=float(np.max(np.abs(r))/(2*(4*np.pi)**2))
a=100;c=.5;q=x*(1-x);d=x-c
f=np.exp(-a*d*d)*(2+4*a*d*(1-2*x)+(2*a-4*a*a*d*d)*q)
u=lambda z:z*(1-z)*np.exp(-a*(z-c)**2)
checks['poisson_localizado']=float(np.max(np.abs(-d2(u,x)-f))/max(1,np.max(np.abs(f))))
x=rng.uniform(-.8,.8,200);nu=.05
u=lambda z:-np.tanh(z/(2*nu))
checks['burgers']=float(np.max(np.abs(u(x)*d1(u,x)-nu*d2(u,x))))
eps=.05
u=lambda z,t:np.tanh((z+.5-t)/(np.sqrt(2)*eps))
f=-(1-u(x,t)**2)/(np.sqrt(2)*eps)
r=d1(lambda z:u(x,z),t)-eps**2*d2(lambda z:u(z,t),x)+u(x,t)**3-u(x,t)-f
checks['allen_cahn']=float(np.max(np.abs(r)))
Re=40;lam=Re/2-np.sqrt(Re**2/4+4*np.pi**2)
vx=lambda x,y:1-np.exp(lam*x)*np.cos(2*np.pi*y)
vy=lambda x,y:lam/(2*np.pi)*np.exp(lam*x)*np.sin(2*np.pi*y)
pres=lambda x,y:(1-np.exp(2*lam*x))/2
x=rng.uniform(-.4,.9,200);y=rng.uniform(-.4,1.4,200)
checks['kovasznay_continuidad']=float(np.max(np.abs(d1(lambda z:vx(z,y),x)+d1(lambda z:vy(x,z),y))))
for name,v,dp in [('x',vx,d1(lambda z:pres(z,y),x)),('y',vy,d1(lambda z:pres(x,z),y))]:
    r=vx(x,y)*d1(lambda z:v(z,y),x)+vy(x,y)*d1(lambda z:v(x,z),y)+dp-(d2(lambda z:v(z,y),x)+d2(lambda z:v(x,z),y))/Re
    checks['kovasznay_momento_'+name]=float(np.max(np.abs(r)))
assert all(v<1e-4 for v in checks.values()),checks
result['independent_finite_difference_checks']={'errors':checks,'threshold':1e-4,'method':'fourth-order central differences, h=2e-4, 200 seeded interior points'}
''' +z[b:]
p.write_text(z,encoding='utf-8')
