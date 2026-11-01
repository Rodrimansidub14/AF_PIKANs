"""CPU: which (e31,e33) sign convention makes the FEM field satisfy div(sigma)=0, div(D)=0 under the PINN's constitutive law?"""
import numpy as np, itertools, json
C11=2.0e9/(1-0.29**2); C12=0.29*C11; G=0.775e9; E31=C12*(-3e-11)+C11*2.2e-11; E33=C12*2.2e-11+C11*(-3e-11)
eps=12*8.854187817620e-12  # eps1=eps2 in loss (D_const_strain; checked below)
Dcs=eps-(2.2e-11*(C11*2.2e-11+C12*-3e-11)+(-3e-11)*(C12*2.2e-11+C11*-3e-11))
print('e31',E31,'e33',E33,'eps_stress',eps,'eps_strain',Dcs)
out={}
for fem in ('fem-grid-indirect-polingminus.npz','fem-grid-indirect.npz'):
    z=np.load(fem); xs=np.linspace(0,0.1,201); ys=np.linspace(0,1e-3,21)
    sh=(21,201); u=z['u'].reshape(sh); v=z['v'].reshape(sh); p=z['phi'].reshape(sh)
    dx=xs[1]-xs[0]; dy=ys[1]-ys[0]
    gx=lambda f:np.gradient(f,dx,axis=1); gy=lambda f:np.gradient(f,dy,axis=0)
    exx=gx(u); eyy=gy(v); exy=0.5*(gy(u)+gx(v)); Ey=-gy(p); Ex=-gx(p)
    top=(ys>0.5e-3)[:,None]*np.ones(sh)
    for e_label,epsv in (('eps_stress',eps),('eps_strain',Dcs)):
      for s31,s33 in itertools.product((1,-1),(1,-1)):
        sg=np.where(top>0,-1.0,1.0)
        e31=sg*s31*abs(E31); e33=sg*s33*abs(E33)
        sx=C11*exx+C12*eyy-e31*Ey; sz=C12*exx+C11*eyy-e33*Ey; tau=G*exy
        Dx=epsv*Ex; Dy=e31*exx+e33*eyy+epsv*Ey
        r1=gx(sx)+gy(tau); r2=gx(tau)+gy(sz); r3=gx(Dx)+gy(Dy)
        sl=(slice(3,-3),slice(5,-5))  # interior; layer interface rows are noisy: mask rows 9..11
        m=np.ones(sh,bool); m[9:12]=False; m[:3]=False; m[-3:]=False; m[:,:5]=False; m[:,-5:]=False
        n=lambda a:float(np.linalg.norm(a[m]))
        sc_s=n(np.abs(sx)+np.abs(sz)+np.abs(tau))/1e-3; sc_d=n(np.abs(Dx)+np.abs(Dy))/1e-3
        key=f'{fem[9:-4]}|{e_label}|s31={s31:+d},s33={s33:+d}'
        out[key]=dict(mech=(n(r1)+n(r2))/sc_s, elec=n(r3)/sc_d)
for k,v in sorted(out.items(),key=lambda kv:kv[1]['mech']+kv[1]['elec']): print(f"{k:60s} mech {v['mech']:.3e} elec {v['elec']:.3e}")
json.dump(out,open('sign-check.json','w'),indent=1)
