import json, csv, glob, statistics as st
rows=[]
for f in sorted(glob.glob('outputs/dfgl/results/corrected-*/run.json')):
    r=json.load(open(f))
    for stage in ('adam_final','lbfgs_final'):
        s=r['score_'+stage]
        rows.append(dict(arm=r['arm'],seed=r['seed'],stage=stage,params=r['params'],u=s['u_relL2'],v=s['v_relL2'],p=s['p_relL2_gauge'],Cd=s['Cd'],Cd_err=s['Cd_relerr'],Cl_abs=s['Cl_abserr'],dp_err=s['dp_relerr'],
          lbfgs_status=r['lbfgs_status'],lbfgs_iters=r['lbfgs_iterations'],adam_s=round(r['adam_seconds']),lbfgs_s=round(r['lbfgs_seconds'])))
w=csv.DictWriter(open('dfg-long-per-seed.csv','w',newline=''),fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('arm stage n  u  v  p  Cd_err  Cl_abs  dp_err')
for arm in('mlp','rbf'):
  for stage in('adam_final','lbfgs_final'):
    R=[x for x in rows if x['arm']==arm and x['stage']==stage]
    f=lambda k:f"{st.mean(x[k] for x in R):.4g}±{st.stdev(x[k] for x in R):.2g}"
    print(arm,stage,len(R),*[f(k) for k in('u','v','p','Cd_err','Cl_abs','dp_err')])
for x in rows:
    if x['stage']=='lbfgs_final': print(x['arm'],x['seed'],f"u={x['u']:.4f} Cd_err={x['Cd_err']:.4f}",x['lbfgs_status'],x['lbfgs_iters'],x['adam_s'],x['lbfgs_s'])
