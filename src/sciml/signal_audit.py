"""Post-hoc spatial signal audit. Never imported by training or sampling code."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def localization(signal,error):
    s=np.asarray(signal,dtype=float).ravel()
    e=np.asarray(error,dtype=float).ravel()
    if s.shape!=e.shape or not s.size or not np.isfinite(s).all() or not np.isfinite(e).all():
        raise ValueError('matching finite nonempty arrays required')
    if np.ptp(s)==0 or np.ptp(e)==0:
        return {'spearman':None,'top_decile_overlap':None,'error_enrichment':None,
                'signal_tail_fraction':None,'error_tail_fraction':None,'points':int(s.size)}
    a=s>=np.quantile(s,.9); b=e>=np.quantile(e,.9)
    return {'spearman':float(spearmanr(s,e).statistic),
            'top_decile_overlap':float(np.count_nonzero(a&b)/np.count_nonzero(b)),
            'error_enrichment':float(e[a].mean()/e.mean()) if e.mean()>0 else None,
            'signal_tail_fraction':float(a.mean()),'error_tail_fraction':float(b.mean()),
            'points':int(s.size)}


def audit(base,output):
    import pandas as pd
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; costs=[]; overlap=[]; hashes={}; figures=[]
    import plotly.graph_objects as go
    for family in ('rbf_fixed','mlp_tanh','bspline_cubic'):
        for problem in ('heat','burgers'):
            tails={}
            for seed in (101,102,103):
                root=base/'models'/family/'download-extracted-bundle-logscale-v1/runs'/f'seed-{seed}'/problem
                path=root/'validation.npz'
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as d:
                    if bool(d['final_test_used']):
                        raise ValueError('audit is restricted to saved validation data')
                    # Exclude IC/boundary traces to avoid exact zeros driving ranks.
                    residual=np.abs(d['residual'][1:-1,1:]); error=d['absolute_error'][1:-1,1:]
                    rows.append(dict(model=family,problem=problem,seed=seed,scope='interior',
                                     **localization(residual,error)))
                    for label,target in (('early',d['t'][1:]<1/np.pi),('late',d['t'][1:]>=1/np.pi)):
                        rows.append(dict(model=family,problem=problem,seed=seed,scope=label,
                                         **localization(residual[:,target],error[:,target])))
                    tails[seed]=residual>=np.quantile(residual,.9)
                    if family=='rbf_fixed' and problem=='burgers':
                        positive=(residual.ravel()>0)&(error.ravel()>0)
                        rr=residual.ravel()[positive][::32]; ee=error.ravel()[positive][::32]
                        figures.append(go.Figure(go.Scattergl(x=rr,y=ee,mode='markers',marker={'size':3})).update_layout(
                            title=f'Burgers RBF seed {seed}: residual vs error (subsampled display only)',
                            xaxis={'title':'Absolute PDE residual','type':'log'},
                            yaxis={'title':'Absolute reference error','type':'log'}))
                summary=json.loads((root/'training_summary.json').read_text())
                diag=json.loads((root/'diagnostic_signals.json').read_text())
                costs.append(dict(model=family,problem=problem,seed=seed,
                    steps=summary['steps'],seconds=summary['training_and_diagnostics_sec'],
                    diagnostic_seconds=summary['diagnostic_sec'],
                    diagnostic_fraction=summary['diagnostic_sec']/summary['training_and_diagnostics_sec'],
                    **{k:v for k,v in diag.items() if 'occupancy' in k or 'effective_basis' in k or 'sensitivity/' in k}))
            for a,b in itertools.combinations(tails,2):
                intersection=np.count_nonzero(tails[a]&tails[b]); union=np.count_nonzero(tails[a]|tails[b])
                overlap.append(dict(model=family,problem=problem,seed_a=a,seed_b=b,
                                    residual_tail_jaccard=float(intersection/union)))
    table=pd.DataFrame(rows); resources=pd.DataFrame(costs); stable=pd.DataFrame(overlap)
    table.to_csv(output/'localization.csv',index=False)
    resources.to_csv(output/'cost-and-basis.csv',index=False)
    stable.to_csv(output/'hotspot-reproducibility.csv',index=False)
    selected=table[(table.model=='rbf_fixed')&(table.problem=='burgers')&(table.scope=='interior')]
    decision={'independent_unit':'seed/run; spatial points and time scopes are not replicates',
              'seeds_per_arm':3,'scope':'post-hoc descriptive pilot, not confirmatory',
              'residual_spearman_min':float(selected.spearman.min()),
              'residual_spearman_max':float(selected.spearman.max()),
              'enrichment_min':float(selected.error_enrichment.min()),
              'enrichment_max':float(selected.error_enrichment.max()),
              'next_action':'controlled RAD trial warranted; no superiority or acceptance inferred',
              'scientific_accepted':False,'source_sha256':hashes}
    (output/'audit.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    html='<html><meta charset="utf-8"><title>EXP-0005 signal audit</title><body><h1>Residual localization audit</h1>'
    html+='<p>Descriptive only: three independent seeds per model/problem. Spatial cells are not replicates. Evaluation error never enters RAD.</p>'
    html+=table.to_html(index=False,float_format=lambda v:f'{v:.5g}')
    html+='<h2>Reproducibility of residual hotspots across seeds</h2>'+stable.to_html(index=False)
    html+='<h2>Measured diagnostic cost and basis signals</h2>'+resources.to_html(index=False)
    for i,fig in enumerate(figures): html+=fig.to_html(full_html=False,include_plotlyjs=True if i==0 else False)
    (output/'audit.html').write_text(html+'</body></html>',encoding='utf-8')
    print(json.dumps({k:v for k,v in decision.items() if k!='source_sha256'},indent=2))
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--base',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); audit(args.base,args.out)
