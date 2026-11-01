from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(r'C:\Users\rodri\Documents\AF_PIKANs')
OUT = ROOT / 'var' / 'agent_runs' / 'research-progress-brief-20260923'
FIG = OUT / 'figures'
HTML = OUT / 'plotly'
FIG.mkdir(parents=True, exist_ok=True)
HTML.mkdir(parents=True, exist_ok=True)

ARCH_CSV = ROOT / 'var/agent_runs/heat-burgers-architecture-screen/analysis/seed-results.csv'
PRIMARY_CSV = ROOT / 'var/agent_runs/heat-burgers-architecture-screen/analysis/primary-summary.csv'
LOCAL_CSV = ROOT / 'var/agent_runs/burgers-rad-exp006/audit/localization.csv'
HOTSPOT_CSV = ROOT / 'var/agent_runs/burgers-rad-exp006/audit/hotspot-reproducibility.csv'
COST_CSV = ROOT / 'var/agent_runs/burgers-rad-exp006/audit/cost-and-basis.csv'
EXP2 = ROOT / 'var/agent_runs/heat-burgers-002-t4x2/download-extracted/runs'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'figure.titlesize':15})
COLORS = {'rbf_fixed':'#2878B5','mlp_tanh':'#E27A3F','bspline_cubic':'#3A9D75'}
LABELS = {'rbf_fixed':'RBF-PIKAN fija','mlp_tanh':'MLP tanh','bspline_cubic':'PIKAN B-spline'}

# Parse and summarize seed-level screening data directly; do not use cached prose summaries.
df = pd.read_csv(ARCH_CSV)
assert len(df) == 9, f'Expected 9 worker-seed rows (18 problem runs), found {len(df)}'
assert set(df['seed'].astype(int)) == {101,102,103}
assert set(df['model_id']) == set(LABELS)

summary = {'worker_seed_rows':int(len(df)),'problem_runs':int(len(df)*2),'generated_from': [str(p.relative_to(ROOT)) for p in [ARCH_CSV, PRIMARY_CSV, LOCAL_CSV, HOTSPOT_CSV, COST_CSV]], 'exp0005':{},'exp0002':{},'signals':{},'readiness':{}}
for model, label in LABELS.items():
    summary['exp0005'][model] = {'label':label,'parameters':int(df[df.model_id==model].parameters.iloc[0]),'problems':{}}
    for prob in ['heat','burgers']:
        col=f'{prob}/field_relative_l2'
        if col not in df.columns: raise KeyError(col)
        s = df[df.model_id==model].sort_values('seed')
        vals = s[col].astype(float).tolist()
        rcol=f'{prob}/residual_rmse_abs'
        runtime_col=f'{prob}/signals/training/training_and_diagnostics_sec'
        steps_col=f'{prob}/signals/training/steps'
        extra = {'field_relative_l2':{'values':vals,'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1))},
                 'residual_rmse_abs':{'values':s[rcol].astype(float).tolist(),'mean':float(s[rcol].mean()),'sd':float(s[rcol].std(ddof=1))},
                 'runtime_sec':{'values':s[runtime_col].astype(float).tolist(),'mean':float(s[runtime_col].mean()),'sd':float(s[runtime_col].std(ddof=1))},
                 'steps':{'values':s[steps_col].astype(float).tolist(),'mean':float(s[steps_col].mean())}}
        if prob=='burgers':
            c=f'{prob}/front_field_rmse_normalized'
            v=s[c].astype(float).tolist()
            extra['front_field_rmse_normalized']={'values':v,'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1))}
        summary['exp0005'][model]['problems'][prob]=extra

# EXP-0002: independently summarize the saved validation JSONs (historical stationary Burgers contract).
for prob in ['heat','burgers']:
    metric_rows=[]
    train_rows=[]
    for seed in [101,102,103]:
        folder=EXP2/f'seed-{seed}'/prob
        val=json.loads((folder/'validation.json').read_text(encoding='utf-8'))
        tr=json.loads((folder/'training_summary.json').read_text(encoding='utf-8'))
        metric_rows.append(val['metrics']); train_rows.append(tr)
    keys=['field_relative_l2','residual_rmse_abs'] + (['front_field_rmse_normalized'] if prob=='burgers' else [])
    m={}
    for key in keys:
        vals=[float(r[key]) for r in [a for a in [x for x in metric_rows]] if key in r]
        if len(vals)!=3: raise ValueError((prob,key,vals))
        m[key]={'values':vals,'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1))}
    m['runtime_sec']={'values':[float(x['training_and_diagnostics_sec']) for x in train_rows], 'mean':float(np.mean([x['training_and_diagnostics_sec'] for x in train_rows])), 'sd':float(np.std([x['training_and_diagnostics_sec'] for x in train_rows],ddof=1))}
    m['steps']={'values':[int(x['steps']) for x in train_rows], 'mean':float(np.mean([x['steps'] for x in train_rows]))}
    summary['exp0002'][prob]=m

loc=pd.read_csv(LOCAL_CSV)
interior=loc[loc.scope=='interior'].copy()
hot=pd.read_csv(HOTSPOT_CSV)
summary['signals']['localization']={}
for model in LABELS:
    summary['signals']['localization'][model]={}
    for prob in ['heat','burgers']:
        sub=interior[(interior.model==model)&(interior.problem==prob)]
        summary['signals']['localization'][model][prob]={
            'spearman_values':sub.spearman.astype(float).tolist(),
            'spearman_mean':float(sub.spearman.mean()),
            'enrichment_values':sub.error_enrichment.astype(float).tolist(),
            'enrichment_mean':float(sub.error_enrichment.mean()),
        }
summary['signals']['hotspot_jaccard_burgers_rbf']=[float(x) for x in hot[(hot.model=='rbf_fixed')&(hot.problem=='burgers')]['residual_tail_jaccard']]
summary['signals']['hotspot_jaccard_burgers_rbf_range']=[min(summary['signals']['hotspot_jaccard_burgers_rbf']),max(summary['signals']['hotspot_jaccard_burgers_rbf'])]

# Figure 1: individual seed errors, preserving the small n rather than hiding it in bars.
fig, axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
for ax,prob,title in zip(axes,['heat','burgers'],['Heat: varilla 1D','Burgers transitorio canónico']):
    for i,model in enumerate(LABELS):
        vals=summary['exp0005'][model]['problems'][prob]['field_relative_l2']['values']
        jitter=np.linspace(-0.11,0.11,len(vals))
        ax.scatter(np.full(len(vals),i)+jitter, vals, color=COLORS[model], s=48, alpha=.85, zorder=3, label=LABELS[model] if ax is axes[0] else None)
        mean=np.mean(vals); sd=np.std(vals,ddof=1)
        ax.errorbar(i,mean,yerr=sd,fmt='D',color='#17202A',capsize=4,markersize=5,linewidth=1.2,zorder=4)
    ax.set_yscale('log'); ax.set_xticks(range(3),[LABELS[m] for m in LABELS],rotation=14,ha='right')
    ax.set_ylabel('Error relativo L2 (log)'); ax.set_title(title); ax.grid(True,which='both',axis='y',alpha=.22)
axes[0].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('EXP-0005 · screening de arquitecturas · tres semillas')
fig.savefig(FIG/'figure_01_architecture_errors.png',dpi=190,bbox_inches='tight'); plt.close(fig)

# Figure 2: quantitative relation between residual ranking and true field error.
fig, axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
for ax,metric,title in [(axes[0],'spearman','Correlación de Spearman residual–error'),(axes[1],'error_enrichment','Enriquecimiento del error en top 10% del residual')]:
    xx=[]
    colors=[]
    vals=[]
    for prob,offset in [('heat',-.17),('burgers',.17)]:
        for i,model in enumerate(LABELS):
            sub=interior[(interior.model==model)&(interior.problem==prob)].sort_values('seed')
            arr=sub[metric].astype(float).to_numpy()
            x=i+offset
            xx.extend(x+np.linspace(-.045,.045,len(arr))); vals.extend(arr); colors.extend([COLORS[model]]*len(arr))
            ax.errorbar(x,float(np.mean(arr)),yerr=float(np.std(arr,ddof=1)),fmt='o',color=COLORS[model],capsize=3,markersize=6,markeredgecolor='black',markeredgewidth=.4)
    for x,y,c in zip(xx,vals,colors): ax.scatter(x,y,color=c,s=19,alpha=.65)
    if metric=='error_enrichment': ax.axhline(1.0,color='#555',linestyle='--',linewidth=1,label='sin enriquecimiento (1×)')
    else: ax.axhline(0,color='#555',linestyle='--',linewidth=1)
    ax.set_xticks(range(3),[LABELS[m] for m in LABELS],rotation=14,ha='right')
    ax.set_title(title); ax.grid(True,axis='y',alpha=.22)
    if metric=='error_enrichment': ax.set_ylabel('Error medio top 10% / error medio total')
    else: ax.set_ylabel('ρ de Spearman')
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Señal post hoc de localización · EXP-0005 · interior · media ± SD y puntos por semilla')
fig.savefig(FIG/'figure_02_residual_localization.png',dpi=190,bbox_inches='tight'); plt.close(fig)

# Figure 3: per-problem time and field error under the actual stopping rules.
fig, ax=plt.subplots(figsize=(7.3,5.0),constrained_layout=True)
markers={'heat':'o','burgers':'s'}
for model,label in LABELS.items():
    for prob in ['heat','burgers']:
        sub=df[df.model_id==model].sort_values('seed')
        x=sub[f'{prob}/signals/training/training_and_diagnostics_sec'].astype(float).to_numpy()
        y=sub[f'{prob}/field_relative_l2'].astype(float).to_numpy()
        ax.scatter(x,y,c=COLORS[model],marker=markers[prob],s=65,edgecolors='black',linewidths=.4,label=f'{label} · {prob}' if prob=='heat' else None,alpha=.83)
        # Seed IDs remain available in the interactive HTML hover.
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Tiempo observado por problema (s, escala log)'); ax.set_ylabel('Error relativo L2 (log)')
ax.set_title('Costo–error con reglas de parada efectivamente observadas')
ax.grid(True,which='both',alpha=.22)
# Explicit legend by encodings and architecture.
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=COLORS[m],markeredgecolor='black',label=LABELS[m],markersize=8) for m in LABELS]
handles += [Line2D([0],[0],marker=markers[p],color='#444',linestyle='None',label=p.title(),markersize=7) for p in ['heat','burgers']]
ax.legend(handles=handles,frameon=False,ncol=2,fontsize=8)
fig.savefig(FIG/'figure_03_cost_error.png',dpi=190,bbox_inches='tight'); plt.close(fig)

# Figure 4: PyVista off-screen scientific field rendering from archived EXP-0002 NPZ.
val_path=EXP2/'seed-101'/'burgers'/'validation.npz'
z=np.load(val_path)
x,t=z['x'],z['t']; X,T=np.meshgrid(x,t,indexing='ij')
try:
    import pyvista as pv
    pv.OFF_SCREEN=True
    plotter=pv.Plotter(shape=(2,2),off_screen=True,window_size=(2000,1400),border=False)
    fields=[('Referencia exacta tanh',z['reference'],'viridis',None,'u'),('RBF-PIKAN predicción',z['prediction'],'viridis',None,'u'),('Error absoluto en log10',np.log10(np.maximum(z['absolute_error'],1e-12)),'magma',None,'log10|e|'),('Residuo PDE firmado',z['residual'],'coolwarm',(-np.max(np.abs(z['residual'])),np.max(np.abs(z['residual']))),'r')]
    for i,(title,arr,cmap,clim,bar_title) in enumerate(fields):
        plotter.subplot(i//2,i%2)
        zgeom=arr if i<2 else np.zeros_like(arr)
        surf=pv.StructuredGrid(X,T,zgeom)
        surf['value']=arr.ravel(order='F')
        kw={'scalars':'value','cmap':cmap,'show_scalar_bar':True,'scalar_bar_args':{'title':bar_title,'title_font_size':16,'label_font_size':12,'n_labels':4,'vertical':True,'position_x':0.88,'position_y':0.18,'height':0.62,'width':0.07},'smooth_shading':False}
        if clim: kw['clim']=clim
        plotter.add_mesh(surf,**kw)
        plotter.add_text(title,font_size=12,position='upper_left',color='black')
        plotter.show_grid(xtitle='x',ytitle='t',ztitle='u',font_size=14,grid='back',location='outer')
        plotter.view_isometric(); plotter.camera.zoom(1.15)
    plotter.screenshot(str(FIG/'figure_04_burgers_pyvista.png'),return_img=False)
    plotter.close()
    summary['visualization']={'pyvista':'created','source_npz':str(val_path.relative_to(ROOT)),'figure':'figures/figure_04_burgers_pyvista.png','note':'EXP-0002 stationary Burgers, seed 101; validation grid 129x65; log10(abs(error)) clipped at 1e-12 for rendering only.'}
except Exception as exc:
    summary['visualization']={'pyvista':'failed','error':repr(exc),'source_npz':str(val_path.relative_to(ROOT))}
    raise

# Interactive HTML companion: hoverable summary, signal, and actual cost/error.
fig=make_subplots(rows=2,cols=2,subplot_titles=('Error relativo L2 por arquitectura (media ± SD)','Enriquecimiento del error en el top 10% del residual','Costo–error por corrida','Burgers EXP-0002 seed 101: predicción y residual'),specs=[[{'type':'bar'},{'type':'bar'}],[{'type':'scatter'},{'type':'heatmap'}]])
for model,label in LABELS.items():
    for prob, col in [('heat','blue'),('burgers','red')]:
        d=summary['exp0005'][model]['problems'][prob]['field_relative_l2']
        fig.add_trace(go.Bar(name=f'{label} · {prob}',x=[f'{LABELS[m]}<br>{p}' for m in LABELS for p in ['Heat','Burgers'] if m==model for p in [prob.title()]],y=[d['mean']],error_y={'type':'data','array':[d['sd']]},marker_color=COLORS[model],legendgroup=model,showlegend=(prob=='heat'),hovertemplate='%{x}<br>media=%{y:.3g}<extra></extra>'),row=1,col=1)
# Enrichment means with individual seed hover.
for prob in ['heat','burgers']:
    for model,label in LABELS.items():
        d=summary['signals']['localization'][model][prob]
        fig.add_trace(go.Bar(name=f'{label} · {prob}',x=[label],y=[d['enrichment_mean']],marker_color=COLORS[model],legendgroup=model,showlegend=False,error_y={'type':'data','array':[float(np.std(d['enrichment_values'],ddof=1))]},customdata=[d['enrichment_values']],hovertemplate='semillas=%{customdata}<br>media=%{y:.2f}×<extra></extra>'),row=1,col=2)
# Cost-error cloud per run/problem.
for model,label in LABELS.items():
    for prob,symbol in [('heat','circle'),('burgers','square')]:
        sub=df[df.model_id==model]
        fig.add_trace(go.Scatter(x=sub[f'{prob}/signals/training/training_and_diagnostics_sec'],y=sub[f'{prob}/field_relative_l2'],mode='markers+text',text=sub.seed.astype(str),textposition='top center',name=f'{label} · {prob}',legendgroup=model,showlegend=False,marker={'color':COLORS[model],'symbol':symbol,'size':9},customdata=np.column_stack([sub.seed,sub[f'{prob}/signals/training/steps']]),hovertemplate='seed=%{customdata[0]}<br>steps=%{customdata[1]}<br>tiempo=%{x:.1f}s<br>L2=%{y:.3g}<extra></extra>'),row=2,col=1)
# Visual field heat map; this is the archived stationary Burgers contract, not EXP-0005 transient one.
fig.add_trace(go.Heatmap(x=x,y=t,z=z['prediction'].T,colorscale='Viridis',colorbar={'title':'u'},hovertemplate='x=%{x:.3f}<br>t=%{y:.3f}<br>u=%{z:.4f}<extra></extra>'),row=2,col=2)
fig.update_yaxes(type='log',title_text='Error relativo L2',row=1,col=1)
fig.update_yaxes(title_text='Enriquecimiento top10% (×)',row=1,col=2)
fig.update_xaxes(type='log',title_text='tiempo (s)',row=2,col=1); fig.update_yaxes(type='log',title_text='Error relativo L2',row=2,col=1)
fig.update_xaxes(title_text='x',row=2,col=2); fig.update_yaxes(title_text='t',row=2,col=2)
fig.update_layout(title='AF_PIKANs · brief de avance · resultados y señales',height=920,barmode='group',template='plotly_white',legend={'orientation':'h','y':1.08,'x':0},margin={'t':120})
fig.write_html(HTML/'brief_results_interactive.html',include_plotlyjs='cdn',full_html=True)

# Machine-readable small summary is the source for the Word tables.
(OUT/'summary_for_brief.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'summary':str(OUT/'summary_for_brief.json'),'figures':[str(p) for p in FIG.glob('*.png')],'html':str(HTML/'brief_results_interactive.html'),'runs_exp0005':len(df),'pyvista':summary['visualization']},ensure_ascii=False,indent=2))




