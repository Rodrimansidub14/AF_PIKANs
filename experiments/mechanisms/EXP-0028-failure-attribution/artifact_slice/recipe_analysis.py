"""EXP-0028 rebanada: derivacion de la RECETA MEDIA con datos (budget_slice.md). Lee los `hist` (perdida cada 100 pasos) de los B de receta completa (1d AC/KPP/Burgers, 1e NS) y corta
(fases 1/1b/1c), y los cocientes e_X/e_B al final de Adam y de L-BFGS. Por problema: M = primer paso en que la mediana-entre-semillas de la perdida de B (mediana movil de 5 puntos) queda
<= 2x la de B al final de la receta completa (y variante 'se queda dentro'). DFG: sin receta completa corrida; se usa la forma de la perdida de dfg_long (RBF16, 20 000 Adam).
Solo lectura (CPU). Uso: python recipe_analysis.py"""
import json,glob,math,sys
import numpy as np
R=str(__import__('pathlib').Path(__file__).resolve().parents[1]/'results')+'/'
def load(pat):
    out={}
    for f in glob.glob(R+pat):
        d=json.load(open(f))
        if 'hist' not in d: continue
        out[(d['arm'],d['seed'])]=d
    return out
SRC={ # problem: (short glob, full glob)
 'ac':(('phase1a/exp0028/ac-*-seed40*.json'),('phase1d-ac/exp0028_1d/ac-*.json')),
 'kpp':(('phase1b/exp0028/kpp-*-seed40*.json'),('phase1d-kpp/exp0028_1d/kpp-*.json')),
 'burgers':(('phase1b/exp0028/burgers-*-seed40*.json'),('phase1d-bu1/exp0028_1d/burgers-*.json')),
 'ns':(('phase1c-t1/exp0028_1c/ns-*-seed*.json'),('phase1e-ns1/exp0028_1e/ns-*-seed*.json')),
 'dfg':(('phase1c-dfg/exp0028_1c_dfg/dfg-*-seed*.json'),None),
}
def arr(runs,arm,key,seeds=None):
    return [d for (a,s),d in sorted(runs.items()) if a==arm and (seeds is None or s in seeds)]
def smooth(h,w=5):
    st=np.array([x['step'] for x in h]); l=np.array([x['loss'] for x in h])
    ls=np.array([np.median(l[max(0,i-w//2):i+w//2+1]) for i in range(len(l))])
    return st,ls
def med(x): return float(np.median(x))
res={}
for p,(sg,fg) in SRC.items():
    short=load(sg); full=load(fg) if fg else {}
    print('=====',p,'short n',len(short),'full n',len(full))
    for name,runs in (('short',short),('full',full)):
        if not runs: continue
        B=arr(runs,'B',None)
        if not B: continue
        seeds=sorted(d['seed'] for d in B)
        print(name,'B seeds',seeds, 'steps',B[0]['adam_steps'])
        # median over seeds of smoothed loss at step grid
        st,_=smooth(B[0]['hist'])
        L=np.array([smooth(d['hist'])[1] for d in B])
        Lm=np.exp(np.median(np.log(L),axis=0))
        la=med([d.get('loss_adam_end',np.nan) for d in B]); lf=med([d.get('loss_final',np.nan) for d in B])
        print('  loss adam end (med) %.3e  loss final %.3e'%(la,lf))
        grid=[1000,2000,3000,4000,5000,6000,8000,10000,12000,16000,20000,24000,30000,40000]
        print('  step:loss(smoothed med) ',' '.join('%d:%.2e'%(g,Lm[list(st).index(g)]) for g in grid if g in list(st)))
        ref=Lm[-1]
        for tag,r in (('Adam-end',ref),('post-LBFGS',lf)):
            ok=[int(s) for s,l in zip(st,Lm) if l<=2*r]
            print('  M(loss<=2x %s=%.2e): %s'%(tag,r,(ok[0] if ok else None)), ' (first multiple of 1000: %s)'%next((s for s in ok if s%1000==0),None))
        # arm ratios
        for arm in ('F','J','E','I'):
            A=arr(runs,arm,None,seeds)
            if len(A)<1: continue
            # paired
            pairs=[(runs[('B',d['seed'])],d) for d in A if ('B',d['seed']) in runs]
            rl=[]; 
            for b,a in pairs:
                _,lb=smooth(b['hist']); _,la_=smooth(a['hist']); rl.append(np.log(la_/lb))
            rl=np.exp(np.median(np.array(rl),axis=0))
            def rat(k): 
                v=[a[k]/b[k] for b,a in pairs if k in a and k in b]; return (med(v),len(v)) if v else (None,0)
            print('  %s n=%d: err ratio adam %s final %s | loss ratio adam-end %s final %s'%(arm,len(pairs),rat('primary_adam'),rat('primary_final'),rat('loss_adam_end'),rat('loss_final')))
            print('     loss ratio vs step: ',' '.join('%d:%.2f'%(g,rl[list(st).index(g)]) for g in grid if g in list(st)))

print('\n##### extra: stays-within, DFG long shape, KPP short')
def stays(p,runs,w=5,factor=2.0):
    B=arr(runs,'B',None); st,_=smooth(B[0]['hist'])
    L=np.array([smooth(d['hist'],w)[1] for d in B]); Lm=np.exp(np.median(np.log(L),axis=0)); ref=Lm[-1]
    ok=Lm<=factor*ref
    # last index where not ok
    bad=np.where(~ok)[0]; idx=(bad[-1]+1) if len(bad) else 0
    return int(st[idx]) if idx<len(st) else None, float(Lm[list(st).index(2000)]/ref) if 2000 in list(st) else None
for p,(sg,fg) in SRC.items():
    if not fg: continue
    full=load(fg)
    print(p,'stays within 2x (w=5, w=11):',stays(p,full,5),stays(p,full,11),' within 3x', stays(p,full,5,3.0))
# dfg_long shape
base=str(__import__('pathlib').Path(__file__).resolve().parents[4]/'var/agent_runs')+'/baseline-competitive-2026-09-27/ns2d-cg-20260930/outputs/dfgl/results/'
Ls=[]
for s in range(801,806):
    h=json.load(open(base+f'corrected-rbf-seed-{s}/history.json'))
    h=[x for x in h if x['stage']=='adam']; Ls.append(([x['step'] for x in h],[x['loss'] for x in h]))
st=np.array(Ls[0][0]); L=np.array([np.array(l) for _,l in Ls]); 
Ls_=np.array([[np.median(l[max(0,i-2):i+3]) for i in range(len(l))] for l in L]); Lm=np.exp(np.median(np.log(Ls_),axis=0))
print('dfg_long RBF16 adam steps',st[0],st[-1],len(st))
for g in (1000,2000,4000,6000,8000,10000,12000,16000,20000):
    if g in list(st): print(g,'%.3e'%Lm[list(st).index(g)], 'ratio to end %.2f'%(Lm[list(st).index(g)]/Lm[-1]))
ok=[int(s) for s,l in zip(st,Lm) if l<=2*Lm[-1]]; print('DFG long first within 2x',ok[0], 'stays', int(st[np.where(Lm>2*Lm[-1])[0][-1]+1]))
# kpp short
ks=load('phase1b/exp0028/kpp-*-seed40*.json'); print('kpp short n',len(ks))
for arm in ('F','J','E','I'):
    pairs=[(ks[('B',s)],ks[(arm,s)]) for (a,s) in ks if a==arm and ('B',s) in ks]
    f=lambda k:np.median([a[k]/b[k] for b,a in pairs]); print('kpp short',arm,len(pairs),'err adam %.3f final %.3f loss adam %.3f'%(f('primary_adam'),f('primary_final'),f('loss_adam_end')))
B=arr(ks,'B',None); print('kpp short B loss adam end',med([d['loss_adam_end'] for d in B]),'err',med([d['primary_adam'] for d in B]),med([d['primary_final'] for d in B]))
for name,runs in (('ac short',load(SRC['ac'][0])),('ac full',load(SRC['ac'][1])),('kpp full',load(SRC['kpp'][1])),('ns short',load(SRC['ns'][0])),('ns full',load(SRC['ns'][1])),('bu short',load(SRC['burgers'][0])),('bu full',load(SRC['burgers'][1])),('dfg short',load(SRC['dfg'][0]))):
    B=arr(runs,'B',None); print(name,'B err adam %.4g final %.4g'%(med([d['primary_adam'] for d in B]),med([d['primary_final'] for d in B])),'loss adam %.3g'%med([d['loss_adam_end'] for d in B]))
