import re,pathlib
EXP={'EXP-0001-heat1d-baseline':'heat','EXP-0007-heat-stepmatched-deepxde':'heat','EXP-0002-heat-burgers-sobol':'heat','EXP-0006-burgers-sobol-rad':'burgers','EXP-0003-allen-cahn-rbf-sobol':'allen-cahn','EXP-0004-dfg-re20-rbf-sobol':'navier-stokes','EXP-0008-baseline-v2-competitive':'baseline-v2'}
def nbgroup(n):
    if n.startswith('allen-cahn'): return 'allen-cahn/EXP-0003-allen-cahn-rbf-sobol'
    if n.startswith('exp0008'): return 'baseline-v2/EXP-0008-baseline-v2-competitive'
    if n.startswith('heat-burgers-architecture-screen'): return 'heat/EXP-0005-architecture-screen'
    if n.startswith('heat-burgers-baseline'): return 'heat/EXP-0002-heat-burgers-sobol'
files=[p for d in('experiments','tests','docs') for p in pathlib.Path(d).rglob('*') if p.is_file() and p.suffix in('.py','.md','.json','.toml','.yaml')]
ch=[]
for p in files:
    s=o=p.read_text(encoding='utf-8')
    for e,g in EXP.items(): s=s.replace(f'experiments/{e}',f'experiments/{g}/{e}')
    s=s.replace('experiments/EXP-0008-*','experiments/baseline-v2/EXP-0008-*')
    s=re.sub(r'notebooks/experiments/([\w\-]+\.ipynb)',lambda m:f'notebooks/{nbgroup(m.group(1))}/{m.group(1)}' if nbgroup(m.group(1)) else m.group(0),s)
    if p.name=='build_bundle.py': s=s.replace('parents[2]','parents[4]')
    if s!=o: p.write_text(s,encoding='utf-8'); ch.append(str(p))
print('\n'.join(ch))
