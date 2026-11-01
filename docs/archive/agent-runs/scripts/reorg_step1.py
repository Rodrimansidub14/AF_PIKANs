import shutil, pathlib, re
R=pathlib.Path('.'); log=[]
def mv(src,dst):
    s,d=R/src,R/dst
    d.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(s),str(d)); log.append((src,dst))
EXP={'EXP-0001-heat1d-baseline':'heat','EXP-0007-heat-stepmatched-deepxde':'heat','EXP-0002-heat-burgers-sobol':'heat',
     'EXP-0006-burgers-sobol-rad':'burgers','EXP-0003-allen-cahn-rbf-sobol':'allen-cahn','EXP-0004-dfg-re20-rbf-sobol':'navier-stokes',
     'EXP-0008-baseline-v2-competitive':'baseline-v2'}
for e,p in EXP.items():
    mv(f'experiments/{e}',f'experiments/{p}/{e}')
    for sub in('notebooks','results','figures','checkpoints'):
        (R/f'experiments/{p}/{e}/{sub}').mkdir(exist_ok=True)
        (R/f'experiments/{p}/{e}/{sub}/.gitkeep').touch()
for p in('piezo',):
    (R/f'experiments/{p}').mkdir(exist_ok=True)
# notebooks
NB={}
for f in sorted((R/'notebooks/experiments').glob('*.ipynb')):
    n=f.name
    if n.startswith('allen-cahn'): g='allen-cahn/EXP-0003-allen-cahn-rbf-sobol'
    elif n.startswith('exp0008'): g='baseline-v2/EXP-0008-baseline-v2-competitive'
    elif n.startswith('heat-burgers-architecture-screen'): g='heat/EXP-0005-architecture-screen'
    elif n.startswith('heat-burgers-baseline'): g='heat/EXP-0002-heat-burgers-sobol'
    else: g='misc'
    mv(f'notebooks/experiments/{n}',f'notebooks/{g}/{n}')
if not any((R/'notebooks/experiments').iterdir()): (R/'notebooks/experiments').rmdir()
mv('notebooks/PIKAN-Warmup','notebooks/warmup')
(R/'MOVED.md').write_text('# MOVED — rutas viejas → nuevas\n\n## Paso 1 (2026-10-01): experiments/ y notebooks/\n\n| Antes | Ahora |\n|---|---|\n'+''.join(f'| `{a}` | `{b}` |\n' for a,b in log),encoding='utf-8')
print(len(log),'moves')
