import shutil,pathlib,hashlib,json
V=pathlib.Path('var/agent_runs/baseline-competitive-2026-09-27'); N=V/'ns2d-cg-20260930'; H=V/'heat-lbfgs-redo-20260930'
E='EXP-0008-baseline-v2-competitive'
C=[('allen-cahn','ac-rerun-20260930',N/'outputs/ac/ac/results',[N/'ac-rerun-results.md',N/'ac-rerun-per-seed.csv']),
 ('navier-stokes','ns2d-cg-20260930',N/'outputs/ns/ns/results',[N/'ns2dcg-results.md',N/'ns2dcg-per-seed.csv']),
 ('navier-stokes','dfg-daniel-cfg-20260930',N/'outputs/dfgd/results',[N/'dfg-daniel-cfg-results.md',N/'dfg-daniel-cfg-per-seed.csv']),
 ('navier-stokes','dfg-long-20261001',N/'outputs/dfgl/results',[N/'dfg-long-results.md',N/'dfg-long-per-seed.csv']),
 ('heat','heat-lbfgs-redo-20260930',H/'gpu-output',[H/'results.md',H/'results-summary.json'])]
for prob,camp,root,summ in C:
    base=pathlib.Path('experiments')/prob/E
    for sub in('notebooks','results','figures','checkpoints'): (base/sub).mkdir(parents=True,exist_ok=True)
    idx={}
    for f in root.rglob('*'):
        if not f.is_file() or 'payload' in f.parts or '__pycache__' in f.parts or f.suffix in('.mat','.dat','.npz'): continue
        rel=f.relative_to(root)
        if f.suffix=='.pt':
            d=base/'checkpoints'/camp/rel; idx[rel.as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
        else: d=base/'results'/camp/rel
        d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,d)
    (base/'results'/camp).mkdir(parents=True,exist_ok=True)
    for s in summ: shutil.copy2(s,base/'results'/camp/s.name)
    (base/'checkpoints'/camp).mkdir(parents=True,exist_ok=True)
    (base/'checkpoints'/camp/'index.json').write_text(json.dumps({'campaign':camp,'sha256':idx},indent=1))
    print(camp,len(idx),'checkpoints')
