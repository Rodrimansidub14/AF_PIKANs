import json, sys, difflib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.research_pipeline.remote.bundle import verify_bundle
from src.research_pipeline.remote.notebook import render_notebook
from src.research_pipeline.remote.runtime import canonical
for name in ['allen-cahn-exp003/bundle-final','heat-burgers-002-t4x2/bundle-final-20260923']:
    root=Path('var/agent_runs')/name
    spec=verify_bundle(root)
    content={n:(root/n).read_bytes() for n in spec['files']}
    content['bundle.json']=canonical(spec)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected=render_notebook(spec,content,manifest)
    actual=json.loads((root/'experiment.ipynb').read_text(encoding='utf-8'))
    print('\nBUNDLE',name,'bundle_id',spec['bundle_id'],'match',actual==expected)
    if actual!=expected:
        for a,b in zip(actual['cells'],expected['cells']):
            if a!=b:
                print('DIFF CELL',a.get('id'))
                actual_source=''.join(a.get('source',[])); expected_source=''.join(b.get('source',[]))
                print(''.join(list(difflib.unified_diff(actual_source.splitlines(True),expected_source.splitlines(True)))[:50]))

