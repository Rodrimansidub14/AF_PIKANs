"""Execute the four analysis notebooks and their consolidated delivery."""
from pathlib import Path
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONUTF8']='1'
os.environ['PYTHONIOENCODING']='utf-8'
os.environ['JUPYTER_RUNTIME_DIR']=str(OUT/'jupyter-runtime')
os.environ['IPYTHONDIR']=str(OUT/'ipython-runtime')


def execute(name,cells):
    nb=nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update({'kernelspec':{'display_name':'AF_PIKANs SciML','language':'python','name':'python3'},
        'research':{'notebook_id':name+'-delivery-20260927','role':'analysis','source_of_truth':'src',
            'scientific_accepted':False,'local_tests_run':False,'new_training':False}})
    path=OUT/(name+'-results.ipynb')
    nbformat.write(nb,path)
    manager=KernelManager(kernel_name='python3')
    manager._kernel_spec=KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='AF_PIKANs SciML',language='python',env={'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1'})
    client=NotebookClient(nb,km=manager,timeout=900,allow_errors=False,resources={'metadata':{'path':str(OUT)}})
    start=time.perf_counter()
    print('Executing',path.name,flush=True)
    try:
        client.execute(cwd=str(OUT),cleanup_kc=True)
    finally:
        nbformat.write(nb,path)
    record={'notebook':path.name,'status':'executed','elapsed_seconds':time.perf_counter()-start,
        'code_cells_executed':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells)}
    print(json.dumps(record),flush=True)
    return record


def main():
    result=[]
    for name,fn in [('heat','heat'),('burgers','burgers'),('allen-cahn','allen_cahn'),('dfg','dfg')]:
        cells=[nbformat.v4.new_markdown_cell(f'# {name} — baseline exploratorio\n\nPosprocesado de corridas GPU terminadas. Sin nuevo entrenamiento ni tests locales. Las referencias y límites se documentan al final.'),
            nbformat.v4.new_code_cell('from IPython.display import display, Markdown\nimport baseline_delivery as delivery'),
            nbformat.v4.new_code_cell(f'result = delivery.{fn}()\ndisplay(result["summary"])'),
            nbformat.v4.new_code_cell('display(result["by_seed"])'),
            nbformat.v4.new_code_cell('result["figure"].show(renderer="plotly_mimetype")'),
            nbformat.v4.new_code_cell('display(Markdown(result["notes"]))')]
        if name=='allen-cahn':
            cells.append(nbformat.v4.new_code_cell('display(result["interfaces"])'))
        result.append(execute(name,cells))
    result.append(execute('baselines',[
        nbformat.v4.new_markdown_cell('# Cuatro baselines — entrega consolidada\n\nEsta tabla reúne los resultados de los cuatro notebooks ejecutados.'),
        nbformat.v4.new_code_cell('from IPython.display import display, Markdown\nimport baseline_delivery as delivery\ntable, report = delivery.consolidate()\ndisplay(table)'),
        nbformat.v4.new_code_cell('display(Markdown(report))')]))
    (OUT/'notebook-execution.json').write_text(json.dumps({'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.executable,'local_validation_skipped':True,'notebooks':result},indent=2),encoding='utf-8')
    archive=OUT.parent/'baselines-delivery-2026-09-27.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.suffix in {'.py','.ipynb','.csv','.json','.md','.html','.txt'}:
                z.write(path,arcname=path.name)
    print('DELIVERY',archive,flush=True)


if __name__=='__main__':
    main()
