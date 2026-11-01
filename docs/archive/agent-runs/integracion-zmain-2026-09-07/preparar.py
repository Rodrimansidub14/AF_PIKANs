from pathlib import Path
from pypdf import PdfReader, PdfWriter
import shutil, json
ROOT=Path(__file__).resolve().parents[3]
RUN=Path(__file__).resolve().parent
OLD=ROOT/'var/agent_runs/revision-metodologia-2026-09-06'
FIG=ROOT/'figuras/revision2026'
FIG.mkdir(parents=True,exist_ok=True)
for name in ['arquitecturas','bases','fem_poisson','fem_convergencia','piezo_esquema']:
    for ext in ['png','pdf']:
        shutil.copy2(OLD/'figuras'/f'{name}.{ext}',FIG)
shutil.copy2(OLD/'generar_figuras.py', RUN/'generar_figuras_previo.py')
shutil.copy2(OLD/'validacion-figuras.json',RUN/'validacion-fem-previa.json')
r=PdfReader('C:/Users/rodri/Downloads/Tesis_corregida_firmada.pdf')
w=PdfWriter()
for n in [82,83,90,103,105,106]: w.add_page(r.pages[n-1])
with (RUN/'referencia-seleccion.pdf').open('wb') as f:w.write(f)
for n in [103,105]:
    w=PdfWriter();w.add_page(r.pages[n-1])
    with (FIG/f'tesis-pdf{n}.pdf').open('wb') as f:w.write(f)
print(ROOT)
