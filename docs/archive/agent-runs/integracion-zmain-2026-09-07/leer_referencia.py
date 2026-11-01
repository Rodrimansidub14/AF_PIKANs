from pypdf import PdfReader
from pathlib import Path
import json
p=Path('C:/Users/rodri/Downloads/Tesis_corregida_firmada.pdf')
r=PdfReader(p);out=Path('var/agent_runs/integracion-zmain-2026-09-07')
texts=[p.extract_text() or '' for p in r.pages]
(out/'tesis-referencia.txt').write_text('\n\n'.join(f'PAGINA_PDF {i+1}\n{v}' for i,v in enumerate(texts)),encoding='utf-8')
(out/'tesis-paginas.json').write_text(json.dumps(texts,ensure_ascii=False),encoding='utf-8')
print('Páginas:',len(r.pages));print('\n'.join(texts[:7]))
