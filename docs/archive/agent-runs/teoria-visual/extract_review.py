from pathlib import Path
import json
from pypdf import PdfReader, PdfWriter

root=Path(__file__).resolve().parent
r=PdfReader(root/'tesis-revision.pdf')
chapters=[(o.title,r.get_destination_page_number(o)) for o in r.outline if not isinstance(o,list)]
start=next(p for t,p in chapters if t.startswith('Marco'))
stop=next(p for t,p in chapters if t.startswith('Metodolog'))
bib=next(p for t,p in chapters if t.startswith('Bibliograf'))
end=next(p for t,p in chapters if t.startswith('Anexos'))
w=PdfWriter()
w.append(r,pages=(start,stop),import_outline=False)
w.append(r,pages=(bib,end),import_outline=False)
w.add_metadata({'/Title':'Marco teórico ilustrado — borrador para revisión','/Subject':'Ampliación de fenómenos, ecuaciones y figuras; incluye bibliografía de la tesis.'})
w.add_outline_item('Marco teórico ampliado',0)
w.add_outline_item('Bibliografía',stop-start)
w.write(root/'marco-teorico-ilustrado.pdf')
new=[]
for i,page in enumerate(w.pages):
    txt=page.extract_text() or ''
    if any(s in txt for s in ['Lectura física','Poisson: equilibrio','Burgers: transporte','Allen–Cahn:','DFG 2D-1:','Caso concreto:','Constitutivo particular','Fronteras e interfaz','Lectura de signos','Sección del bimorfo','Geometría y mecanismo','Continuidad de potencial']):
        new.append(i+1)
(root/'pdf-review.json').write_text(json.dumps({'thesis_pages':len(r.pages),'theory_pages':stop-start,'bibliography_pages':end-bib,'review_pages':len(w.pages),'selected_review_pages':new,'source_ranges_zero_based':[[start,stop],[bib,end]]},indent=2),encoding='utf-8')
print('Extracted', len(w.pages),'pages; theory:',stop-start,'selected:',new)
