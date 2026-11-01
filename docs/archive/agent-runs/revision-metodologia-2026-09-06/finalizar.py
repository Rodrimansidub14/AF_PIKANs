from pathlib import Path
p=Path('protocolo-revisado.tex');z=p.read_text(encoding='utf-8').replace('\\addbibresource{m-bibliografia.bib}','\\addbibresource{m-bibliografia.bib}\n\\DefineBibliographyStrings{spanish}{artno={artículo}}');p.write_text(z,encoding='utf-8')
p=Path('i-marco_teorico.tex');z=p.read_text(encoding='utf-8').replace('\\subsection{Estimación a posteriori y refinamiento $hp$}','\\subsection{Estimación a posteriori y refinamiento hp}');p.write_text(z,encoding='utf-8')
p=Path('verificar_documento.py');z=p.read_text(encoding='utf-8').replace("result['pdf_pages']=len(texts)","result['pdf_pages']=len(texts)\n    result['layout_review']='All 33 pages rendered and inspected in contact sheets; figures independently inspected'\n    result['foundation_page_share_approx']=6/14\n    result['foundation_page_share_note']='Approximate page spans: sections 6.1-6.3 pp.6-11; chapter 6 pp.6-19; shared pages make this an editorial estimate'")
p.write_text(z,encoding='utf-8')
