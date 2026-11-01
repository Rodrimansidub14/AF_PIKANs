from pathlib import Path
p=Path('.')
f=p/'protocolo-revisado.tex'
s=f.read_text(encoding='utf-8-sig').replace('\\usepackage[T1]{fontenc}','\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n\\usepackage{csquotes}').replace('\\begin{document}','\\begin{document}\n\\hypersetup{pageanchor=false}').replace('\\end{titlepage}','\\end{titlepage}\n\\hypersetup{pageanchor=true}\n\\pagenumbering{roman}').replace('\\chapter{Introducción}', '\\clearpage\\pagenumbering{arabic}\n\\chapter{Introducción}')
f.write_text(s,encoding='utf-8')
f=p/'generar_figuras.py';s=f.read_text(encoding='utf-8').replace("xlim=(-1.5,10)","xlim=(-1.5,10.5)").replace("'Estado actual\\ncoeficientes + geometría'","'Estado actual\\nparámetros y geometría'").replace("'Revertir si falla\\nRegistrar cada evento'","'Registrar resultado\\nContinuar entrenamiento'").replace("'Aceptar si conserva\\ny cumple presupuesto'","'Aceptar o revertir\\nsegún restricciones'")
f.write_text(s,encoding='utf-8')
f=p/'i-marco_teorico.tex';s=f.read_text(encoding='utf-8').replace('Por tanto, no es defendible', 'AB-PINNs también adapta y enriquece subdominios durante el entrenamiento \\cite{abpinn2025}. Por tanto, no es defendible')
s=s.replace('\\section{Aprendizaje informado', '\\begin{figure}[htbp]\\centering\\includegraphics[width=.68\\linewidth]{figuras/fem_convergencia.png}\\caption{Convergencia comprobada de la demostración FEM en tres mallas. No valida el controlador futuro.}\\label{fig:convergencia}\\end{figure}\n\n\\section{Aprendizaje informado')
f.write_text(s,encoding='utf-8')
f=p/'m-bibliografia.bib';s=f.read_text(encoding='utf-8-sig').replace('note         = {Verificar volumen y páginas}','note         = {Identificación mediante DOI}').replace('note         = {Tesis doctoral; institución pendiente de confirmar}', 'url          = {https://d-nb.info/989762831/34}')
f.write_text(s,encoding='utf-8')
