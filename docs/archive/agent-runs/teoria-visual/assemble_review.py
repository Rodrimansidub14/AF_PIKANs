"""Build an isolated, reviewable thesis variant without editing source chapters."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PREFIX=HERE.relative_to(REPO).as_posix()

def read(path): return path.read_text(encoding='utf-8-sig')

first=read(REPO/'capitulos/marco/01-aproximacion.tex')
anchor=r'\section{Métodos clásicos y adaptación de la resolución}'
assert first.count(anchor)==1
first=first.replace(anchor,read(HERE/'pdes-expansion.tex')+'\n\n'+anchor)
(HERE/'01-aproximacion-ampliada.tex').write_text(first,encoding='utf-8')

piezo=read(REPO/'capitulos/marco/04-piezo.tex')
anchor=r'\subsection{Antecedente local y lectura de sus figuras}'
assert piezo.count(anchor)==1
extra=read(HERE/'piezo-expansion.tex')
fiziko=r'''
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{bimorph_fiziko.pdf}
\caption[Sección del bimorfo dibujada con Fiziko.]{Sección esquemática del
bimorfo construida con la biblioteca MetaPost Fiziko. Las dos capas comparten
la interfaz adherida; el campo eléctrico se orienta hacia el electrodo de
menor potencial. La altura se amplifica para distinguir capas y electrodos.
Los trazos interiores identifican las capas y no representan líneas de campo
ni una discretización. Fuente gráfica: elaboración propia con Fiziko
(\url{https://github.com/jemmybutton/fiziko}).}
\label{fig:bimorfo-fiziko}
\end{figure}
'''
extra=extra.replace(r'\subsubsection{Constitutivo particular y ecuaciones fuertes}',fiziko+'\n'+r'\subsubsection{Constitutivo particular y ecuaciones fuertes}')
piezo=piezo.replace(anchor,extra+'\n\n'+anchor)
(HERE/'04-piezo-ampliada.tex').write_text(piezo,encoding='utf-8')

chapter=read(REPO/'i-marco_teorico.tex')
chapter=chapter.replace('capitulos/marco/01-aproximacion',f'{PREFIX}/01-aproximacion-ampliada')
chapter=chapter.replace('capitulos/marco/04-piezo',f'{PREFIX}/04-piezo-ampliada')
(HERE/'i-marco-ampliado.tex').write_text(chapter,encoding='utf-8')

main=read(REPO/'z-main.tex')
main=main.replace('{i-marco_teorico}',f'{{{PREFIX}/i-marco-ampliado}}')
main=main.replace(r'\graphicspath{ {figuras/} }',r'\graphicspath{ {'+PREFIX+r'/figures/} {'+PREFIX+r'/fiziko/} {figuras/} }')
(HERE/'tesis-revision.tex').write_text(main,encoding='utf-8')

paths=['i-marco_teorico.tex','capitulos/marco/01-aproximacion.tex','capitulos/marco/04-piezo.tex','z-main.tex','m-bibliografia.bib']
snapshot={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in paths}
(HERE/'source-snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print('Review variant assembled in', HERE)
