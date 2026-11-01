"""Assemble the requested physical-illustration / solution convention in review copies."""
from pathlib import Path
import re,json,hashlib,shutil
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
OLD=HERE.parent/'teoria-visual'
P=HERE.relative_to(REPO).as_posix()
def read(p):return p.read_text(encoding='utf-8-sig')
def write(name,text): (HERE/name).write_text(text,encoding='utf-8')
def caption_text(path):
    text=read(path).strip()
    if text.startswith(r'\caption{') and text.endswith('}'):
        text=text[len(r'\caption{'):-1]
    return text.replace('Solución FEM real','Solución FEM calculada')
def fig(path,short,caption,label):
    size=r'width=\linewidth,height=.30\textheight,keepaspectratio' if '-problem.pdf' in path else r'width=\linewidth'
    return '\n'+r'\begin{figure}[htbp]\centering'+'\n'+r'\includegraphics['+size+']{'+path+'}\n'+r'\caption['+short+']{'+caption+'}'+r'\label{'+label+'}\n'+r'\end{figure}'+'\n'
def replace_fig(text,label,new):
    pattern=r'\\begin\{figure\}.*?\\end\{figure\}'
    found=0
    def sub(match):
        nonlocal found
        if r'\label{'+label+'}' in match[0]: found+=1;return new
        return match[0]
    result=re.sub(pattern,sub,text,flags=re.S)
    assert found==1,(label,found)
    return result
def add_after_heading(text,heading,addition):
    assert text.count(heading)==1,heading
    return text.replace(heading,heading+'\n\n'+addition)

ev=json.loads(read(HERE/'solutions/evidence.json'))
phys={
1:r'Ilustración de P1 con Fiziko: placa cuadrada, fuente distribuida y valor impuesto en los bordes. Puede interpretarse como una diferencia de temperatura respecto de la frontera. Para $k=4$, la fuente manufacturada alterna generación y extracción; el dibujo identifica los datos y no representa un campo calculado.',
2:r'Ilustración de P2 con Fiziko: medio unidimensional en el que el propio campo controla el transporte y la viscosidad suaviza el frente. Las flechas explican el mecanismo de Burgers y las fronteras corresponden a trazas de la onda exacta. Es un modelo escalar, no una reconstrucción de un conducto tridimensional.',
3:r'Ilustración de P3 con Fiziko: regiones de fase y una zona de transición, con difusión y reacción. Los símbolos $-1$ y $+1$ identifican estados preferidos. La disposición de fases es conceptual; la evolución del dato inicial se muestra por separado en la gráfica de solución.',
4:r'Ilustración de P4 con Fiziko: canal y cilindro fijo, entrada parabólica, paredes sin deslizamiento y salida natural. Las flechas de entrada son datos prescritos. La geometría y las fronteras se distinguen de las líneas de corriente calculadas en la figura de solución.',
5:r'Ilustración de P5 con Fiziko: dos capas de PVDF adheridas, empotramiento izquierdo, electrodos a 100 y 0 V, e interfaz material. El campo apunta hacia menor potencial. La geometría se exagera para distinguir capas y condiciones; la respuesta mecánica se obtiene mediante el FEM acoplado.'}
shorts={1:'P1: ilustración del problema de Poisson',2:'P2: ilustración del transporte y la difusión',3:'P3: ilustración de fases e interfaz',4:'P4: ilustración del canal con cilindro',5:'P5: ilustración del bimorfo piezoeléctrico'}
solcaps={
1:r'Solución exacta de Poisson para $k=1$ y $k=4$, cortes en $y=1/(2k)$ y comprobación independiente por diferencias finitas para $k=4$. Se resuelve el sistema discreto con 33, 65 y 129 nodos por eje. El panel de error compara la aproximación con la fórmula exacta en los nodos interiores; la pendiente de referencia es de segundo orden.',
2:r'Solución exacta de Burgers con $A=0.5$, $c=0.5$, $x_0=-0.25$ y $\nu=0.02$: mapa espacio--tiempo y perfiles a $t=0,0.5,1$. La línea discontinua del mapa sigue $x_f(t)=ct+x_0$. Las condiciones iniciales y de frontera son las trazas de la misma solución.',
3:r'Solución numérica del problema Allen--Cahn con $d=0.001$, $u(x,0)=x^2\cos(\pi x)$ y $u(\pm1,t)=-1$. Se emplean diferencias centradas y BDF, con 1601 nodos en la solución mostrada. Los paneles incluyen campo, cortes, energía discreta y diferencias entre refinamientos, evaluadas en nodos y tiempos comunes.'}
methods={
1:r'''\paragraph{Solución y comprobación.}
Los mapas se obtienen de la fórmula exacta. Como comprobación independiente,
se discretiza $-\Delta$ con el stencil centrado de cinco puntos y se resuelve
el sistema disperso con Dirichlet homogéneo. Al pasar de 33 a 65 y 129 nodos
por eje, el error relativo discreto se reduce aproximadamente por cuatro en
cada refinamiento, coherente con segundo orden. El residual algebraico se
comprueba por separado: resolver bien el sistema discreto no elimina su
error de discretización. Esta comparación aplica el método de diferencias
finitas desarrollado más adelante sin usar datos de entrenamiento.''',
2:r'''\paragraph{Solución y comprobación.}
La gráfica representa la onda exacta y sus trazas, no un ajuste a muestras.
La sustitución de sus derivadas verifica la identidad de Burgers para los
cuatro cruces de $c$ y $\nu$ del contrato, con residual del orden del
redondeo de doble precisión. Por ello esta fórmula sirve como referencia
directa para medir error de campo y error del frente en futuras aproximaciones.''',
3:r'''\paragraph{Solución numérica y sensibilidad a la resolución.}
Se discretiza el término difusivo con diferencias centradas de segundo
orden y se integra el sistema de ecuaciones ordinarias mediante BDF, usando
su Jacobiano disperso analítico. Las fronteras se fijan exactamente en $-1$.
Se comparan 401, 801 y 1601 nodos espaciales sobre 301 tiempos de salida
comunes. La diferencia máxima entre los dos últimos refinamientos es
$4.36\times10^{-4}$, y el cociente entre diferencias sucesivas es 3.94.
Al reducir las tolerancias relativas de $10^{-9}$ a $10^{-11}$ y absolutas
de $10^{-11}$ a $10^{-13}$ en la malla fina, la diferencia máxima es
$6.14\times10^{-9}$: en esta comparación domina la sensibilidad espacial.
Estas son comprobaciones de autoconvergencia y de tolerancia, no una cota
demostrada del error absoluto.

La energía discreta usa diferencias entre nodos para aproximar
$\int d|u_x|^2/2+W(u)\,dx$ y decrece en los tiempos de salida. Para una
solución suave con fronteras fijas, multiplicar
$u_t=d u_{xx}-W'(u)$ por $u_t$ e integrar da
\[
\frac{\mathrm d\mathcal E}{\mathrm dt}=-\int_{-1}^1|u_t|^2\,dx\leq0.
\]
Así se relaciona el panel de energía con el mecanismo físico. Allen--Cahn
describe un parámetro de orden no conservado: la reacción puede cambiar su
integral espacial, aunque la energía disminuya. La forma de los frentes
resulta del problema inicial completo, a diferencia del perfil ideal de una
sola interfaz en la recta.'''}

pdes=read(OLD/'pdes-expansion.tex')
start=pdes.index('Las figuras de esta sección son esquemas matemáticos')
end=pdes.index(r'\subsection{P1.',start)
pdes=pdes[:start]+r'''Cada problema se acompaña de dos figuras con funciones distintas. La
ilustración en tinta, realizada con Fiziko, describe el sistema y sus datos.
La gráfica en color representa una solución exacta o numérica e identifica
su método de obtención. Los cálculos siguientes verifican referencias para
la explicación del problema; no evalúan todavía el mecanismo adaptativo
PIKAN propuesto.

'''+pdes[end:]
titles={1:r'\subsection{P1. Poisson: equilibrio y frecuencia distribuida}',2:r'\subsection{P2. Burgers: transporte no lineal, difusión y frente}',3:r'\subsection{P3. Allen--Cahn: reacción, energía e interfaz}',4:r'\subsection{P4. DFG 2D-1: balance de cantidad de movimiento}'}
labels={1:'fig:expansion-p1-poisson',2:'fig:expansion-p2-burgers',3:'fig:expansion-p3-allen-cahn',4:'fig:expansion-p4-dfg'}
for k in (1,2,3,4):
    pdes=add_after_heading(pdes,titles[k],fig(f'p{k}-problem.pdf',shorts[k],phys[k],f'fig:p{k}-problem'))
    if k<4:
        content=methods[k]+fig(f'p{k}-solution.pdf',f'P{k}: solución y comprobación',solcaps[k],labels[k])
    else:
        dfg_method=read(HERE/'dfg/method.tex')
        if 'La discretización empleó' in dfg_method:
            dfg_method=r'\paragraph{Solución FEM y comprobación.}'+'\n'+dfg_method[dfg_method.index('La discretización empleó'):]
        content=dfg_method+fig('p4-solution.pdf','P4: solución FEM de velocidad y presión',caption_text(HERE/'dfg/caption.tex'),labels[k])
    pdes=replace_fig(pdes,labels[k],content)
pdes=pdes.replace('Las cuatro figuras acompañan esas preguntas con objetos distintos: funciones\nmanufacturadas en P1, derivadas exactas en P2, energía y perfil estacionario\nen P3, y geometría más balance en P4.',
'''Las parejas de ilustración y solución acompañan esas preguntas desde dos
perspectivas: los datos del sistema y el campo que los satisface. En P1 y P2
se dispone de soluciones exactas; en P3 y P4 se usan aproximaciones numéricas
con comprobaciones de resolución y balance.''')
pdes=pdes.replace(titles[4],r'\subsection{P4. Navier--Stokes: flujo DFG alrededor de un cilindro}')
pdes=pdes.replace('El benchmark publica valores de arrastre, sustentación y caída de presión que',
r'''En esta integral, $\mathbf n$ apunta hacia fuera del cilindro y hacia el
fluido; es el opuesto de la normal exterior del dominio fluido sobre esa
frontera interna. Declarar esta orientación evita invertir los signos de
las fuerzas.

La solución permite leer el balance espacialmente. El obstáculo desvía el
flujo hacia los pasos laterales, donde aumenta la rapidez, mientras que el
no deslizamiento obliga a anularla en la superficie. La presión participa
en esa desviación y en el frenado frontal; la viscosidad transmite el efecto
de las paredes al fluido cercano. La presión no se calcula mediante una
ley de estado de compresibilidad: en este modelo ajusta el campo de
velocidad para satisfacer simultáneamente el balance y la restricción
$\nabla\!\cdot\mathbf u=0$. Las líneas de corriente son tangentes a la
velocidad calculada; al ser estacionario este caso, también describen
trayectorias de partículas ideales del campo resuelto.

El benchmark publica valores de arrastre, sustentación y caída de presión que''')
write('pdes-expansion.tex',pdes)

piezo=read(OLD/'piezo-expansion.tex')
piezo=replace_fig(piezo,'fig:p5-bimorfo',fig('p5-problem.pdf',shorts[5],phys[5],'fig:p5-bimorfo'))
piezo=piezo.replace('La flecha de la figura siguiente se exagera para hacer visible la tendencia;\nno es una magnitud calculada.',
'La ilustración siguiente identifica materiales y condiciones; la gráfica de solución se obtiene por separado mediante el FEM acoplado.')
piezo=piezo.replace('que comunica la primera figura. La flecha, su magnitud y la distribución de\ncurvatura deben obtenerse del campo acoplado; no se presentan como resultados.',
'que predice esta lectura constitutiva. La flecha y la distribución de curvatura se obtienen del campo acoplado; la solución numérica se muestra al final de esta subsección.')
piezo+= '\n'+read(HERE/'pvdf/method.tex')+fig('p5-solution.pdf','P5: solución FEM piezoeléctrica acoplada',read(HERE/'pvdf/caption.tex'),'fig:p5-solution')
write('piezo-expansion.tex',piezo)

first=read(REPO/'capitulos/marco/01-aproximacion.tex')
first=first.replace(r'\section{Métodos clásicos y adaptación de la resolución}',pdes+'\n'+r'\section{Métodos clásicos y adaptación de la resolución}')
if (HERE/'illustrations/heat-rod.pdf').exists():
    heat=r'''\paragraph{Lectura física del operador difusivo.}
En una varilla homogénea, la temperatura $T(x,t)$ puede modelarse mediante
$\rho c\,T_t=-\partial_xq+s$, con flujo de Fourier
$q=-k\,T_x$. Para $k$ constante, $T_t=\alpha T_{xx}+s/(\rho c)$,
donde $\alpha=k/(\rho c)$. El gradiente fija la dirección del flujo y su
divergencia determina el calentamiento local. Sin fuente y con extremos
aislados, el calor total se conserva y las diferencias de temperatura se
suavizan. Esta ilustración introduce difusión; no es el problema Burgers ni
un caso adicional del banco experimental.
'''+fig('heat-rod.pdf','Difusión térmica en una varilla','Ilustración introductoria de la difusión térmica con Fiziko. El flujo se orienta de mayor a menor temperatura. Los extremos y fuentes dibujados son datos de este esquema, no de P2.','fig:heat-rod')
    first=first.replace(r'\subsection{Error, residual y estabilidad}',heat+'\n'+r'\subsection{Error, residual y estabilidad}')
write('01-aproximacion-ampliada.tex',first)
fourth=read(REPO/'capitulos/marco/04-piezo.tex')
fourth=fourth.replace(r'\subsection{Antecedente local y lectura de sus figuras}',piezo+'\n'+r'\subsection{Antecedente local y lectura de sus figuras}')
write('04-piezo-ampliada.tex',fourth)
second=read(REPO/'capitulos/marco/02-pinn.tex')
second=second.replace('revision2026/arquitecturas.pdf','architecture.pdf')
second=second.replace('Redibujo conceptual de la comparación MLP--KAN','Diagrama propio con Fiziko/MetaPost de la comparación MLP--KAN')
if (HERE/'illustrations/pinn-loss.pdf').exists():
    second=second.replace('revision2026/perdida.pdf','pinn-loss.pdf')
    second=second.replace('Redibujo conceptual de la construcción PINN','Diagrama propio con Fiziko/MetaPost de la construcción PINN')
    second=second.replace('la salida neuronal se diferencia y se combina con fuentes y condiciones para formar la pérdida. Se ilustra la variante sin datos interiores; la disposición es propia.',
        r'los términos de residual, frontera y datos se agregan con pesos explícitos. El término de datos es opcional; para la variante directa sin datos interiores se fija $\lambda_d=0$. La disposición es propia y representa la agregación de la pérdida, no un entrenamiento ejecutado.')
write('02-pinn-ilustrada.tex',second)

chapter=read(REPO/'i-marco_teorico.tex')
for orig,new in [('01-aproximacion','01-aproximacion-ampliada'),('02-pinn','02-pinn-ilustrada'),('04-piezo','04-piezo-ampliada')]:
    chapter=chapter.replace('capitulos/marco/'+orig,P+'/'+new)
write('i-marco-ampliado.tex',chapter)
main=read(REPO/'z-main.tex').replace('{i-marco_teorico}','{'+P+'/i-marco-ampliado}')
graphicdirs=[P+'/illustrations/',P+'/solutions/',P+'/dfg/',P+'/pvdf/','var/agent_runs/teoria-visual/figures/','figuras/']
main=main.replace(r'\graphicspath{ {figuras/} }',r'\graphicspath{'+''.join('{'+d+'}' for d in graphicdirs)+'}')
write('tesis-revision.tex',main)
for file in ('extract_review.py','contact_sheets.py'):
    shutil.copyfile(OLD/file,HERE/file)
snapshot={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in ['i-marco_teorico.tex','capitulos/marco/01-aproximacion.tex','capitulos/marco/02-pinn.tex','capitulos/marco/04-piezo.tex','m-bibliografia.bib','z-main.tex']}
write('source-snapshot.json',json.dumps(snapshot,indent=2))

# Compact atlas: one pair per problem, plus the architecture and diffusion illustration.
atlas=r'''\documentclass[11pt,letterpaper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[spanish]{babel}
\usepackage{lmodern,amsmath,amssymb,graphicx,booktabs,xcolor}
\usepackage[margin=1.5cm]{geometry}\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\graphicspath{'''+''.join('{'+d+'}' for d in graphicdirs)+r'''}
\begin{document}
'''
names={1:'Poisson: fuente y equilibrio',2:'Burgers: transporte y difusión',3:'Allen--Cahn: fases e interfaces',4:'Navier--Stokes: flujo alrededor de un cilindro',5:'PVDF: acoplamiento y flexión'}
for k in range(1,6):
    cap=solcaps[k] if k<4 else caption_text(HERE/('dfg' if k==4 else 'pvdf')/'caption.tex')
    atlas+=r'\section*{'+names[k]+'}\n'+r'\textbf{Ilustración del problema · Fiziko}\par'+'\n'
    atlas+=r'\includegraphics[width=\linewidth,height=.24\textheight,keepaspectratio]{p'+str(k)+'-problem.pdf}\n'
    atlas+=r'\textbf{Gráfica de solución}\par'+'\n'
    atlas+=r'\includegraphics[width=\linewidth,height=.48\textheight,keepaspectratio]{p'+str(k)+'-solution.pdf}\n'
    atlas+=r'{\small '+cap+'}\n'+r'\clearpage'+'\n'
atlas+=r'\section*{Arquitecturas: operaciones y conexiones}'+'\n'+r'\includegraphics[width=\linewidth,height=.55\textheight,keepaspectratio]{architecture.pdf}'+'\n'
atlas+=r'\par Diagrama con Fiziko/MetaPost. En MLP se aprenden pesos y sesgos; en KAN se aprenden funciones en aristas y se suman sus contribuciones en nodos.'+'\n'
if (HERE/'illustrations/pinn-loss.pdf').exists():
    atlas+=r'\par\includegraphics[width=\linewidth,height=.30\textheight,keepaspectratio]{pinn-loss.pdf}'+'\n'
if (HERE/'illustrations/heat-rod.pdf').exists():
    atlas+=r'\clearpage\section*{Difusión: transferencia de calor en una varilla}'+'\n'+r'\includegraphics[width=\linewidth,height=.6\textheight,keepaspectratio]{heat-rod.pdf}'+'\n'
    atlas+=r'\par El flujo térmico $q=-kT_x$ responde al gradiente de temperatura. El balance $\rho c T_t=-q_x+s$ produce la ecuación de difusión. Es una ilustración introductoria, separada del banco P1--P5.'+'\n'
atlas+=r'\end{document}'
atlas=re.sub(r'(\\includegraphics[^\n]+)',lambda m:r'{\centering '+m[0]+r'\par}',atlas)
write('atlas-figuras.tex',atlas)
print('Assembled chapter and compact figure atlas.')
