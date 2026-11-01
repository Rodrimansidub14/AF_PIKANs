from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'var/agent_runs/revision-metodologia-2026-09-06'
def put(name,text): (ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
put('2-paquetes_y_comandos_usuario.tex',r'''
\usepackage{lmodern}
\usepackage{tabularx}
\usepackage{longtable}
\DefineBibliographyStrings{spanish}{artno={artículo}}
\setlength{\LTpre}{1ex}
\setlength{\LTpost}{1ex}
''')
put('i-marco_teorico.tex',r'''
\label{sec:marco-teorico}
Este capítulo reúne los antecedentes y el fundamento matemático del proyecto. El recorrido parte de las ecuaciones diferenciales y su aproximación, desarrolla las representaciones neuronales y sus estrategias adaptativas, y termina con el acoplamiento piezoeléctrico y la síntesis de literatura. La definición del experimento y del mecanismo se presenta por separado en el Capítulo~\ref{sec:metodologia}.
\input{capitulos/marco/01-aproximacion}
\input{capitulos/marco/02-pinn}
\input{capitulos/marco/03-kan}
\input{capitulos/marco/04-piezo}
\input{capitulos/marco/05-sintesis}
''')
put('j-capitulos.tex',r'''
\chapter{Metodología}
\label{sec:metodologia}
\input{capitulos/metodologia/01-problemas}
\input{capitulos/metodologia/02-mecanismo}
\input{capitulos/metodologia/03-evaluacion}
''')
put('a-prefacio.tex',r'''
Este documento desarrolla el planteamiento de una investigación sobre focalización adaptativa en redes informadas por física. Su versión actual corresponde al cierre de fundamentos y metodología previo a la implementación del mecanismo. Las conclusiones se refieren al planteamiento; las afirmaciones sobre desempeño requerirán la campaña experimental definida en el texto.

La lectura puede seguir dos recorridos complementarios. El marco teórico explica los conceptos, ecuaciones y antecedentes que sostienen la pregunta. La metodología establece qué se estudiará, qué podrá modificar el mecanismo y qué evidencia permitirá evaluar su utilidad. Las figuras identifican si proceden de una fuente, de una recreación numérica o de la propuesta del proyecto.

La supervisión del trabajo corresponde a Gabriel Barrientos y la asesoría a Daniel Gonzales, conforme a los datos proporcionados para esta versión. Los espacios del tribunal no designado se mantienen como campos de firma sin atribuir nombres ni aprobaciones.
''')
put('b-resumen.tex',r'''
Las redes de Kolmogorov--Arnold informadas por física (PIKAN) representan soluciones de ecuaciones diferenciales mediante funciones univariadas aprendibles. La adaptación de sus grillas ya tiene antecedentes; la pregunta de esta investigación es cuándo una reasignación de capacidad guiada por representación interna aporta valor adicional frente a residual, curvatura o muestreo adaptativo.

Se propone estudiar una regla que mueva e intercambie intervalos de B-splines cúbicas con presupuesto fijo, transfiera coeficientes protegiendo valores y derivadas y acepte modificaciones según un sondeo emparejado con una continuación sin cambio. El banco se limita a cinco familias: Poisson, Burgers, Allen--Cahn, Navier--Stokes DFG 2D-1 y piezoelectricidad en un bimorfo de PVDF. Estas cubren frecuencia distribuida, frentes, reacción--difusión, geometría e interacción entre campos.

El protocolo define operadores, fronteras, escalas, referencias, controles, ablaciones, costos y criterios de falsación. El antecedente piezoeléctrico local se integra con una auditoría explícita de convenciones. Se incluyen recreaciones matemáticas, una demostración FEM de Poisson y figuras atribuidas a sus fuentes; no se presentan como resultados del mecanismo. La contribución potencial será evaluada por reducción reproducible de error localizado a costo comparable, sin degradación de otras regiones o campos. Su importancia permanece como hipótesis hasta completar implementación y evaluación.

\textbf{Palabras clave:} PIKAN, B-splines, adaptación de representación, ecuaciones diferenciales, piezoelectricidad, verificación numérica.
''')
put('c-abstract.tex',r'''
Physics-informed Kolmogorov--Arnold networks (PIKANs) approximate differential-equation solutions through trainable univariate functions. Adaptive grids already have substantial precedents. This research therefore asks when reallocating representational capacity using internal information adds value beyond residual, curvature, or adaptive sampling.

The proposed study considers moving and exchanging cubic B-spline intervals under a fixed parameter budget, transferring coefficients while protecting values and derivatives, and accepting changes through paired short training probes against an unchanged continuation. Five problem families cover the experimental scope: Poisson, viscous Burgers, Allen--Cahn, the DFG 2D-1 Navier--Stokes benchmark, and piezoelectricity in a PVDF bimorph. Together they address distributed frequency, moving fronts, reaction--diffusion, geometry, and coupled fields.

The protocol specifies operators, boundary conditions, scales, references, controls, ablations, costs, and falsification criteria. The local piezoelectric precedent is incorporated with an explicit audit of constitutive conventions. Mathematical recreations, a Poisson finite-element demonstration, and attributed source figures are included; they are not results of the proposed mechanism. Its potential contribution will be assessed through reproducible localized-error reduction at comparable cost without degrading other regions or fields. Its significance remains a hypothesis until implementation and evaluation are complete.

\textbf{Keywords:} PIKAN, B-splines, representation adaptation, differential equations, piezoelectricity, numerical verification.
''')
put('d-introduccion.tex',r'''
\label{sec:introduccion}
Resolver una ecuación diferencial implica representar una respuesta que puede variar de forma muy desigual. Una región extensa puede ser suave, mientras una transición estrecha concentra gradientes; en un dispositivo acoplado, distintos campos pueden exigir resoluciones diferentes. Los métodos numéricos clásicos atienden esa distribución mediante mallas y espacios de aproximación, y las redes informadas por física incorporan alternativas basadas en datos de colocación, pérdidas y representación \cite{fenics2016,raissi2019}.

Las PIKAN ofrecen funciones univariadas parametrizadas, frecuentemente mediante B-splines, cuya geometría puede modificarse. Esa posibilidad ya está presente en KAN y en trabajos de adaptación posteriores \cite{liu2024kan,rigas2024,rigas2026}. Por ello, el problema de investigación no consiste en añadir adaptatividad de manera genérica. Consiste en determinar si una decisión explícita de retirar y reasignar capacidad, informada por su efecto y limitada por costo y preservación, produce un beneficio adicional verificable.

La dificultad central es que el soporte interno de una base no equivale necesariamente a una región física compacta. Además, una modificación que conserva valores puede alterar derivadas y con ellas el residual de la PDE. En un sistema piezoeléctrico, mejorar desplazamiento sin conservar potencial o flujo puede dar una interpretación incompleta del resultado \cite{lahmer2008,gonzalez2024tesis}.

Este trabajo plantea una investigación sobre el mecanismo, con piezoelectricidad como prueba de acoplamiento vinculada al antecedente local. El marco teórico desarrolla las herramientas necesarias para comprender esa relación; la metodología delimita cinco familias de problemas y una primera regla de decisión implementable. La evaluación separará error, residual, costo y estabilidad mediante controles cercanos y ablaciones. La presente versión deja definido ese protocolo sin anticipar evidencia experimental que aún debe producirse.
''')
put('e-antecedentes.tex',r'''
\label{sec:antecedentes}
Este capítulo sitúa el origen del proyecto. El desarrollo conceptual, la literatura actual y sus ecuaciones se reúnen en el Capítulo~\ref{sec:marco-teorico}, para evitar repetir aquí un segundo marco teórico.

El antecedente local es la tesis de José Daniel González Carrillo, \emph{Desarrollo de una herramienta de simulación para modelar el efecto piezoeléctrico utilizando PINNs}, presentada en la Universidad del Valle de Guatemala en 2024 \cite{gonzalez2024tesis}. Su comparación entre arquitecturas PINN y FEM aporta el contexto de una viga bimorfa y permite examinar exigencias mecánicas y eléctricas conjuntas. El preprint posterior de González y colaboradores se conserva como continuación bibliográfica relacionada, con su estado editorial explícito \cite{gonzalez2026pinnpiezo}.

En el repositorio de este proyecto existen notebooks de calentamiento sobre calor, Navier--Stokes y variantes KAN. Constituyen una base de implementación y exploración; no se consideran evidencia confirmatoria de la regla que aquí se define. La investigación amplía esa base hacia una pregunta sobre reasignación de representación, conservando la conexión piezoeléctrica y un banco acotado.

La transición de antecedentes a propuesta exige reconocer qué ya existe: grillas adaptables, muestreo por residual, bases locales y transferencias multinivel. El mapa crítico del marco teórico establece esos precedentes y evita identificar una nueva combinación de componentes con una contribución ya demostrada.
''')
put('f-justificacion.tex',r'''
\label{sec:justificacion}
La adaptación tiene valor cuando asigna recursos donde producen una mejora que compensa su costo. Aumentar uniformemente una red o evaluar más puntos puede mejorar precisión, pero no explica si una representación interna permite una asignación más eficaz. Estudiar esa relación es relevante para problemas con transiciones estrechas, escalas distintas y campos acoplados, donde un promedio global puede ocultar errores físicamente importantes \cite{grossmann2024,wang2023expertguide}.

El aporte buscado es una regla de decisión caracterizada experimentalmente, apoyada en componentes publicados. Los preprints sobre densificación residual, adaptación dinámica, bases radiales y transferencia multinivel orientan rutas adyacentes y comparadores cercanos \cite{leong2026,rigas2026,freerbf2026,actor2026}. No invalidan la investigación; precisan qué tendría que demostrar para aportar conocimiento adicional.

La piezoelectricidad justifica evaluar más que una única salida escalar. Desplazamiento, potencial y flujos se relacionan mediante un constitutivo que hace visibles los efectos de alterar derivadas. El antecedente local ofrece continuidad temática y figuras de referencia, mientras una formulación explícita y una referencia numérica convergida permiten distinguir errores del mecanismo de errores del modelo \cite{gonzalez2024tesis,lahmer2008}.

El banco limitado a cinco familias reduce costo de desarrollo y mantiene diversidad de estrés. Su utilidad científica dependerá de la selección de controles, la independencia de evaluación y la comunicación de fallas. Un resultado negativo bien atribuido también delimitará cuándo la complejidad adicional del controlador no se justifica; no se confundirá ese aporte de caracterización con superioridad del método.
''')
put('g-objetivos.tex',r'''
\label{sec:objetivos}
\section{Objetivo general}
Diseñar y evaluar una regla de focalización adaptativa guiada por representación interna en PIKAN, que reasigne un presupuesto fijo de bases y permita establecer cuándo mejora el error localizado a costo comparable sin degradar otras regiones o campos.
\section{Objetivos específicos}
\begin{enumerate}
\item Establecer cinco contratos de problemas con operadores, fronteras, escalas y referencias verificables que cubran frecuencia, frentes, reacción--difusión, geometría e interacción piezoeléctrica.
\item Implementar una PIKAN base y controles de muestreo y adaptación geométrica que permitan distinguir efectos de capacidad, señal y entrenamiento.
\item Construir y verificar acciones de movimiento e intercambio de intervalos, transferencia con protección de derivadas y aceptación mediante sondeos emparejados.
\item Cuantificar beneficio, costo, estabilidad y límites del mecanismo con evaluación independiente, ablaciones y análisis por región y campo.
\item Caracterizar su comportamiento en el bimorfo piezoeléctrico vinculado al antecedente local, explicitando convenciones y límites de validez del modelo efectivo.
\end{enumerate}
''')
put('h-alcance.tex',r'''
\label{sec:alcance}
La investigación abarcará cinco familias: Poisson 2D, Burgers viscosa, Allen--Cahn, Navier--Stokes estacionario DFG 2D-1 y un bimorfo piezoeléctrico 2D de PVDF. Las variantes de parámetros servirán para separar estreses dentro de esos contratos. Se estudiará una PIKAN con B-splines cúbicas como sustrato inicial y una regla que conserve el presupuesto total de coeficientes. Las alternativas radiales permanecerán documentadas como continuidad posible, sin duplicar la campaña principal.

El trabajo evaluará problemas directos deterministas. No incluirá turbulencia, optimización de geometría del dispositivo, identificación inversa de parámetros, validación clínica, ni una campaña experimental de fabricación. La referencia FEM verificará soluciones del modelo; no sustituirá validación material del PVDF. Las comparaciones no establecerán un ranking universal entre FEM, PINN y PIKAN.

La evidencia principal combinará error por región y campo, residuales, condiciones, estabilidad y cómputo total. La aprobación de una hipótesis requerirá los criterios metodológicos; la compilación del documento y las demostraciones clásicas incluidas no la satisfacen. Esta versión entrega fundamentos, contratos y especificación previa al desarrollo del mecanismo. La implementación experimental y sus resultados constituyen la etapa siguiente.
''')
put('k-conclusiones.tex',r'''
Las conclusiones de esta versión corresponden al planteamiento y la metodología; no reportan eficacia experimental de un mecanismo todavía por desarrollar.
\begin{enumerate}
\item La adaptación de grillas, el muestreo residual, la descomposición y la transferencia tienen antecedentes identificables. Una contribución deberá demostrar valor adicional frente a ellos y no basarse en atribuir novedad a la mera existencia de bases adaptables.
\item El soporte de una base KAN es local en su argumento interno; su efecto físico depende de la composición de la red. Esta distinción justifica evaluar sensibilidades y errores regionales en lugar de inferir focalización a partir de la distribución de nudos.
\item La preservación de valores no asegura preservar derivadas, residuales o flujos. El modelo piezoeléctrico hace explícita esa diferencia y requiere coherencia en cortante, permitividad y condiciones naturales.
\item Los cinco contratos cubren estreses complementarios dentro de un alcance computacional acotado. La metodología fija sustrato, acciones, transferencias, controles y criterios para que el desarrollo pueda comenzar sin redefinir retrospectivamente el éxito.
\item La regla propuesta tiene una contribución potencial falsable. Su importancia para el campo dependerá de beneficio reproducible, costo y explicación mediante ablaciones; todavía no existe evidencia para declararla una contribución demostrada.
\end{enumerate}
''')
put('l-recomendaciones.tex',r'''
\begin{enumerate}
\item Iniciar la implementación por los contratos de operadores y por las referencias; una ecuación o condición incorrecta no puede corregirse mediante adaptación de la red.
\item Mantener versionada la ficha del PVDF y contrastar las convenciones documentales con la implementación original antes de afirmar reproducción exacta. Conservar como modelo propio explícito cualquier reformulación necesaria.
\item Completar primero el núcleo de atribución y sus controles. Abrir las extensiones con las decisiones congeladas permitirá distinguir generalización de ajuste al caso.
\item Registrar acciones rechazadas, costos y fallas. Reducir costo del controlador será parte del problema científico si su diagnóstico consume la mejora que produce.
\item Actualizar la búsqueda de literatura al cerrar resultados, incluyendo nuevas versiones de los preprints citados. Si un antecedente converge con la regla propuesta, ajustar el alcance de la contribución y fortalecer la comparación en lugar de omitirlo.
\item Revisar las afirmaciones finales con base en los criterios de falsación. Publicar un límite reproducible cuando corresponda y reservar afirmaciones de superioridad para el dominio respaldado por evidencia.
\end{enumerate}
''')
put('n-anexos.tex',r'''
\section{Notación y separación conceptual}
\begin{longtable}{p{.20\linewidth}p{.68\linewidth}}\toprule
Símbolo & Significado\\\midrule
$\mathbf u,\mathbf d,\varphi$ & Campos generales; desplazamiento mecánico; potencial eléctrico.\\
$r,\mathbf e$ & Residual del operador y error frente a solución o referencia.\\
$\theta,\xi$ & Coeficientes entrenables y geometría/estructura de bases.\\
$G_j,p,P$ & Intervalos por arista, grado spline y presupuesto entrenable.\\
$C^E,e,\kappa^S$ & Rigidez, acoplamiento piezoeléctrico y permitividad.\\
$\sigma,\mathbf D$ & Tensor de tensión y desplazamiento eléctrico.\\
$T,D,V_A,V,E$ & Entrenamiento, diagnóstico, aceptación, validación y prueba.\\
$J,\widehat U_a$ & Diagnóstico físico y beneficio estimado por costo de una acción.\\\bottomrule
\end{longtable}

\section{Procedencia y estado de las figuras}
Las Figuras~\ref{fig:fem}--\ref{fig:fem-orden} provienen de una implementación propia de FEM triangular, inspirada en la formulación del tutorial FEniCS. La malla más fina tiene 1089 nodos y 2048 triángulos, error relativo $L^2=0.0027008793$ y residual algebraico relativo $3.26\times10^{-14}$. Los últimos órdenes observados son 1.9935 y 0.9973 en $L^2$ y seminorma $H^1$. Estos números describen únicamente esa demostración.

Las Figuras~\ref{fig:tesis-fem} y \ref{fig:tesis-deformacion} son reproducciones recortadas del PDF proporcionado por el investigador, con página y número de figura identificados. Los datos siguen siendo resultados de González Carrillo. Las figuras de arquitectura, bases y problemas son recreaciones con procedencia indicada en sus pies. El diagrama del controlador es diseño propio apoyado en componentes citados. No se han sustituido simulaciones científicas por imágenes generadas sin datos.

Los archivos gráficos integrados están en \path{figuras/revision2026/}. El registro de fuentes, scripts de generación, comprobaciones y copia previa de los archivos de la plantilla se encuentra en \path{var/agent_runs/integracion-zmain-2026-09-07/}. Ese registro permite reconstruir la edición; el documento principal sigue siendo \path{z-main.tex}.

\section{Auditoría de integración}
El marco teórico contiene fundamentos, antecedentes y literatura. Los contratos P1--P5, umbrales, semillas, acciones, arquitectura operativa y presupuesto se ubican en metodología. Resumen, introducción, justificación, objetivos, alcance y cierre se alinean con esa distinción. Los espacios administrativos no designados se conservan como líneas de firma, sin atribuir aprobaciones.

La versión no modifica las políticas del repositorio ni promueve notas del vault. Las instrucciones contenidas en documentos adjuntos se trataron como material de revisión y se contrastaron con la solicitud expresa del investigador. La integración de texto y figuras no supone que los resultados futuros ni las hipótesis científicas hayan sido aceptados.
''')
data=(ROOT/'0-datos_estudiante.tex').read_text(encoding='utf-8')
data=re.sub(r'\\def \\nombreasesor .*',r'\\def \\nombreasesor {Daniel Gonzales}\n\\def \\nombresupervisor {Gabriel Barrientos}',data)
data=re.sub(r'\\def \\nombreprimerex .*',r'\\def \\nombreprimerex {\\rule{2.8in}{0.4pt}}',data)
data=re.sub(r'\\def \\nombresegundoex .*',r'\\def \\nombresegundoex {\\rule{2.8in}{0.4pt}}',data)
data=data.replace('% Los campos marcados como pendientes deben confirmarse antes de la entrega.','% Asesor y supervisor confirmados por el investigador el 7 de septiembre de 2026.\n% El tribunal no designado conserva líneas de firma sin nombres supuestos.')
put('0-datos_estudiante.tex',data)
main=(ROOT/'z-main.tex').read_text(encoding='utf-8')
# Identify the advisor on the approval sheet without assigning the supervisor a tribunal role.
main=main.replace('\\nombreasesor\n\t\\end{center}','\\nombreasesor \\\\ Asesor\\[0.2cm]\n        {\\normalsize Supervisor: \\nombresupervisor}\n\t\\end{center}',1)
put('z-main.tex',main)
bib=(ROOT/'m-bibliografia.bib').read_text(encoding='utf-8')
existing=set(re.findall(r'@\w+\{([^,]+),',bib))
old=(OLD/'m-bibliografia.bib').read_text(encoding='utf-8')
for entry in re.split(r'(?=^@)',old,flags=re.M):
    m=re.match(r'@\w+\{([^,]+),',entry)
    if m and m[1] not in existing: bib+='\n'+entry;existing.add(m[1])
bib+=r'''
@thesis{gonzalez2024tesis,
 author={González Carrillo, José Daniel},
 title={Desarrollo de una herramienta de simulación para modelar el efecto piezoeléctrico utilizando PINNs},
 institution={Universidad del Valle de Guatemala}, type={Trabajo de graduación}, date={2024},
 url={https://repositorio.uvg.edu.gt/entities/publication/75e814c0-864f-4d33-b9df-224f259de11f/full},
 note={PDF proporcionado por el investigador, 128 páginas; figuras 38 y 40 reproducidas con atribución}}
@online{deepxdeAC,
 author={{DeepXDE contributors}}, title={Allen--Cahn equation: forward PINN example},
 url={https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html},
 urldate={2026-09-07}, note={Documentación versión 1.12.2; d=0.001, condiciones Dirichlet}}
@online{dfgRe20,
 author={{Featflow, TU Dortmund}}, title={DFG flow around cylinder benchmark 2D-1, laminar case (Re=20)},
 url={https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html},
 urldate={2026-09-07}, note={Geometría, condiciones y valores de referencia oficiales}}
@misc{freerbf2026,
 author={Chiu, Shao-Ting and Cheung, Siu Wun and Braga-Neto, Ulisses and Lee, Chak Shing and Li, Rui Peng},
 title={Free-RBF-KAN: Kolmogorov-Arnold Networks with Adaptive Radial Basis Functions for Efficient Function Learning},
 date={2026}, eprint={2601.07760}, eprinttype={arxiv},
 url={https://arxiv.org/abs/2601.07760v3}, note={Preprint; ficha de la versión v3, 6 de febrero de 2026, consultada}}
@article{burgerswaves2025,
 author={Samanta, Pintu and Chidella, Srinivasa Rao},
 title={Exact and asymptotic solutions of Burgers' equation on half-line},
 journaltitle={The Quarterly Journal of Mechanics and Applied Mathematics},
 volume={78}, number={1}, eid={hbaf001}, date={2025}, doi={10.1093/qjmam/hbaf001},
 note={Antecedente de ondas viajeras; el contrato finito de este proyecto y sus parámetros se derivan por sustitución}}
'''
put('m-bibliografia.bib',bib)
print('Integradas secciones raíz, datos confirmados y bibliografía; políticas y vault sin cambios.')
