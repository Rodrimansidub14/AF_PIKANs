# Plan de caracterización del mecanismo en piezo (objetivo 3), 2026-10-26

Estado: **propuesta para decidir; nada firmado.** Revisa y amplía `spec.md` de EXP-0030 (descriptivo, ya corrido en CPU sobre EXP-0029). Alcance asumido: la caracterización es del **mecanismo (E y su variante graduada M) en piezo**, con B como referencia; el banco de otros problemas entra solo como contexto ya medido (`results/phase1*_scored.md`), sin análisis nuevos. Si quisieras caracterizar también el banco, dilo: cambia el costo.

## 1. Revisión de EXP-0030 como está
**Bien:** declara que es exploratorio y post hoc, no emite claims, separa efecto, señales, física, incertidumbre y responsible AI, y descubrió dos cosas que importan: (i) hardware × brazo está confundido por completo en M frente a E; (ii) las cifras de cuota por grupo dependen de cómo se agregue (mediana por arista de la capa 0, o suma sobre todas las aristas).
**Faltaba:** (a) preguntas explícitas que organicen el análisis; (b) explicar el *por qué* (dinámica del entrenamiento y estructura de curvatura), no solo describir; (c) una decisión razonada de qué métodos de la lista de xAI sirven y cuáles no; (d) la lista de simetrías de admisibilidad (declarada, no calculada); (e) `CIERRE_OBJETIVO3.md`, que el registro dice haber escrito, **no está en el repositorio**.
**Corrido ahora (CPU, gratis):** bloque H (alineación de gradientes por grupo físico y dispersión por arista) en `analysis/a07_alineacion_gradientes.py`, resultados en `results/H_alineacion.md`.

## 2. Preguntas de la caracterización
| # | Pregunta | Qué cuenta como respuesta | Estado |
|---|---|---|---|
| Q1 | ¿Qué diferencia hay entre B, E y M en error final, cola y tiempo de salida del llano, y cuánto resiste a las semillas y al hardware? | razones pareadas, subconjuntos de semillas, banda del mismo hardware | hecho (A, A2, B) |
| Q2 | ¿Por qué E sale antes del llano? ¿En qué paso se separan las trayectorias y qué precede a la separación? | trayectorias de gradiente, conflicto entre grupos, movimiento efectivo de coeficientes | H corrido; falta movimiento efectivo y paso de separación |
| Q3 | ¿Qué cambia E en la estructura de curvatura (rango casi uno, dispersión entre aristas, espectro de P)? | espectros de B y P, dispersión de curvatura antes y después de P | parcial (C, `analisis_piezo/`); falta la dispersión en c′ por paso |
| Q4 | ¿Qué hizo cada pata de M y por qué no conservó la ventaja de E? | señales por recálculo, T1/T2 (mismo hardware, arranque en caliente), confusión con hardware | hecho como descripción; **no identificable** con estos datos |
| Q5 | ¿Qué ecuaciones y campos mejora o empeora cada brazo, y las soluciones son admisibles? | residual por ecuación en malla reservada, carga neta, consistencia constitutiva, simetrías | D hecho; **simetrías pendientes** |
| Q6 | ¿Cómo usan las aristas de la red B, E y M (qué aristas llevan la salida, qué bases se ocupan)? | importancia por arista, ocupación de bases, funciones de arista de la capa 0 | **no hecho** (datos disponibles) |
| Q7 | ¿Qué tan identificable es la solución entre semillas y dónde se concentra la incertidumbre? | campo mediano, dispersión, correlación dispersión-error, voltaje flotante | F hecho; H2 (`check_Vf`) hecho |
| Q8 | ¿Qué tan sensible es el resultado a los hiperparámetros del mecanismo? | Morris sobre tope de P2, cadencia, piso de P4, calendario de P5 | **diferido** (requiere cómputo) |
| Q9 | ¿Dentro de qué dominio valen estas conclusiones? | tarjetas de experimento, procedencia, catálogo de fallos | G hecho |

## 3. Qué hacer con cada método de la lista
Criterio: un método entra si responde una de las preguntas; la mayoría de los métodos de xAI explican *qué usa una predicción de sus entradas*, y nuestra pregunta es *por qué el optimizador se comporta distinto*, así que muchos no aplican. Lo digo para cada uno.

| Método | Veredicto | Pregunta | Cómo / por qué |
|---|---|---|---|
| Physics residual analysis | **aplica, hecho, ampliar** | Q5 | residual por ecuación en malla reservada; añadir paso intermedio (`s004000`, `s010000`) para ver cuándo aparece cada diferencia |
| Invariance / symmetry testing | **aplica, pendiente** | Q5 | enunciar de `ALINEACION_PAPER.md` las simetrías de geometría y polarización del bimorfo, evaluarlas sobre los campos de `s020000`; no hay umbral de aprobación |
| Uncertainty and identifiability | **aplica, hecho, ampliar** | Q7 | dispersión entre semillas por brazo; ya se mostró que V̂_f es identificable (`check_Vf`); añadir rango efectivo de los coeficientes por arista |
| Neural-network pruning (importancia por arista) | **aplica, nuevo** | Q6 | ranking de aristas por energía de salida y fracción que lleva el 90 %, por brazo y semilla; solo lectura de estados; sin podar realmente |
| PDP / ALE | **adaptado: las funciones de arista de una KAN ya son la dependencia parcial** | Q6 | graficar y comparar las funciones de la primera capa entre semillas y brazos; ALE no hace falta (las entradas x, y son independientes por construcción de la malla) |
| Counterfactual explanations | **aplica en forma barata, ya hecha una vez** | Q2, Q4 | sustituir una pieza en un estado guardado (V̂_f por su óptimo: sin efecto) y medir; con arranque en caliente (T1/T2) se hizo para S/M; más ramas requieren cómputo |
| TCAV / conceptos | **se absorbe en la alineación de gradientes** | Q2 | los «conceptos» son los grupos físicos (mecánica, eléctrica, equilibrio); medir si el gradiente de cada grupo se alinea o se opone es el equivalente directo (Wang et al. 2025) |
| Morris | **diferido** | Q8 | factores: tope de P2, cadencia, piso de P4, calendario de P5; 4 niveles y ≈ 30 trayectorias en el precedente de Guo et al.; requiere cientos de corridas, hoy sin crédito |
| Sobol | **no** | Q8 | sensibilidad global necesita muchísimas más corridas que Morris; no cabe |
| Integrated Gradients, DeepLIFT, LRP | **no** | — | atribuyen la predicción a las entradas (x, y); no responden por qué cambia la optimización; LRP/DeepLIFT para capas RBF exigirían adaptación sin ganancia clara. Si en la defensa se pregunta qué usa la red, se responde con Q6 |
| Symbolic regression | **no** | — | la solución del bimorfo no tiene forma cerrada; no hay ecuación que recuperar |

**Estadística** (n ≤ 5, sin p-valores por preferencia tuya): razones pareadas por semilla, conteo de subconjuntos de semillas (hecho), **probabilidad de mejora y perfiles de desempeño** de Agarwal et al. como descripción; sus intervalos bootstrap con n = 5 serían tan anchos que no informan, y los mostraría solo si lo pides. Q2 y Q3 son descriptivos (trayectorias, espectros). Q8, cuando haya cómputo, es el único lugar de diseño de experimentos.

## 4. Literatura que respalda el enfoque (todo leído como resumen; verificar el texto completo antes de citar)
| Fuente | Se usa para | Estado |
|---|---|---|
| [Wang, Bhartari, Li, Perdikaris (arXiv:2502.00604)](https://arxiv.org/abs/2502.00604) | conflicto de gradientes tipo I (magnitud) y tipo II (dirección); el precondicionamiento de segundo orden mejora la alineación | resumen |
| [AutoBalance (arXiv:2510.06684)](https://arxiv.org/html/2510.06684v1) | espectros Hessianos distintos por componente de pérdida desordenan el precondicionador; base de P2 | resumen |
| Krishnapriyan et al. 2021; Rathore et al. 2024 | paisaje de pérdida y condicionamiento en PINN | ya en `REVISION_LITERATURA_PIEZO.md` |
| [Agarwal et al. (arXiv:2108.13264)](https://arxiv.org/pdf/2108.13264) | evaluar con pocas corridas: perfiles, probabilidad de mejora, IQM | resumen |
| Musah 2026 (arXiv:2608.21683) | admisibilidad antes de comparar residuales | resumen |
| Bouthillier et al. 2021; Benavoli et al. 2017 | varianza entre semillas; equivalencia práctica | resumen |
| [KAN survey y GINN-KAN (arXiv:2408.14780)](https://arxiv.org/abs/2408.14780) | interpretabilidad de KAN, poda y regresión simbólica | resumen; no cubre Sobol, IG ni poda en PINN |

La búsqueda de xAI específica de KAN y PINN **no encontró** un protocolo que mapee esos métodos a un mecanismo de optimización; el mapeo de §3 es juicio mío apoyado en cómo funciona cada método, no una receta de la literatura.

## 5. Cómo se concluye el objetivo 3
Con las preguntas respondidas o declaradas sin respuesta: Q1, Q3, Q5, Q7 y Q9 con datos; Q2 con el bloque H y el movimiento efectivo; Q6 con el análisis de aristas; Q4 como «no identificable con estos datos» (honesto y útil como lección de diseño); Q8 como trabajo futuro. El capítulo de caracterización se escribe como: efecto (Q1), por qué (Q2–Q3), qué pasó con las patas (Q4), física (Q5), qué aprendió la red (Q6), robustez (Q7, Q9) y límites (Q4, Q8).

## 6. Lo que falta para cerrar sin cómputo
(1) simetrías (Q5); (2) importancia por arista y funciones de arista (Q6); (3) movimiento efectivo y paso de separación B/E (Q2); (4) dispersión en c′ por paso (Q3); (5) probabilidad de mejora y perfiles (Q1); (6) redactar `CIERRE_OBJETIVO3.md` de nuevo, porque no existe. Todo en CPU con los estados guardados.
