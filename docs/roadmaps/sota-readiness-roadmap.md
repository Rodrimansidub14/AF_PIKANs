# Ruta de evidencia hacia resultados comparables con SOTA

**Actualizado:** 2026-09-23
**Estado:** guía de investigación; no es un preregistro firmado ni una afirmación de superioridad.
**Decisión de alcance:** documentar los resultados exploratorios de EXP-0002 y EXP-0005, precisar qué significaría una comparación SOTA para AF_PIKANs y establecer qué evidencia falta antes de ampliar las afirmaciones.

La secuencia de campañas y la distinción entre baseline instrumental, comparadores y candidatos están en el [plan de campañas](../plans/campaign-plan-exploration-to-sota.md).

## Resumen ejecutivo

AF_PIKANs conserva EXP-0002 como baseline instrumental reproducible de Heat y un Burgers estacionario, con RBF-PIKAN fija, PINA/PyTorch, dos T4, W&B y artefactos locales. EXP-0005 completó una comparación nueva y pareada de RBF-PIKAN, MLP PINN y B-spline PIKAN para Heat y el Burgers transitorio canónico de Raissi/PINNacle. No se reutilizan como si los dos contratos de Burgers fueran equivalentes. EXP-0005 tiene tres semillas y sigue siendo exploratorio; no establece SOTA.

El objetivo científico de mayor valor no es afirmar que una arquitectura sea “SOTA” en general. La hipótesis de proyecto que se puede someter a prueba es que señales cuantificables de la red y del residual pueden guiar la focalización de puntos y mejorar el compromiso entre error físico, costo y estabilidad de una PIKAN. Esa afirmación debe evaluarse por problema y frente a controles explícitos.

El siguiente paso es interpretar las señales cuantificadas de EXP-0005 y definir un preregistro nuevo: métrica primaria, tolerancias, costo máximo, criterio de aceptación y una única intervención contra Sobol fijo. Después se decide si la evidencia justifica una campaña de focalización o avanzar a Allen–Cahn/DFG. No hay claim de superioridad pendiente de publicar; EXP-0005 registra mediciones exploratorias y sus gates científicos no fueron preregistrados.

## Progresión de problemas físicos

La secuencia metodológica recomendada es **Heat en una varilla → Burgers viscoso 1D → Navier–Stokes DFG Re=20**. Heat verifica derivadas segundas y condiciones iniciales/frontera con solución exacta. Burgers añade advección no lineal, difusión y formación de un frente; es un puente escalar hacia modelos de flujo, pero no incluye presión ni incompresibilidad. DFG Re=20 aporta el ancla de Navier–Stokes 2D con geometría, presión, continuidad y observables integrales. Esta secuencia no reemplaza automáticamente el conjunto objetivo acordado: Poisson permanece pendiente mientras se formaliza cualquier cambio de alcance.

## Qué quiere decir “SOTA” en este proyecto

Antes de usar esa etiqueta hay que declarar la unidad de comparación y la afirmación exacta.

| Tipo de afirmación | Qué se entrena | Evidencia que exigiría |
|---|---|---|
| Solución de una instancia (función) | Una red representa la solución de una PDE concreta con coeficientes, condiciones y dominio fijos. Es el alcance de EXP-0002. | Referencia independiente convergida, comparadores equivalentes, métricas físicas y de campo, presupuesto igualado y repeticiones. |
| Operador de solución | Una red aprende a mapear una familia de condiciones iniciales, forzamientos, coeficientes o condiciones de frontera a familias de soluciones. | Particiones train/validation/test por instancias PDE, evaluación de generalización dentro y fuera de distribución, costo de entrenamiento e inferencia amortizado. |
| Resultado general en SciML | Una conclusión pretende cubrir varias PDE o dominios. | Comparación independiente por problema y agregación justificada; un promedio entre PDEs no sustituye los análisis específicos. |

PDENNEval distingue métodos de aprendizaje de funciones y métodos de aprendizaje de operadores, y evalúa doce métodos en diecinueve problemas PDE. Por ello, un buen resultado en una única solución de Burgers no prueba un resultado de operator learning ni una superioridad general en SciML ([PDENNEval, IJCAI 2024](https://www.ijcai.org/proceedings/2024/573)).

**Afirmación acotada que el proyecto podría buscar:** “Una estrategia de focalización guiada por señales medidas de RBF-PIKAN mejora el error local y/o reduce el costo bajo un presupuesto fijado, frente a Sobol fijo y comparadores elegidos, en los casos PDE declarados”. La frase solo se puede usar si cada término queda medido y pasa los gates preregistrados.

## Estado empírico disponible: EXP-0002

La campaña `heat-burgers-sobol-002-t4x2` es exploratoria. Se usaron semillas 101, 102 y 103; una semilla por worker/GPU en dos Tesla T4; seis runs (dos PDE por tres semillas). El registro local indica que los seis runs W&B sincronizaron. Los promedios y desviaciones siguientes se calcularon sobre los tres archivos `metrics.json` de cada problema; la desviación es muestral.

| Problema | Configuración principal | Error relativo de campo | RMSE del residual de validación | Tiempo de entrenamiento y diagnósticos |
|---|---|---:|---:|---:|
| Heat 1D | `alpha=0.1`; 4.096 puntos interiores; 5.000 pasos; malla de validación `129×65` | `(8.8755 ± 4.5426)×10⁻⁵` | `(9.2246 ± 1.1627)×10⁻⁴` | `191.85 ± 10.57 s` |
| Burgers 1D | `nu=0.02`; `c=0`, `A=0.5`, `x0=-0.25`; 8.192 interiores; 20.000 pasos; malla `129×65` | `(2.7561 ± 0.3154)×10⁻³` | `(3.6360 ± 1.3023)×10⁻³` | `1021.88 ± 52.21 s` |

Para Burgers, el RMSE de campo normalizado en la región del frente fue `(3.8932 ± 0.8217)×10⁻³`; la región del frente tiene una métrica separada del complemento. El error global y el error local no son intercambiables.

La malla `129×65` es una validación exploratoria, no una demostración de convergencia de cuadratura/refinamiento para la afirmación final. Tres semillas describen dispersión inicial, pero no bastan para declarar robustez estadística. Los resultados no se comparan con trabajos que usen otra viscosidad, amplitud, dominio, condición inicial, precisión o presupuesto.

**Estado del gate:** la campaña no tiene umbrales numéricos de aceptación científica; en el manifiesto actual las métricas indican dirección de optimización, no límites de aprobación. El archivo [`evaluation.json`](../../var/agent_runs/heat-burgers-002-t4x2/download-retry/evaluation.json) registra `execution_valid=true`, `eligible_for_review=false` y `thresholds_pass=false`; la aprobación científica humana sigue pendiente. La corrida se conserva como evidencia exploratoria, no se renombra como SOTA ni como baseline aceptado.

**Definición nueva para EXP-0005:** Heat conserva la conducción 1D en varilla con modo seno y solución exacta. Burgers ahora usa `u_t + u u_x - (0.01/π)u_xx=0`, dominio `[-1,1]×[0,1]`, condición inicial `-sin(πx)` y fronteras nulas; la referencia se calcula por Cole–Hopf. El evaluador identifica ese caso como `burgers_pinnacle_1d` y mantiene el shock estacionario bajo el contrato histórico `burgers`. Véase el [contrato detallado y referencias](../../var/agent_runs/heat-burgers-architecture-screen/problem-definition-proposal.md).

**Desfase del índice:** al corte de esta guía, `experiments/README.md` y `experiments/roadmap-next-problems.md` todavía describen EXP-0002 como ejecutándose. Los artefactos de evaluación y el registro `wandb-sync.json` muestran la campaña terminada y sincronizada. Hay que reconciliar ese estado en el índice del experimento antes de firmar el siguiente pre-registro.

### Parámetros e instrumentación existentes

Las decisiones completas de dominio, bases, red, precisión, normalización, pools, semillas, batch, optimizador, pasos, límites de tiempo, diagnóstico, W&B y versiones están en [EXP-0002: decisiones de parámetros](../../experiments/heat/EXP-0002-heat-burgers-sobol/parameter-decisions.md) y en el manifiesto versionado `manifest-v3-t4x2.json`. No se vuelven a decidir aquí.

El modelo exploratorio usa arquitectura `[2,16,16,1]`, 11 bases RBF gaussianas por arista, centros y ancho fijos, float64 y 3.648 parámetros. El muestreo de entrenamiento es Sobol scrambled fijo, con pools guardados. Este diseño aísla señales bajo una geometría estable; no demuestra que centros, anchos, arquitectura o densidad sean óptimos.

La instrumentación disponible cuantifica, entre otras señales:

- **Generación de puntos:** conteos por región, límites observados, discrepancia centrada Sobol, distancias al vecino más cercano, candidatos y pools exactos.
- **Entrenamiento y costo:** pérdidas, pasos, ejemplos procesados, throughput, tiempo de entrenamiento y diagnósticos, memoria pico y telemetría GPU.
- **Autodiferenciación y física:** `u`, derivadas físicas disponibles (`u_x`, `u_t`, `u_xx`), residual, medias/RMS/percentiles/máximos y escalas normalizadas en pool diagnóstico independiente.
- **Red y bases:** gradiente global y por capa, normas de parámetros/coefs, ocupación y número efectivo de bases RBF, activaciones, saturación y sensibilidad `du/dtheta` en 32 puntos.
- **Validación:** error global y, para Burgers, error/residual de frente y complemento; condiciones iniciales/frontera; campos guardados y visualizaciones Plotly.

Conflicto entre gradientes de pérdidas, rango efectivo del Jacobiano y norma de actualización no deben reportarse como medidos en esta campaña: están pendientes. Las señales descriptivas tampoco demuestran causalidad; para demostrar que guían una mejora hay que intervenir sobre ellas en una campaña posterior.

## Alcance de problemas y métricas

El conjunto objetivo de cuatro problemas originalmente acordado es **Poisson, Burgers, Allen–Cahn y DFG Re=20**. Heat se añadió como control inicial de desarrollo en EXP-0002; no reemplaza a Poisson en el conjunto objetivo. Poisson permanece pendiente. Los borradores de Allen–Cahn y DFG tampoco quedan aprobados para ejecutarse por estar documentados aquí.

| Problema | Papel y estado | Métricas y señales necesarias para una comparación seria |
|---|---|---|
| Heat 1D | Conducción en varilla; control suave, solución analítica y control arquitectónico en EXP-0005. | Error relativo L2/RMSE y máximo en mallas refinadas, residual PDE, condición inicial/frontera, costo y tasa de convergencia. |
| Poisson 2D | Parte del conjunto objetivo; pendiente definir dominio y condiciones del protocolo. | Error de solución y gradiente/energía, residual, condición de frontera y referencia analítica o numérica convergida; revisar singularidades o esquinas si el dominio las contiene. |
| Burgers 1D | EXP-0002: shock estacionario tanh. EXP-0005: benchmark transitorio Raissi/PINNacle con frente estrecho y referencia Cole–Hopf. | Error de campo global y local alrededor del frente, L∞/percentiles, residual, condición inicial/frontera; no combinar ambos contratos al agregar resultados. |
| Allen–Cahn 1D | Borrador EXP-0003; campo final bloqueado hasta converger/auditar la referencia BDF. | Error temporal/de campo solo tras validar referencia; ubicación/ancho de interfaz, energía Allen–Cahn, residual y condiciones impuestas. Separar señales exploratorias de métricas desbloqueadas. |
| DFG 2D-1, Re=20 | Borrador EXP-0004; auditar referencia FEM, outlet e integrales antes de comparar campo o QoI. | Errores de velocidad/presión con gauge declarado, continuidad/divergencia, entrada/pared/cilindro/salida, regiones near-cylinder/wake y `drag`, `lift`, `Δp` frente a FEATFLOW/referencia convergida. |

PDEBench propone un análisis más amplio que RMSE y combina desempeño de solución con características de física y de conjuntos de simulaciones; el conjunto de métricas se debe ajustar al fenómeno, no copiar como una métrica única para todas las PDE ([PDEBench, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html)).

## Protocolo de comparación recomendado

Este protocolo es una recomendación para futuros preregistros, no un cambio retroactivo a EXP-0002.

1. **Fijar el contrato de cada PDE.** Congelar ecuación, coeficientes, unidades, dominio, condiciones, referencia y tolerancia de convergencia. Para cada caso declarar si el modelo resuelve una instancia o aprende un operador.
2. **Separar datos de decisión y datos finales.** Usar entrenamiento para optimizar; validation para seleccionar mecanismo/hiperparámetros; reservar una evaluación final bloqueada. Un cambio motivado por resultados held-out crea una nueva hipótesis y comparación, no modifica los resultados pasados.
3. **Fijar comparadores y presupuesto.** Como mínimo, RBF-PIKAN fija, PINN-MLP y PIKAN B-spline; añadir métodos especializados solo con código, hiperparámetros y condiciones reproducibles. Igualar condiciones físicas, semillas, pools cuando corresponda, precisión, tiempo/pasos y presupuesto de ajuste. Informar también parámetros, memoria, throughput y costo.
4. **Cambiar un factor experimental a la vez.** Mantener Sobol fijo como control; comparar muestreo aleatorio/LHS y después RAD o RAR-D con el mismo presupuesto total de puntos y el mismo pool de evaluación. Separar muestreo adaptativo, balanceo de pérdidas y geometría RBF aprendible en hipótesis distintas. Wu et al. estudian seis métodos uniformes —incluido Sobol— y estrategias adaptativas RAD/RAR-D en miles de simulaciones PINN; eso los vuelve comparadores razonables, no una garantía de mejora en AF_PIKANs ([Wu et al.](https://arxiv.org/abs/2207.10289)).
5. **Robustez y reporte.** Usar las tres semillas existentes solo como screening. Para finalistas, planificar al menos cinco semillas independientes como mínimo operativo del proyecto y más si la dispersión observada deja intervalos inconclusos. Reportar cada semilla, media/desviación, intervalo de incertidumbre y fallos/tiempos de parada; no ocultar divergencias.
6. **Definir gates antes de abrir evaluación.** Cada pre-registro debe fijar métrica primaria, magnitud mínima de mejora relevante, tolerancias físicas, costo máximo y falsadores. No escoger umbrales tras mirar el score final. `evals/` es el verificador protegido; cambios a grader o umbral requieren el gate de integridad indicado por `AGENTS.md`.
7. **Revisión científica humana.** El harness puede validar artefactos y aplicar gates predefinidos; el humano decide si el resultado sostiene el claim y autoriza su incorporación a la tesis o al registro de claims.

### Screening inicial completado: EXP-0005

EXP-0005 completó 18 runs (tres arquitecturas × dos PDE × tres semillas) con pools Sobol fijos pareados, límites de pasos/tiempo declarados y 18/18 runs sincronizados con W&B. El reporte [HTML interactivo](../../var/agent_runs/heat-burgers-architecture-screen/analysis/architecture-screen-comparison.html), el [notebook reproducible](../../notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-comparison.ipynb), [datos por semilla](../../var/agent_runs/heat-burgers-architecture-screen/analysis/seed-results.csv) y [señales cuantificadas](../../var/agent_runs/heat-burgers-architecture-screen/analysis/quantified-signals.csv) conservan la evidencia y sus límites. Se registraron 2.622 observaciones señal-semilla de 354 métricas escalares. Los parámetros por modelo y decisiones individuales están en `var/agent_runs/heat-burgers-architecture-screen/models/`.

Como descriptivo (media ± desviación estándar muestral, n=3), el menor error relativo L2 de Heat fue B-spline (`2.30e-5 ± 9.49e-6`); en Burgers fue RBF (`0.0540 ± 0.0393`), y RBF también tuvo el menor RMSE normalizado del frente (`0.292 ± 0.229`). La variabilidad entre semillas es material. El evaluador no marca aceptación científica porque no se establecieron umbrales numéricos antes de correr. Estos resultados sirven para plantear hipótesis y no permiten afirmar SOTA ni causalidad.

| Etapa | Comparación | Qué decide |
|---|---|---|
| A. Referencia y controles | Revisar solución Heat, referencia Burgers y evaluación final; repetir RBF fija solo si la reconstrucción del control es necesaria; correr PINN-MLP y PIKAN B-spline bajo el mismo harness y protocolo equivalente. | Si la ventaja o dificultad observada depende de la arquitectura y no de presupuesto/datos. |
| B. Señal y muestreo | Mantener el mejor control y comparar Sobol fijo contra una política adaptativa (RAD o RAR-D primero; no ambas junto con otros cambios). | Si la distribución adaptativa mejora error local/global o costo, y si la señal predice dónde mejorar. |
| C. Bases | Como hipótesis aparte, comparar centros/anchos fijos contra geometría RBF aprendible. Free-RBF-KAN de 2026 es un candidato bibliográfico reciente, pero es preprint y su configuración publicada no coincide con este problema; reimplementar o transferir sus resultados no basta como validación propia ([Free-RBF-KAN](https://arxiv.org/abs/2601.07760)). | Si la flexibilidad de base aporta fuera de la diferencia de cómputo/optimización. |
| D. Confirmación | Tomar solo finalistas; usar más semillas, referencia convergida, malla/test bloqueados y análisis de incertidumbre. | Si la mejora se mantiene y merece pasar a la siguiente PDE o a un claim. |

No combinar en el primer screening Free-RBF-KAN, RAD/RAR-D, balanceo de pérdidas y cambios de optimizador: así no se podría atribuir el resultado a una causa. La comparación FAIR MLP/KAN reporta que algunas variantes KAN pueden ser sensibles a inicialización/semillas y que el costo/precisión de B-spline KAN no es universalmente favorable; es otra razón para medir controles y dispersión en los casos reales del proyecto ([Shukla et al.](https://arxiv.org/abs/2406.02917)).

Para Allen–Cahn y otros casos con grillas factorizables se puede añadir SPIKAN como candidato arquitectónico separado. Su estructura separable tiene un protocolo de cálculo distinto; no se debe comparar su conteo de puntos directamente con el pool disperso Sobol sin igualar el problema, evaluación y costo ([SPIKANs](https://doi.org/10.1088/2632-2153/ae05af)).

## Estructura de autoresearch / eval-based development

Cada iteración automatizada debe tener una sola hipótesis y una superficie de edición permitida:

```text
hipótesis y presupuesto preregistrados
  → candidato versionado y manifiesto inmutable
  → smoke/regresión barata (Tier 0/1)
  → ejecución autorizada (Tier 2)
  → artefactos, W&B y mediciones locales
  → verifier congelado / scorecard
  → aceptar para siguiente estudio, descartar o marcar inconcluso
  → gate humano antes de cualquier claim científico
```

El loop debe optimizar una métrica primaria previamente declarada sujeto a tolerancias físicas, tiempo y memoria; los demás indicadores son métricas secundarias y diagnósticos. No debe leer/editar graders para mejorar la puntuación. Un resultado “mejor” de una sola semilla o un residual de entrenamiento menor no basta para aceptar un mecanismo.

## Artefactos y visualización

La combinación acordada sigue siendo adecuada para las etapas actuales:

| Uso | Herramienta / artefacto |
|---|---|
| Entrenamiento | PINA + PyTorch; RBF-PIKAN instrumentada. |
| Seguimiento | W&B y copia local reproducible, con config, versiones, métricas, hashes y checkpoints. |
| Curvas/señales | Plotly: pérdidas, cobertura de puntos, residuales, error local/global, ocupación de bases y costo. |
| Mallas, campos y geometrías | PyVista en modo sin ventana; principalmente para DFG y geometrías/mallas que lo requieran. |
| Aplicación de análisis | Panel + Plotly/PyVista; instrucciones y rutas locales en [dashboard de análisis](../reference/analysis-dashboard.md). |
| Notebook | Notebook de desarrollo con secciones, código ejecutable, tablas medibles, visualizaciones y rutas relativas de artefactos. |

Para mejorar la lectura de magnitudes pequeñas, las curvas positivas de pérdida, error, gradiente y residual se muestran con eje logarítmico. El mapa de error absoluto usa `log10(|error|)` para el color; los campos con signo se quedan lineales. Ceros exactos se dejan sin color en esa escala, en lugar de sumar un epsilon que altere su magnitud. Las mismas figuras de entrenamiento y validación se guardan como HTML reproducible y se registran como Plotly en W&B; el panel agregado del dashboard usa eje log si todos los valores observados son positivos.

Los outputs actuales están en `var/agent_runs/heat-burgers-002-t4x2/download-extracted/analysis/`; el [notebook de resultados](../../notebooks/heat/EXP-0002-heat-burgers-sobol/heat-burgers-baseline-t4x2-results.ipynb) y sus HTML/CSV son evidencia derivada de la corrida. El archivo local de sync registra seis de seis runs sincronizados. No guardar claves API, tokens ni `.env` en el notebook, en W&B config ni en este documento.

## Criterios mínimos antes de decir “SOTA”

Una afirmación debe cumplir todos los puntos que apliquen a su alcance:

- El benchmark exacto y la clase de tarea (instancia o operador) están nombrados.
- Los protocolos comparados corresponden a la misma PDE, dominio, coeficientes, condiciones, referencia, resolución y presupuesto, o se explican las diferencias.
- Hay al menos un control MLP PINN y un control PIKAN pertinente, además de los métodos que el benchmark declare como comparadores.
- La referencia numérica es independiente y está convergida/auditada; los errores se reportan globales y en regiones físicas difíciles.
- La evaluación final permanece bloqueada hasta congelar la selección; se presentan múltiples semillas, dispersión y fallos.
- Se muestran curvas error/costo y señales cuantificadas, no solo una lectura visual o un número de loss.
- El resultado y configuración se pueden reconstruir desde código, manifiesto, versiones, pools, semillas y artefactos locales/W&B.
- La suite congelada aprueba los gates preinscritos y el investigador humano aprueba el claim. Si no hay literatura/protocolo idénticos, la conclusión se formula como “mejora sobre los comparadores evaluados”, no como “SOTA”.

## Registro de decisiones y pendientes

| ID | Decisión o estado | Justificación / referencia local |
|---|---|---|
| D-SOTA-01 | EXP-0002 se clasifica como exploratorio, no como claim ni baseline científico aceptado. | Manifiesto con `protocol_phase=exploratory`; umbrales de dirección sin cutoff numérico; tres semillas piloto. |
| D-SOTA-02 | Sobol scrambled fijo es el control de muestreo actual. | Mantiene una distribución reproducible para atribuir señales; estrategias adaptativas serán una hipótesis separada. |
| D-SOTA-03 | Heat es control auxiliar; Poisson sigue dentro del objetivo de cuatro y pendiente. | Acuerdo previo del investigador; Heat entró en la primera campaña junto a Burgers. |
| P-SOTA-01 | Aprobar problema, referencia y umbrales de cada scorecard antes de corrida confirmatoria. | Requiere pre-registro humano conforme a `experiments/README.md`. |
| P-SOTA-02 | Auditar/converger referencia de Burgers y hacer estudio de refinamiento de la evaluación. | La evaluación actual usa malla `129×65`; no basta para claim final. |
| P-SOTA-03 | Ejecutar comparación matched-budget frente a PINN-MLP y PIKAN B-spline. | Aún no hay comparadores pareados para EXP-0002. |
| P-SOTA-04 | Preregistrar RAD o RAR-D contra Sobol fijo; investigar si las señales medidas explican la mejora. | No inferir focalización a partir de mapas de señal sin intervención controlada. |
| P-SOTA-05 | Completar referencia de Allen–Cahn y auditoría FEM/QoI de DFG; definir dominio/protocolo de Poisson. | Ver borradores [EXP-0003](../../experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/spec.md), [EXP-0004](../../experiments/navier-stokes/EXP-0004-dfg-re20-rbf-sobol/spec.md) y [roadmap](../../experiments/roadmap-next-problems.md). |
| P-SOTA-06 | Reconciliar el estado “en curso” de los índices de experimentos con la campaña EXP-0002 terminada. | Evitar que el dashboard y la documentación del siguiente preregistro lean estados contradictorios. |

Los puntos `P-SOTA-*` son propuestas pendientes, no parámetros aprobados ni autorización para consumir GPU. Cualquier campaña futura debe crear manifiesto y decisión de parámetros nuevos, preservando íntegro el resultado actual.

## Referencias principales

- Wei et al., [PDENNEval: A Comprehensive Evaluation of Neural Network Methods for Solving PDEs](https://www.ijcai.org/proceedings/2024/573), IJCAI 2024.
- Takamoto et al., [PDEBench: An Extensive Benchmark for Scientific Machine Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html), NeurIPS 2022.
- Wu et al., [A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs](https://arxiv.org/abs/2207.10289), 2022.
- Shukla et al., [A comprehensive and FAIR comparison between MLP and KAN representations for differential equations and operator networks](https://arxiv.org/abs/2406.02917), 2024.
- Chiu et al., [Free-RBF-KAN: Kolmogorov-Arnold Networks with Adaptive Radial Basis Functions for Efficient Function Learning](https://arxiv.org/abs/2601.07760), preprint 2026. Es candidato para probar, no benchmark aceptado ni evidencia directa para los parámetros AF_PIKANs.


