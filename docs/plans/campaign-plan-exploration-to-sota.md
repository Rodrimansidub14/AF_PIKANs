# Plan de campañas: de exploración de señales a comparación SOTA

**Fecha:** 2026-09-23
**Estado:** EXP-0005 completó el screening autorizado: 18/18 runs terminaron, 18/18 sincronizaron con W&B y los artefactos pasaron verificación. Son resultados exploratorios; al no haber umbrales de aceptación preregistrados, no hay aceptación científica ni claim SOTA.
**Documento complementario:** [ruta de evidencia hacia SOTA](../roadmaps/sota-readiness-roadmap.md).

## Ejecución del siguiente paso: EXP-0006

La auditoría reproducible de EXP-0005 está en `var/agent_runs/burgers-rad-exp006/audit/`:
el residual del interior de Burgers/RBF tiene correlación Spearman 0.118–0.296 con
el error y el decil superior concentra error medio 2.60–4.84 veces el promedio.
Son tres semillas independientes; las celdas espaciales no son réplicas.

Se implementó y probó RAD con distribución residual cuadrática, 50% de mezcla
uniforme, 8192 puntos activos y 32768 candidatos, con PINA conservando el estado
de entrenamiento. La propuesta de cinco parejas, criterios físicos y presupuesto
está en `experiments/burgers/EXP-0006-burgers-sobol-rad/`. Aún falta firma científica y
congelación de baseline; el paquete preparado bloquea entrenamiento sin ellas.
La autorización general de ejecución ya fue recibida; no se aceptó ningún claim.

Allen–Cahn: estudio BDF candidato ejecutado en CPU con 513, 1025 y 2049 nodos;
último cambio relativo de campo 6.27e-5 y cambio por tolerancia temporal 1.26e-8.
La referencia sigue sin desbloquearse: queda precisar el criterio de cambio
respecto al efecto y revisar la regularidad en las esquinas de t=0. DFG:
primero resolver formulación y referencias de fuerzas, implementar FEM y evaluar
convergencia. El informe de brechas está en
`var/agent_runs/burgers-rad-exp006/next-problem-readiness.md`.
**Resultados:** [comparación interactiva](../../var/agent_runs/heat-burgers-architecture-screen/analysis/architecture-screen-comparison.html), [notebook reproducible](../../notebooks/heat/EXP-0005-architecture-screen/heat-burgers-architecture-screen-comparison.ipynb), [tabla por semilla](../../var/agent_runs/heat-burgers-architecture-screen/analysis/seed-results.csv) y [señales cuantificadas](../../var/agent_runs/heat-burgers-architecture-screen/analysis/quantified-signals.csv).

## Decisión central

La exploración **no** es una corrida masiva de todos los modelos en todos los problemas. EXP-0002 produjo el primer control instrumental: **RBF-PIKAN de centros y anchos fijos**, con Sobol scrambled, en Heat y un Burgers estacionario. EXP-0005 conserva Heat y vuelve a entrenar las tres arquitecturas en el Burgers transitorio canónico de Raissi/PINNacle, porque ambos contratos de Burgers son distintos. Esta comparación mide señales y diferencias arquitectónicas; no atribuye mejoras al muestreo ni establece SOTA.

El estudio tendrá dos tipos de baseline:

1. **Baseline físico e instrumental por problema:** una RBF-PIKAN fija con protocolo reproducible, para mapear errores, puntos, derivadas, bases, gradientes y costo. Heat es un control técnico; los problemas objetivo inicialmente acordados son Poisson, Burgers, Allen–Cahn y DFG Re=20. Poisson sigue pendiente y puede salir del claim final solo tras registrar un cambio de alcance.
2. **Baselines comparativos por arquitectura:** PINN-MLP, PIKAN B-spline y RBF-PIKAN fija bajo la misma PDE y presupuesto. Un método PINN fuerte, como PirateNet, se añade a la confirmación donde la implementación y el presupuesto permitan una comparación verificable. Free-RBF-KAN y el muestreo adaptativo se consideran **candidatos/intervenciones**, no controles intercambiables con la RBF fija.

Separar estas funciones importa: los trabajos de comparación MLP/KAN muestran que costo, precisión y estabilidad cambian entre variantes y semillas ([Shukla et al.](https://arxiv.org/abs/2406.02917)); una RBF con bases aprendibles tiene un mecanismo adicional respecto a nuestra geometría fija ([Free-RBF-KAN](https://arxiv.org/abs/2601.07760)).

## Secuencia y decisiones de avance

| Fase | Ejecución | Salida verificable | Decisión para avanzar |
|---|---|---|---|
| 0. Contrato y referencias | Heat: conducción 1D en varilla con referencia exacta. Burgers: benchmark transitorio Raissi/PINNacle con referencia Cole–Hopf independiente. No cambiar `evals/` tras mirar candidatos. | Fichas físicas, prueba de referencia y mallas versionadas. | El error de solución puede medirse sin sustituirlo por el residual de entrenamiento. |
| 1. Exploración de señales | EXP-0002 se usa para señales del Burgers estacionario histórico; EXP-0005 vuelve a entrenar RBF fija con el contrato canónico junto a MLP PINN y B-spline PIKAN. | Tablas cuantitativas por semilla, curvas, campos/error, puntos exactos, tiempos, versiones y W&B. | Se mide repetibilidad de señales y costo; no se declara superioridad. |
| 2. Controles de arquitectura | Heat y Burgers canónico con RBF fija, PINN-MLP y PIKAN B-spline; mismos pools, física, referencia y presupuesto por semilla. `3 arquitecturas × 2 PDE × 3 semillas = 18 runs` nuevos. | Error y costo por semilla, parámetros, memoria, curvas error/tiempo y error/pasos. | Se elige qué control y problema pasan a una intervención focalizada. |
| 3. Intervención única | En Burgers, comparar Sobol fijo contra **una** política de focalización (RAD o RAR-D) basada en señales de entrenamiento/diagnóstico. Mantener modelo, referencia y presupuesto constantes. Probar bases RBF aprendibles en otra campaña. | Efecto pareado sobre error global, frente, residual, costo y estabilidad; mapas de puntos antes/después. | Solo una mejora reproducible justifica transferir la política a Allen–Cahn, DFG y eventualmente Poisson. |
| 4. Confirmación | Evaluar finalistas en los problemas objetivo con referencias auditadas, evaluación final bloqueada y al menos cinco semillas como mínimo operativo. Incorporar un comparador PINN fuerte cuando corresponda. | Intervalos de incertidumbre, fallos, métricas físicas, curvas de Pareto y artefactos reproducibles. | El investigador decide el claim permitido por la evidencia. |

La definición física y el evaluador de Burgers canónico están implementados; las referencias y seis combinaciones de smoke CPU pasaron. EXP-0005 completó las 18 combinaciones de tres arquitecturas, dos problemas y tres semillas. Los manifiestos solicitaron dos T4; los resúmenes registran asignaciones a los slots GPU 0 y 1. Los siguientes problemas y cualquier focalización se mantienen abiertos hasta interpretar las señales y preregistrar su hipótesis y umbrales.

## Orden inmediato propuesto

1. **Interpretar el screening completado.** Comparar por problema los errores, variación entre semillas, señales de puntos, residuales, gradientes y costo. Separar mediciones de asociaciones y no atribuir causalidad a una arquitectura o señal con tres semillas.
2. **Cerrar los contratos de confirmación.** Mantener Heat y Burgers con sus referencias auditadas; refinar la malla de evaluación antes de un claim final. Allen–Cahn requiere referencia BDF convergida y DFG una referencia FEM auditada en outlet, presión y fuerzas. La evaluación final permanece reservada.
3. **Preregistrar una intervención.** Elegir una señal medible y una sola política (RAD o RAR-D) contra el control Sobol, fijando pool, presupuesto, métrica primaria, tolerancias físicas, costo máximo y regla de aceptación antes del siguiente run.
4. **Adaptar la siguiente PDE a la evidencia.** Revisar si las señales y limitaciones observadas justifican continuar con Allen–Cahn, DFG o una prueba intermedia. Cada cambio conserva la propuesta original y recibe manifiesto versionado; no se inicia otra corrida hasta que el nuevo protocolo quede definido y autorizado.

## Resultado del screening EXP-0005

Se descargaron y validaron tres bundles —RBF fija, MLP tanh y B-spline cúbica— con tres semillas cada uno. Los 18 runs de W&B aparecen sincronizados. El análisis conserva **2.622 observaciones señal-semilla de 354 métricas escalares** entre Heat y Burgers, además de configuración, parámetros, puntos, tiempo y asignación de GPU. Los pools Sobol scrambled son fijos y pareados; cada arquitectura tiene entre 3.648 y 3.655 parámetros.

La telemetría de GPU se registra a nivel de campaña porque cada seed entrena Heat y luego Burgers; no se atribuye su memoria/utilización a una sola PDE. Los resúmenes por GPU registran asignaciones a slots 0 y 1.

| Métrica de validación independiente | RBF-PIKAN fija | PINN MLP tanh | PIKAN B-spline cúbica |
|---|---:|---:|---:|
| Heat: error relativo L2, media ± SD (n=3) | 8.88e-5 ± 4.54e-5 | 2.52e-4 ± 8.87e-5 | **2.30e-5 ± 9.49e-6** |
| Burgers: error relativo L2, media ± SD (n=3) | **0.0540 ± 0.0393** | 0.1159 ± 0.0240 | 0.2210 ± 0.0510 |
| Burgers: RMSE normalizado del frente, media ± SD (n=3) | **0.292 ± 0.229** | 0.351 ± 0.226 | 0.943 ± 0.115 |

La lectura preliminar es dependiente del problema: B-spline tuvo menor error de campo en este Heat y RBF en estas métricas de Burgers. La dispersión entre semillas es visible, sobre todo para el frente. Esto sirve para diseñar el siguiente estudio; no demuestra superioridad general, beneficio de una señal ni estado del arte. El evaluador marcó `thresholds_pass=false` porque el screening no tenía umbrales preregistrados; los runs y artefactos sí son válidos.

La literatura de muestreo estudia Sobol, LHS, aleatorio y políticas RAD/RAR-D bajo distintos PINNs; ninguna gana automáticamente en nuestros casos ([Wu et al.](https://arxiv.org/abs/2207.10289)). PirateNet sirve como referencia de una arquitectura PINN moderna con entrenamiento profundo más estable, pero sus resultados publicados no son números directamente comparables con nuestros contratos de Burgers/DFG ([Wang et al.](https://arxiv.org/abs/2402.00326)).

## Presupuesto y comparación justa

La restricción es **menos de ocho horas por campaña**, con dos T4 de Kaggle cuando estén disponibles. EXP-0002 midió aproximadamente `192 s` para Heat y `1022 s` para Burgers por semilla RBF, incluyendo diagnósticos. Esos tiempos sirven para dimensionar el siguiente screening; no predicen el costo de MLP, B-spline o DFG. Cada nueva arquitectura comienza con un smoke de memoria/tiempo. Se asigna un proceso por GPU, con semillas independientes.

Para comparar arquitecturas, mantener iguales ecuación, condiciones, puntos de colocación, particiones, normalización física, precisión y criterios de parada. Registrar dos vistas de eficiencia: **error a pasos comparables** y **error a tiempo de GPU comparable**. Un mismo tope de pasos/tiempo (“lo que ocurra primero”) no vuelve iguales ambas cantidades; por eso se conservan curvas y el motivo de parada. Informar cantidad de parámetros y, cuando sea viable, una capacidad MLP cercana a la RBF de 3.648 parámetros. Los intentos de ajuste también consumen presupuesto y se registran.

El screening usa tres semillas como filtro económico. Solo los finalistas pasan al mínimo operativo de cinco semillas, aumentando si la variación impide decidir. Una corrida divergente cuenta como resultado. Los umbrales de error, física y costo se fijan **antes** de abrir la evaluación final; EXP-0002 no tiene tales umbrales y sigue siendo exploratoria.

## Señales y análisis causal

Todas las arquitecturas comparten el núcleo medible: estadísticas de puntos; pérdidas por término; gradientes y parámetros por capa; derivadas de autodiferenciación; residual y error sobre pools independientes; pasos, ejemplos, tiempo y GPU. Se agregan señales propias sin forzar equivalencias falsas: ocupación/entropía de bases en KAN, saturación y sensibilidad en RBF-PIKAN, activaciones y gradientes en MLP. Se estudian tres preguntas por señal:

1. **Repetibilidad:** ¿mantiene rango y localización entre semillas?
2. **Valor predictivo:** ¿anticipa el error en validación independiente, además del residual de entrenamiento?
3. **Utilidad de intervención:** ¿mejora el resultado cuando guía puntos/bases bajo el mismo presupuesto?

La relación entre residual y error es específica del problema; un mapa visual coincidente solo motiva una prueba. Las señales de conflicto de gradientes, rango efectivo y norma de actualización siguen pendientes de instrumentación y no se atribuyen a EXP-0002.

## Salidas visuales y de datos

Las dos imágenes aportadas ilustran el tipo de superficie espacio-temporal deseada. En Heat, Burgers y Allen–Cahn se exportará un panel de cuatro vistas con **referencia, predicción, error absoluto y residual**, usando escalas explícitas. Pérdidas, errores absolutos, gradientes y magnitudes de residual positivos usan eje logarítmico; el color del error absoluto usa `log10(|error|)`. Campos con signo, como `u` de Burgers, se mantienen lineales. Los ceros exactos del error no se mapean a log y se muestran vacíos, sin sumar un epsilon artificial. Las superficies 3D se acompañan de mapas 2D y cortes temporales para que un frente o error estrecho no quede oculto. En Burgers se muestran el frente y la densidad de puntos antes/después de focalizar; en DFG se usan geometría y campos/streamlines PyVista con regiones cerca del cilindro y estela.

Cada notebook conserva estructura de desarrollo: problema y contrato, configuración/manifiesto, generación de puntos, entrenamiento, señales, tablas por semilla, visualizaciones, validación, costo y decisión. Plotly genera curvas y mapas interactivos; PyVista exporta geometría/campos sin ventana; Panel presenta el análisis; W&B registra configuración, métricas y artefactos; CSV/JSON/HTML y notebook quedan descargados localmente. Las tablas dan valores, unidades, tamaño de muestra y definición de cada señal; una gráfica sola no cuenta como medición.

## Matriz por problema

| Problema | Baseline instrumental | Referencia de campo/QoI | Primera señal de interés | Próxima acción |
|---|---|---|---|---|
| Heat 1D | EXP-0002 completado | Analítica; falta refinar evaluación final | Consistencia de derivadas, bases y pérdida | Control barato del screening MLP/B-spline. |
| Burgers 1D | EXP-0002 completado | Exacta para el caso configurado; auditar cuadratura/refinamiento | Frente, residual local, saturación/gradientes | Primer caso de comparación e intervención. |
| Poisson 2D | Pendiente | Definir contrato y referencia | Fronteras, esquinas o regiones de alto residual según dominio | Mantener en alcance; no bloquear el screening actual. |
| Allen–Cahn 1D | Borrador EXP-0003 | BDF convergido antes de campo final | Interfaz, energía, estabilidad temporal | Smoke físico RBF y auditoría BDF. |
| DFG 2D-1 Re=20 | Borrador EXP-0004 | FEM/QoI auditados antes de comparación final | Cilindro, estela, divergencia y paredes | Smoke de geometría/BC y referencia FEM. |

La comparación final distingue solución de una instancia de aprendizaje de operadores; el pipeline actual resuelve instancias concretas. Si se quiere SOTA en operator learning, hará falta otra campaña con familias de condiciones y parámetros, particiones por instancia y comparadores como FNO/DeepONet/PINO ([PDENNEval](https://www.ijcai.org/proceedings/2024/573), [PDEBench](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html)).
