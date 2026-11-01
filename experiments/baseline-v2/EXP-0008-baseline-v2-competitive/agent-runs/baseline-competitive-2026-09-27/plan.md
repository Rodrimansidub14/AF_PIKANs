# Plan ejecutable para el baseline competitivo (2026-09-27)

> Snapshot histórico del 2026-09-27. El estado ejecutado y las métricas vigentes están en competitive-baseline-execution.md y competitive-baseline-completion-plan.md.
## Estado actual (2026-09-28)

Se completaron en Kaggle GPU comparadores de cinco semillas para Heat, Burgers, Allen–Cahn y DFG. La réplica PINNacle Adam→L-BFGS de Burgers dio 0.01326422 ± 0.00002883 en la referencia FEM oficial, coincidente con 0.0133 ± 0.000053 publicado. La comparación Heat pareada sigue pendiente, y Allen–Cahn/DFG no cierran aún. El mecanismo continúa diferido. La cuota observada es 11:25/30 h usadas, con 18:35 h restantes; el piso solicitado de 12 h deja hasta 6:35 h para nuevas sesiones. No se ejecutaron notebooks ni validaciones locales.

**Alcance vigente:** cuatro problemas: Heat 1D, Burgers transitorio canónico, Allen–Cahn 1D y DFG 2D-1 Re=20 (Navier–Stokes estacionario). Heat sustituye a Poisson en esta campaña por la instrucción actual del investigador. Este documento no congela baselines, modifica `evals/` ni acepta claims. Estado inicial: resultados exploratorios disponibles, `scientific_accepted=false`.

## Qué significa «competitivo»

1. **Reproducción comparable:** mismo problema (PDE, dominio, unidades, IC/BC, parámetros), misma definición de error y referencia, precisión y protocolo de evaluación declarados. Un número publicado para otro contrato es contexto, no umbral de SOTA.
2. **Comparador fuerte realmente ejecutado:** además de RBF fija + Sobol scrambled, reproducir por problema al menos una arquitectura/estrategia fuerte de la literatura en el contrato común. Registrar todos los ensayos, fallos y ajuste de hiperparámetros; no escoger el mejor seed.
3. **Dos vistas de costo:** error frente a pasos y frente a tiempo efectivo de GPU, con hardware, memoria, parámetros, puntos, optimizador y precision registrados. Una corrida con el mismo tope de pasos pero distinto tiempo no es isocosto.
4. **Confirmación:** mínimo cinco semillas independientes y pareadas para los finalistas, reportando todos los runs, media, dispersión e intervalo de incertidumbre. La declaración «comparable al SOTA» exige alcanzar la banda publicada bajo un contrato igualado, sin violar los umbrales físicos preregistrados; «supera SOTA» requeriría una comparación directa todavía más estricta. Si la literatura no ofrece la misma instancia y métrica, se informa una comparación contextual y no se emite ese claim.

El universo del eventual claim debe quedar explícito: **solvers PINN/PIKAN por instancia**, no superioridad general frente a FEM/espectral ni frente a modelos operadores preentrenados con datos. En DFG, FEATFLOW es verdad numérica de QoIs y FEM será referencia/costo fuerte; la ausencia actual de un PINN publicado para este exacto caso impide un ranking «SOTA PINN» directo hasta reproducir un comparador bajo el mismo contrato.

## Punto de partida medido

| Problema | Evidencia existente | Impedimento para cerrar |
|---|---|---|
| Heat | RBF EXP-0005 n=3: `8.872368e-5 ± 4.530022e-5`; extensión n=2 separada: `3.626496e-4 ± 3.644286e-4`. La auditoría estática de ramas Heat permite el agregado **descriptivo** n=5: `1.98294e-4 ± 2.38199e-4` (no confirmatorio). B-spline n=3: `2.30142e-5 ± 9.52085e-6`, ~301 s frente a ~191 s RBF. Referencia analítica. | Gran dispersión RBF y B-spline más preciso a mayor costo; falta comparador publicado de la misma instancia/evaluador y confirmación pareada. |
| Burgers canónico | Sobol RBF n=5: `0.0395498 ± 0.02564915`; RAD de la configuración ensayada empeoró los cinco pares. Referencia Cole–Hopf. | Error alto y variabilidad; falta un método fuerte reproducido bajo el contrato y presupuesto comunes. |
| Allen–Cahn | RBF n=5: L2 de campo `0.01843233 ± 0.003375772` en malla DeepXDE 201×101, aún provisional. Referencia candidata BDF AC-L4 converge en espacio y tiempo según el protocolo. | La candidata requiere revisión externa; falta comparador fuerte ejecutado en GPU. |
| DFG Re=20 | RBF n=5: `Cd=3.967438 ± 0.179625`, `Cl=-0.026026 ± 0.124769` (corregido 2026-09-30 con σ·n; antes Cd 1.075206, Cl −0.046521 con fórmula tangencial errónea), `Δp=0.077682 ± 0.005540`; FEATFLOW `5.579535`, `0.010619`, `0.117520`. | Cd 28.9% y Δp 33.9% de error antes de L-BFGS (Cd 80.7% era el valor erróneo); sin campo FEM independiente. r4 completó 303–305, pero Kaggle marcó error durante el cierre de visualización posterior al entrenamiento. |

Los datos y su procedencia están en el [informe entregado](../baseline-close-2026-09-27/delivery/baselines-report.md). Los valores entre semillas son descriptivos; no son umbrales preregistrados ni comparaciones SOTA.

## Anclas de literatura seleccionadas

| Problema | Control publicable y acción | Resultado publicado que **no** se traslada como score nuestro |
|---|---|---|
| Heat | Adaptar el [PINN tanh del tutorial DeepXDE](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html) a `α=.1`, junto al B-spline local; evaluarlos en la misma solución analítica y malla 512×256. | [HyPINO](https://arxiv.org/html/2509.05117) usa el Heat `α=.1`, pero su MSE y preentrenamiento supervisado de 77 M parámetros/4×4090 hacen impropio comparar su número con L2 de un solver por instancia. |
| Burgers | Reproducir [Vanilla PINN de PINNacle](https://arxiv.org/html/2306.08827) en el contrato y evaluador AF (paper: L2RE `1.45e-2 ± 1.59e-3`, n=3; L-BFGS `1.33e-2 ± 5.30e-5`). | [vRBA](https://www.nature.com/articles/s44387-026-00084-4) informa ~`8e-9` con 5k Adam +60k SSBroyden y muestreo adaptativo; falta resolver la descripción contradictoria del borde y equiparar costo/evaluación. Sirve como techo de alta precisión contextual. |
| Allen–Cahn | Reproducir [DeepXDE AC](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html): misma `d=.001`, IC/BC duras, 8k interiores, 40k Adam + L-BFGS; [gPINN+RAR](https://arxiv.org/abs/2111.02801) y [LearnQuad](https://papers.neurips.cc/paper_files/paper/2025/file/382b95a6580cfb0b5ca33c74b4e0e770-Paper-Conference.pdf) son comparadores de método posteriores bajo evaluador común. | [AEH-PINN](https://arxiv.org/html/2512.17607) reporta `8.12e-5` en el mismo PDE, pero con hasta 100k épocas y otra referencia/protocolo; la cifra no es umbral directamente comparable. PirateNets usa otra difusión/BC. |
| DFG | Referencia oficial [FEATFLOW](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html) y FEM Taylor–Hood de [FEniCS](https://fenics-handson.readthedocs.io/en/latest/navierstokes/doc.html)/[jNO](https://fhg-iisb.github.io/jNO/tutorials/08-fem-and-varpinns/navier-stokes-cylinder-dfg/) como punto de partida para campo independiente y comparador fuerte. | No se identificó en esta revisión un score PINN publicado con **ese** DFG estacionario Re20 y sus tres QoIs; Re100 inverso/estela no es ancla SOTA válida. La comparación inicial será contra FEM/FEATFLOW y PINN reproducidos aquí. |

Las fichas completas con caveats están en [Heat/Burgers](literature-heat-burgers.md) y [Allen–Cahn/DFG](literature-ac-dfg.md). Los métodos adaptativos publicados son **comparadores externos**, no una intervención incorporada a RBF+Sobol; su costo de búsqueda y muestreo cuenta en el presupuesto.

## Secuencia con decisiones de avance

### 0. Matriz de literatura y contratos (antes de gastar GPU)

- Para cada fuente primaria, registrar PDE exacta, IC/BC, región y cuadratura del error, referencia, semillas, optimizador, número de puntos, pasos y costo. Separar **directamente comparable**, **reproducible en nuestro contrato** y **contexto**.
- Fijar una métrica primaria por problema antes de abrir nuevas corridas: Heat/Burgers/Allen–Cahn L2 relativo de campo sobre referencia independiente; DFG vector de QoIs con criterio conjunto (Cd, error absoluto de Cl, Δp) y error de campo FEM cuando exista. El residual y las regiones difíciles son secundarios, nunca sustitutos de solución.
- Seleccionar los comparadores por evidencia y factibilidad: PINN fuerte moderno donde sea reproducible; B-spline local en Heat; método de tratamiento de BC/interfaz para Allen–Cahn; método de Navier–Stokes DFG con fuerza/flujo bien definidos. Registrar por qué cada uno es pertinente a nuestro contrato.
- **Salida:** matriz trazable y lista de diferencias que impiden comparación directa; ningún «SOTA» sin celda comparable.

### 1. Referencias y definición física

- **Heat:** mantener solución analítica. La [auditoría estática](heat-cohort-equivalence.md) encontró equivalentes las ramas Heat de arquitectura, entrenamiento, muestreo y score de EXP-0005 y la extensión; presentar sus cinco semillas como agregado descriptivo con procedencia doble, nunca como confirmación de un único bundle. Congelar una sola malla de puntuación y una malla refinada independiente para estimar error de cuadratura.
- **Burgers:** conservar solo la instancia transitoria `ν=0.01/π` y referencia Cole–Hopf correspondiente. No mezclar con Burgers estacionario histórico. Medir L2 global y error localizado en el frente; preservar condiciones y normalización de la referencia publicada.
- **Allen–Cahn:** el candidato BDF nuevo pasa dos pares espaciales consecutivos y el chequeo temporal del protocolo. Mantenerlo provisional hasta revisión de procedencia y cotejarlo en puntos coincidentes con la malla publicada. Puntuar interfaz (posición y anchura), energía y residual además de L2 de campo; no desbloquear el evaluador protegido por interpretación de un agente.
- **DFG:** reconciliar geometría, inlet, no-deslizamiento, outlet, presión de referencia, definición y signo de Cd/Cl, y puntos de Δp con la especificación oficial. Construir o importar una solución FEM independiente con refinamiento de malla y fuerzas/Δp convergidos; evaluar balance de masa y regiones de cilindro/estela/paredes. El QoI FEATFLOW es una referencia, no un campo FEM descargable por sí solo.
- **Salida:** ficha y artefacto de referencia por PDE con fuente, malla, incertidumbre numérica, hashes y causas de cualquier bloqueo. Si AC/DFG no cierran, no gastar confirmación GPU en ellos.

### 2. Diagnóstico de fallos actuales (sin diseñar aún el mecanismo)

- Heat: comparar por semilla error/costo y ubicación del error; distinguir saturación de representación de optimización.
- Burgers: inspeccionar pérdidas separadas, error del frente y gradientes por término/tiempo; determinar si el cuello está en BC/IC, rigidez o entrenamiento. RAD queda como resultado negativo de **esa configuración**.
- Allen–Cahn: cortes temporales e interfaz contra referencia confiable, energía y evolución de pérdidas; no inferir causalidad de los mapas.
- DFG: comprobar por separado fuerzas por presión/cizalla, continuidad global y cerca del cilindro, residuos x/y, no-deslizamiento, caudal inlet/outlet y dependencia del gauge de presión. Antes de optimizar modelo, descartar discrepancia de escala/QoI/outlet.
- **Salida:** tabla de hipótesis falsables y evidencia por región/semilla. Cualquier modificación de PDE o modelo central necesita gate `code_surface`.

### 3. Screening competitivo, una variable a la vez

- Elegir tratamientos respaldados por la literatura **que preserven el contrato**: imposición exacta de IC/BC cuando sea compatible, balance dinámico de pérdidas/gradientes, Adam seguido de L-BFGS, y una arquitectura fuerte con presupuesto igualado. No introducir RAD ni señales de focalización como ventaja encubierta del baseline.
- El entrenador existente `src/research_pipeline/sciml/train.py` instancia Adam directamente y no expone L-BFGS. Reproducir DeepXDE/PINNacle con L-BFGS requiere un runner/adaptador versionado; no etiquetar un run Adam-only como réplica de la receta publicada. Cambiar el entrenador central dispara el gate `code_surface`; una adaptación de notebook/bundle en la superficie permitida deberá revisar independencia del score final.
- En cada PDE, conservar RBF fija + Sobol como control y ejecutar hasta tres semillas de screening por candidato, con mismo pool de puntos, precisión, hardware y presupuesto efectivo por comparación. Se registra el costo de búsqueda. El orden de candidaturas será determinado por la matriz de literatura y el diagnóstico, no por selección posterior sobre la malla final.
- Selección sobre split de desarrollo independiente del score final. Adelantar solo variantes que mejoren error de campo/QoI sin empeorar tolerancias físicas; si ninguna lo logra, registrar fallo y revisar capacidad/optimización antes de ampliar cómputo.
- **Salida:** manifiestos y bundles con hashes, presupuesto por semilla, stop rule, selección predefinida y scorecard de screening. Cada campaña remota tendrá menos de 8 h de pared y máximo dos T4 simultáneas; para Tier 2 se respetará además el tope de 4 GPU-h declarado por la suite existente. Costos reales prevalecen sobre nominales. DFG necesita una aprobación específica Tier 3: la suite Tier 3 genérica vigente es `ns_cavity_2d`, no DFG.

### 4. Confirmación y comparación SOTA

- Antes de la primera confirmación, firmar pre-registro de hipótesis, semillas, presupuesto, métrica primaria, tolerancia física y falsación; conservar `evals/` de solo lectura.
- Ejecutar ≥5 semillas independientes y pareadas por finalista/control/comparador fuerte en cada problema, incluyendo divergencias. Comparar distribución del error y curvas error–GPU-s; reportar memoria y parámetros. Repetir o ampliar solo si la regla preregistrada de incertidumbre no decide.
- Hacer dos informes separados: (a) reproducción de la literatura en su contrato cuando sea posible; (b) tabla isocontrato en nuestros cuatro problemas. Un número de papel de otra geometría, normalización o referencia se muestra como contexto.
- **Salida:** cuaderno ejecutado, CSV por semilla, gráficas, artefactos y scorecard con `comparable_to_published` explícito. `scientific_accepted` y congelación del baseline quedan para decisión humana.

### 5. Evidencia diagnóstica útil para el mecanismo

- En pools held-out fijados, analizar si ocupación/sensibilidad RBF, conflicto de gradientes u otras señales predicen error de campo/region **después de controlar por el residual**, por problema y entre semillas. El error de referencia se usa solo para evaluar la señal; no entra en la política de puntos.
- Reportar estabilidad, calibración y ganancia incremental sobre el residual. Las celdas espaciales no son réplicas; la unidad independiente es la semilla/corrida.
- **Salida:** shortlist de señales con evidencia de transferencia o de fallo entre PDEs. El diseño de la intervención se inicia después de decidir el baseline competitivo, en un pre-registro separado.

## Reglas de ejecución y límites

- **Presupuesto vigente:** la instrucción del investigador fija un suelo de 12 h de cuota semanal restante. La última lectura de Kaggle fue 10:51/30 h usadas (19:09 restantes), por lo que el gasto adicional máximo de esta campaña es 7:09 h, incluido setup. Volver a leer la cuota antes de cada envío; no consumir el margen si el trabajo no completa una comparación útil.
- Orden crítico: **matriz → referencias/física → diagnóstico → screening → confirmación → señal**. Heat/Burgers pueden avanzar al screening mientras AC/DFG esperan referencia; ningún problema bloquea trabajo independiente de otro.
- El pedido de saltar validaciones locales sigue vigente: no correr tests, preflights ni validadores locales; el análisis de artefactos ya producidos y la ejecución de notebooks de resultados sí forman parte del trabajo científico. Registrar esta limitación en cualquier scorecard.
- La campaña actual quedó autorizada para GPU bajo el suelo de 12 h semanales restantes. Esta autorización cubrió los comparadores ejecutados en Kaggle y no elimina el gate científico de DFG: referencia de campo, outlet y balance deben seguir documentados antes de aceptar el caso.
- El transporte remoto exige `approval.json` ligado al SHA-256 del bundle preparado (`src/research_pipeline/remote/README.md`); las aprobaciones de EXP-0002/0003/0006 y DFG r3/r4 no autorizan bundles nuevos. La suite Tier 2 está declarada pero aún no implementada y no hay baselines congelados en `evals/baselines/`; no inferir que un score antiguo satisface un gate inexistente.
- Los resultados exploratorios previos permanecen disponibles; ninguna referencia o umbral en `evals/` se edita por el proposer. Si falta referencia, el campo queda `null` con razón.

## Historial de preparación del plan (no representa el estado actual)

- [x] Inspección de política, planes previos y resultados ya entregados; reconciliado el alcance actual de cuatro problemas.
- [x] Agentes GPT-6 Luna con esfuerzo máximo encargados de literatura Heat/Burgers, literatura Allen–Cahn/DFG y auditoría del repo; sus hallazgos se incorporarán a la matriz y al plan.
- [x] Matriz primaria con citas y comparabilidad exacta para los cuatro problemas; se halló ausencia de un score PINN DFG 2D-1/Re20 directamente emparejable.
- [x] Análisis numérico descriptivo de corridas archivadas y auditoría documental de referencia AC/DFG, sin validar localmente ni reentrenar.
- [x] Protocolos de piloto Heat/Burgers y referencias AC/DFG con costos/semillas concretos, aún sin firma ni corrida nueva.
- [x] Generador BDF Allen–Cahn ejecutado como referencia candidata; no alteró `evals/` y el manifiesto mantiene `evaluation_ready=false`.
- [x] Auditoría semántica de cohortes Heat completada sin tests; agregado descriptivo n=5 calculado.
- [x] Adaptador Heat/DeepXDE de un seed preparado como `DRAFT_UNVERIFIED`; sin ejecución ni score.
- [x] Adaptador Burgers/PINNacle Vanilla 5×100 preparado como `DRAFT_UNVERIFIED`; aún falta confirmar pool RBF completo y correr bajo la versión SciPy archivada.
- [x] Andamiaje Gmsh/Taylor–Hood DFG preparado como `DRAFT_UNVERIFIED`; faltan solve, convergencia y trazas de presión sobre el cilindro.
- [x] Protocolo de estudio de señales y fuga de referencia escrito; no es todavía un mecanismo.
- [ ] Nuevos entrenamientos, condicionados al pre-registro y presupuesto exactos.

## Bloqueos encontrados al preparar los comparadores

- `remote.prepare` solo empaqueta archivos `.py` explícitos de `src/` o `evals/` y ejecuta comprobaciones locales de manifiesto e imports. Los adaptadores propuestos viven bajo `var/agent_runs`, por lo que todavía no existe un bundle remoto válido para Heat o Burgers. Promoverlos a `src/` o cambiar el empaquetador exige revisar la superficie de código y el contrato de la campaña; no se presentarán los borradores como corridas listas.
- Para Heat, el RBF/B-spline existente impone IC/Dirichlet de forma exacta y optimiza un residual PDE normalizado; el adaptador DeepXDE propuesto usa condiciones suaves y otra escala de pérdida. Se registrará como diferencia de método, manteniendo idéntica instancia física, malla de score y presupuesto; además faltan checkpoints por tiempo efectivo de GPU para las curvas isocosto.
- Se localizaron archivos completos `pools.npz` de Burgers RBF para semillas 201–205 en la campaña RAD archivada. El borrador del comparador regenera los puntos con el sampler común y SciPy 1.15.3; falta cotejar hashes/contenido antes de afirmar paridad exacta. El smoke contiene muchos menos puntos y no sirve para ello.
- El solver FEM DFG y los adaptadores siguen sin ejecución. El equipo local no expone CUDA (`torch.cuda.is_available=false`), pero el token de Kaggle está presente en `.env`; la consulta al CLI confirmó 26.95 GPU-h disponibles. Las corridas existentes de Heat/AC, Burgers y DFG se ejecutaron en Tesla T4. La referencia BDF auxiliar duró 12.3 s de CPU antes de la aclaración de usar GPU; es solo una candidata de referencia y no evidencia de rendimiento del modelo.
- Los [borradores Heat](bundles/heat/README.md) y [Burgers](bundles/burgers/README.md) contienen hipótesis, manifests y topes de 14,400 y 6,600 GPU-s, respectivamente. Son propuestas para revisión: faltan cierre técnico, empaquetado oficial y firma humana ligada al hash final.
- Descarga de Kaggle CLI del 2026-09-27: Heat/AC r3 y Burgers v4 están `COMPLETE`; DFG r4 está `ERROR` al cerrar, aunque el manifiesto y `metrics.json` confirman las tres semillas completadas con 1,200 s por semilla. El [readout actualizado](baseline-readout-20260927.md) consolida métricas y comparabilidad.
- Los notebooks `heat-results`, `burgers-results`, `allen-cahn-results`, `dfg-results` y `baselines-results` ya se ejecutaron sobre artefactos archivados; son reportes, no entrenamientos nuevos. Véase `../baseline-close-2026-09-27/delivery/notebook-execution.json`.


