# EXP-0013 — resultados (2026-10-05)

**Análisis posteriores no pre-registrados** (declarados como tales): `robustness.py` (centrado dentro de condición) y las reglas de modo de falla; el resto sigue el spec firmado.

Estudio exploratorio, solo CPU, sobre checkpoints ya guardados. Pre-registro firmado el 2026-10-05; el criterio de señal importante y la regla de falsación no se movieron. `scientific_accepted=false`. Nada de esto es un claim.

## 1. Qué se calculó

| Elemento | Resultado |
|---|---|
| Checkpoints analizados | **248** (238 cargados con clases de modelo + 10 de piezo reconstruidos con el código upstream de Daniel en solo lectura) en 6 problemas |
| Señales sin referencia por checkpoint | **272 columnas distintas**, 248/248 corridas sin error; datos ausentes registrados con su razón (`results/nulls.csv`) |
| Etiquetas (error contra referencia) | 248/248, 85 columnas; reproducen los baselines congelados (p. ej. AC 0.0027/0.0177; Burgers grande 7.2e-5/1.05e-4; DFG larga 0.191/0.421; piezo 0.092/0.136) |
| Fichas por corrida | 248 (`cards/`) |
| Matriz problema × señal | 868 celdas (`results/analysis/matrix_signal_problem.csv`, figura `figures/matrix_signal_problem.png`) |
| Costo (criterio d) | toda la batería < 10 % de un paso de entrenamiento si se mide cada ≥ 100 pasos (≈ 2–5 s por medición; `analysis/cost_bench.csv`) |
| Exclusión declarada | 5 filas: la etapa L-BFGS de las réplicas del control de EXP-0011 duplica a la de Adam (se corrieron con 0 s de L-BFGS) |

Registro completo: `results/run-log.jsonl` (cada cálculo con hora, entorno y hashes del código), `results/signals/<problema>/*.json`, `results/labels/`, `results/inventory.csv` (hashes de checkpoints). Los campos por punto (error y residuo) están en `var/agent_runs/exp0013-fields/` (fuera de git).

## 2. Veredicto frente al criterio pre-registrado (H-DIAG-1)

Criterio: señal sin referencia con misma dirección en ≥ 2 problemas (|ρ_w| ≥ 0.5, p_Holm < 0.05, permutación por bloques de semilla), estabilidad entre semillas τ ≥ 0.5 y costo < 10 %.

**Aplicando la regla tal cual, H-DIAG-1 no se rechaza: 9 señales cumplen (b) y 7 cumplen (b)+(c)+(d).** Pero leerlas como "hay señales útiles" sería engañoso por tres razones que se declaran antes de interpretar:

1. **3 de las 7 son variables de presupuesto del diseño, no señales del modelo** (`train/adam_steps`, `train/adam_n_logged`, `train/lbfgs_seconds`): son constantes dentro de una campaña, y las campañas tienen presupuestos distintos (p. ej. Burgers 20 000 vs 24 000 pasos), así que "predicen" el error porque identifican la campaña. El spec no previó esta confusión; **las clasifico como confusores y las descarto como candidatas** (decisión mía, a revisar).
2. **Las 4 señales restantes son estadísticas de magnitud/forma del residuo tras L-BFGS** (`res/all/rms`, `p50`, `p95`, `entropy_norm`), y no se sostienen de forma uniforme: `entropy_norm` tiene **signo opuesto** en Burgers (ρ = −0.78, significativa) frente a Heat y DFG (+0.84, +0.90); y en la comprobación de robustez no pre-registrada (centrando y permutando **dentro de cada condición**, es decir solo entre semillas; `analysis/robustness_within_condition.csv`) **tras L-BFGS casi ninguna se sostiene**: `rms`, `p50` y `p95` en Burgers (ρ = 0.35, 0.14, 0.22; p = 0.12–0.41), Heat (0.35, 0.53, 0.29; p = 0.07–0.35) y DFG (0.06, 0.28, 0.56; solo el p95 tiene p = 0.044 sin corrección, Holm 0.22). La única excepción es `entropy_norm` en DFG (ρ_cond = 0.39, Holm 0.025, por debajo del umbral de 0.5), mientras que en Burgers y Heat su signo es negativo y no significativo. Es decir, el residuo tras L-BFGS **separa condiciones buenas de malas (brazos, campañas), pero no ordena el error entre semillas de una misma condición**.
3. **En la etapa Adam sí hay señal entre semillas, pero solo en dos problemas:** el RMS del residuo y su p95 ordenan el error en Burgers (ρ_cond = 0.65 y 0.52; Holm 0.006 y 0.025) y en Heat (0.90 y 0.92; Holm 0.001); **en NS2d-CG la relación es negativa** (−0.78 y −0.62) y en DFG no existe (0.14 y 0.30, no significativa). Por eso fallan (c) (estabilidad de rango τ ≈ 0.06) y no llegan a candidatas.

**Lectura resumida:** el residuo funciona como *indicador de cuánto error queda* (estimador a posteriori) en problemas con frente o decaimiento suave (Burgers, Heat) durante Adam, y como discriminador entre condiciones tras L-BFGS. **No se identificó una señal sin referencia que diga *dónde* está el error de forma consistente entre problemas.** Eso es el resultado central, y es honestamente modesto.

Validación dejando una semilla fuera (`analysis/loso_direction.csv`): la dirección de las señales residuales se mantiene en todos los pliegues en las celdas donde cumplen (b), salvo `res/all/rms` en DFG (6/15 pliegues, ambas etapas) y en NS2d-CG (3/5), y `p95` en DFG-Adam (6/15). **No se activó el bloque de confirmación en GPU** (la regla del spec lo exige solo si una señal que cumple (a)–(d) no sostiene su dirección en LOSO; las candidatas que se mantienen sí la sostienen).

## 3. Respuestas a las preguntas diagnósticas (medianas por semilla; n = 5 por celda)

**Q1 — ¿Dónde está el error y dónde el residuo?** (`analysis/q1_residual_vs_error.csv`; solape esperado por azar del 5 % superior: 0.05)

| Problema | Spearman residuo–error | Solape del 5 % superior | Lectura |
|---|---|---|---|
| DFG | 0.30–0.45 | **0.47–0.59** | el residuo apunta al error; ambos se concentran 6–14× en la región de mayor gradiente |
| Burgers | 0.30–0.47 (Adam); **0.05–0.07 tras L-BFGS** | 0.16–0.39 | el residuo señala la *zona* del frente (concentración del residuo 15–19×, del error 10–19×) pero no el punto exacto, y la alineación se pierde tras L-BFGS |
| Allen–Cahn | 0.17–0.46 | 0.00–0.17 | alineación baja; el error no está en la región de mayor gradiente (concentración 0.01–0.35) |
| Heat | −0.05 a 0.31 | 0.02–0.16 | residuo no informa el error (problema suave) |
| NS2d-CG | −0.28 a 0.17 | 0.15–0.19 | residuo no informa el error |
| Piezo | **−0.57 (MLP), −0.40 (RBF)** | 0.00 | **anticorrelacionado**: el residuo es mayor donde el error es menor |

**Q2 — ¿Capacidad u optimización?** La pérdida ordena el error entre corridas en DFG (Spearman 0.82–0.94) y Allen–Cahn (0.87–0.88): ahí el límite lo marca la optimización de la pérdida. En Burgers solo en Adam (0.52); tras L-BFGS la pérdida de Adam ya no ordena el error (−0.01). Para DFG hay además evidencia externa: el RBF de 4 032 parámetros ajustado directamente al FEM llega a 1–5 %, o sea no es un límite de capacidad.

**Q3 — Adam → L-BFGS.** Razón de error L-BFGS/Adam: Burgers 0.008–0.12, Allen–Cahn 0.07–0.09, Heat (control) 0.012; DFG con la receta de la tesis de Daniel 0.92–0.99 (L-BFGS casi no mejora: sin presupuesto) y con receta larga 0.43–0.67. La dispersión entre semillas (CV) **baja** mucho con L-BFGS en Heat (0.82 → 0.10) pero no en Burgers ni Allen–Cahn.

**Q4 — Modos de falla** (reglas fijas en `make_cards.py`, no causales; `analysis/q4_failure_modes.csv`). `null_state`: solo en DFG con MLP (11/20 al final de Adam). `budget_limited` (L-BFGS terminó por tope): Burgers 10/15–20, DFG 5/20–25. `residual_misaligned`: piezo 10/10, Heat 4–5 por condición. `concentrated_error`: Burgers y DFG. La etiqueta `low_jacobian_rank_per_param` (umbral no calibrado) se activa casi siempre y **no se usa como evidencia**.

**Q5 — Estructura frente a bases (capa 0, dimensión x).** Razón (anchura de estructura del propio modelo)/(espaciado de bases): **Burgers 0.066 (15× más fina que las bases)**, Allen–Cahn 0.24 (4× más fina), DFG 1.5→0.51, Heat 6.1, NS2d-CG 4.1, piezo 9.4. Es consistente con tu hipótesis de bases en los problemas con frente (Burgers, Allen–Cahn), y es consistente con la limitación de RA-PIKAN (la extensión de grilla dirigida no resuelve choques más finos que la grilla). **Es un indicio, no una prueba:** las capas ocultas pueden afinar la estructura, y la razón es constante dentro de cada problema (mismo espaciado), así que no se puede probar su efecto dentro del problema con estos datos.

**Q6 — ¿Anticipa Adam cuánto ganará L-BFGS?** Descriptivo (sin corrección): solo DFG muestra relación fuerte (Spearman −0.74 con el Gini y −0.78 con la masa del 5 % superior del residuo, n = 45; más concentración al final de Adam → mayor ganancia de L-BFGS). En Burgers y Allen–Cahn las correlaciones son débiles (|ρ| ≤ 0.44, n = 10–35).

## 4. ML básico con validación dejando una semilla fuera (`analysis/loso_ml.csv`)

Predicción de log10(error) con señales sin referencia (sin variables de presupuesto), frente a una línea base que solo conoce la condición (media de la condición):

| Problema / etapa | R² con señales (bosque) | R² solo condición | ¿Aportan las señales? |
|---|---|---|---|
| Burgers Adam / L-BFGS | 0.31 / 0.84 | −0.12 / 0.36 | **sí** |
| Heat Adam | 0.83 | 0.23 | **sí** |
| Heat L-BFGS | 0.68 | 0.96 | no (la condición lo explica casi todo) |
| DFG Adam / L-BFGS | 0.94 / 0.94 | 0.80 / 0.82 | sí, modesto |
| NS2d-CG Adam | 0.65 | 0.77 | no |
| Allen–Cahn, piezo | n = 10 | | no concluyente |

Importancia por permutación (`analysis/loso_ml_importance_top8.csv`): en Burgers dominan señales de **estructura** de la solución del modelo (|Δu| máx., antisimetría, p95 de |∇u|, anchura mínima del frente) y no el residuo; en DFG, el **balance de caudal** (`dfg/flux_*`); en Heat, la alineación residuo–gradiente y el p50 del residuo.

## 5. Qué sugiere para el mecanismo (interpretación; no es un resultado)

Ordenadas por cuántas preguntas respaldan y con qué solidez:

1. **Awareness de *cuándo* / cuánto queda (presupuesto y cambio de fase).** Respaldo: el residuo como indicador de error en Adam (Burgers, Heat), la ganancia de L-BFGS anticipable en DFG (Q6), y el patrón de EXP-0011/0012 (una etapa mejora, la otra empeora). Límite: falla en NS2d-CG y no se sostiene entre semillas tras L-BFGS.
2. **Awareness de *dónde* basado en la estructura de la solución, no solo en el residuo.** El residuo alinea con el error en DFG y por zona en Burgers, pero no en Heat, Allen–Cahn, NS2d-CG ni piezo (anticorrelacionado). Las señales de **estructura** (anchura mínima del frente, |Δu|, |∇u|) dominan la importancia en Burgers. Esto coincide con la limitación de los métodos guiados solo por residuo (RAD en EXP-0011/0012; RA-PIKAN en sub-grilla).
3. **Awareness de bases (dónde poner resolución).** Respaldo indirecto: estructura 4–15× más fina que las bases de la capa 0 en Allen–Cahn y Burgers. No hay prueba causal en estos datos; probarlo requiere un experimento nuevo (variar espaciado/ancho de bases) con su pre-registro.
4. **Balance entre términos físicos** (DFG: caudal; piezo: tensiones constitutivas vs equilibrio). Señales específicas de problema, sin evidencia transversal; no se pudo medir conflicto de gradientes en problemas con IC/BC duras (un solo término).

Ningún antecedente externo se usa aquí para afirmar novedad.

## 6. Límites y amenazas (declaradas)

- **n = 5 semillas por celda**; los intervalos son anchos y el p mínimo por bloques es 2e-4 (parada temprana a partir de 250 excesos; semilla fija 20261005).
- **Confusión por condición:** el análisis pre-registrado mezcla campañas y brazos dentro de un problema; la comprobación dentro de condición (post-hoc, no pre-registrada) la corrige a costa de potencia y no es evaluable donde hay < 3 condiciones (Allen–Cahn, piezo).
- **Variables de presupuesto** tratadas como confusores por decisión mía.
- **Reglas de modo de falla** (umbrales fijados en código antes de ver resultados, no calibrados); `jac/cond` sin significado numérico y no se usa.
- **Referencias de distinta calidad:** el FEM de DFG es propio (aceptado como referencia de trabajo) y el de piezo es del mismo grupo (sin contraste de terceros).
- **Señales ya vistas en una primera pasada** (caudal DFG, bases muertas, masa del frente) son exploratorias respecto a ese tramo.
- Los checkpoints float32 se evaluaron en float64; la etapa L-BFGS de NS2d-CG y piezo no existe.
- Correlaciones, no causalidad: actuar sobre una señal es lo que probaría un mecanismo.

## 7. Siguiente paso (decisión del investigador)

`decision.md` queda pendiente. Opciones: (A) cerrar el estudio como está y diseñar un EXP-0014 (mecanismo primitivo) sobre Burgers/Allen–Cahn con las familias 1 y 2; (B) añadir primero un EXP corto, de CPU/GPU baja, que pruebe la familia 3 (efecto del espaciado/ancho de bases en Burgers y Allen–Cahn); (C) ampliar semillas (GPU) donde la potencia limitó (Allen–Cahn, piezo, NS2d-CG con L-BFGS). Mi recomendación: A, con B como EXP aparte si el mecanismo toca las bases.

## 8. Reproducir

```
py -3.13 experiments/diagnostics/EXP-0013-run-diagnosis/inventory.py
py -3.13 .../run_signals.py ; py -3.13 .../run_labels.py
py -3.13 .../analyze.py ; py -3.13 .../questions.py ; py -3.13 .../robustness.py ; py -3.13 .../loso.py
py -3.13 .../make_cards.py ; py -3.13 .../catalog.py ; py -3.13 .../cost_bench.py
```
