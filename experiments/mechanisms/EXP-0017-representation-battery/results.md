# EXP-0017 — resultados (2026-10-20)

Exploratorio. `scientific_accepted=false`. Spec firmado el 2026-10-20; los criterios no se movieron. Solo CPU, sobre los 40 pares (estado de Adam, estado tras L-BFGS iniciado en él) de la etapa A del EXP-0016: Burgers 5 semillas × 5 traspasos, Allen–Cahn 3 × 5. Ninguna señal ni la atribución leen la referencia; la referencia solo da la etiqueta |u − u_ref| (y el techo A*, `oracle_ceiling`). Código: `signals.py`, `stats.py`, `run_exp0017.py`, `analyze_exp0017.py`; pruebas en `tests/test_exp0017_signals.py` (15). Datos por estado en `results/states/`, resumen en `results/analysis.json`.

**Verificación previa.** El residuo calculado con este código reproduce al decimal el del EXP-0016 en los 40 estados (diferencia máxima 0). S1 coincide con el gradiente por punto calculado a mano, y S3 con la segunda derivada por autograd (pruebas).

**Desviación respecto del spec.** **S4 no es computable**: la etapa A registró la trayectoria de la capa 0, no la de la capa 1, y el asentamiento exige coeficientes cada 500 pasos. Se reporta como ausencia de medición (no cero) y entra a la corrección de Holm con p = 1, como si hubiera fallado. Todas las señales se evalúan en la misma submalla fija de 8 192 puntos (spec); el residuo de verificación, en la malla completa del EXP-0016.

## 1. Parte 1 — batería de localización (H-LOC-1): no se sostiene para ninguna señal

Mediana sobre los estados (Burgers n = 25, Allen–Cahn n = 15). Azar del solape = 0.05. «p_H» = p unilateral de la diferencia pareada con el residuo, permutación con signo por estado (10 000), corregida por Holm sobre las 20 pruebas.

| Señal | Burgers Spearman | Burgers solape | Allen–Cahn Spearman | Allen–Cahn solape |
|---|---|---|---|---|
| R (residuo, línea base) | +0.453 | 0.398 | +0.366 | 0.020 |
| S1 ‖∂u/∂θ‖ | **+0.705** (p_H 0.026) | 0.212 (p_H 1.0) | **+0.635** (p_H 0.063) | **0.400** (p_H 0.090) |
| S2 bases activas | −0.294 | 0.017 | −0.232 | 0.000 |
| S3 curvatura de aristas | −0.352 | 0.000 | −0.231 | 0.000 |
| S5 desajuste de resolución | −0.364 | 0.207 | +0.358 | 0.061 |
| S4 asentamiento capa 1 | no computable | | | |

H-LOC-1 exige que la señal supere al residuo en **ambos** problemas y **ambas** métricas con p_H < 0.05: ninguna lo logra (S1 pierde en solape de Burgers, y en Allen–Cahn no llega a significancia). Con la prueba exacta que invierte el signo por **semilla** (unidad conservadora; mínimo p posible 1/32 y 1/8) ninguna diferencia es significativa.

### El caso de S1 es un artefacto de la restricción dura (análisis posterior, no preregistrado)

En Burgers y Allen–Cahn la salida es u = g(x, t) + (1 − x²) t · raw(x, t): la condición inicial y las de frontera se imponen por construcción y **el error es exactamente cero donde el factor w = (1 − x²) t se anula**. S1 = w · ‖∂raw/∂θ‖ hereda ese factor. Separando (`posthoc_s1_confound.py`, mediana Spearman / solape 5 %):

| | Burgers | Allen–Cahn |
|---|---|---|
| F = el factor w solo (geometría del dominio, **sin red**) | +0.697 / 0.120 | +0.644 / 0.305 |
| S1 | +0.705 / 0.212 | +0.635 / 0.400 |
| S1/w = ‖∂raw/∂θ‖ (lo que aporta la red) | **−0.317** / 0.077 | **−0.278** / 0.000 |

La ventaja de S1 sobre el residuo se explica por la geometría: la ubicación de la restricción dura, sin ninguna red, ya alcanza Spearman 0.70 y 0.64. La parte que depende de la red tiene correlación **negativa** con el error. **Consecuencia metodológica:** una correlación de Spearman sobre todo el dominio queda dominada por «dónde puede haber error»; el Spearman del residuo (0.45 y 0.37) está *por debajo* de la línea base geométrica. Lo que muestra localización más allá de la geometría es el solape del 5 % superior en Burgers (residuo 0.398 frente a 0.120 de la geometría); en Allen–Cahn ninguna señal de la red lo supera (geometría 0.305, residuo 0.020).

## 2. Parte 2 — atribución en el espacio de la representación: ni H-ATTR-1 ni H-ATTR-2 se sostienen

| | Burgers (25 pares) | Allen–Cahn (15 pares) |
|---|---|---|
| Spearman(A, Δ), mediana (umbral 0.30) | +0.329 | +0.099 |
| Control de ocupación O: Spearman(O, Δ) | **+0.464** | **+0.513** |
| Techo A* (con el error real) | +0.260 | +0.284 |
| A frente a O (p, permutación con signo) | 1.000 | 1.000 |
| **H-ATTR-1** | no (A no supera a la ocupación) | no |
| Reparto del 20 % superior de A (umbral 0.50; difuso = 0.20) | 0.367 | 0.282 |
| ... con el error real (A*) | 0.364 | 0.621 |
| **H-ATTR-2** | no | no |

Las bases que más se mueven en el refinamiento son las más usadas (ocupación), no las que más residuo «ven»; y la responsabilidad es difusa. Incluso con el error real (A*) la correlación con lo que L-BFGS movió es menor que la de la ocupación. En Allen–Cahn el error real sí se concentra en pocas bases (0.62) pero el residuo no (0.28): el residuo es una mala aproximación del error allí. Descriptivo: por capa, la correlación A–Δ es mayor en las capas 0 y 1 (Burgers 0.54 y 0.56) que en la 2 (0.17). Comprobación de efecto: Spearman entre la A de cada base de la capa 0 y la caída de error en su franja de soporte, mediana 0.69 (Burgers, 25 pares) y 0.56 (Allen–Cahn, 15): en las franjas con más residuo el error baja más al refinar, lo que describe el refinamiento observado y no prueba causalidad.

## 3. Qué dice el árbol de decisión firmado

«Nada pasa»: se diseña igual la focalización en el espacio de la representación, **primero con su techo oráculo (A* con el error real) en GPU** antes de cualquier versión con residuo; el resultado negativo de la Parte 1 se reporta como hallazgo.

Lectura sobre esa fila, para quien decide: el A* ya calculado es *descriptivo* (correlación con lo que L-BFGS movió) y no supera a la ocupación; el techo causal (¿mejora el error enfocar el refinamiento en las bases de mayor A*?) sigue sin medirse y es lo que respondería el EXP-0018.

## 4. Límites

- n = 5 y 3 semillas; los 5 estados de una semilla comparten la trayectoria de Adam: los p por estado son optimistas; la prueba exacta por semilla es la conservadora y no puede alcanzar significancia con esos n.
- Δ mide lo que movió L-BFGS, no lo que habría convenido mover.
- S4 no se pudo calcular; no se probó el asentamiento de una capa no separable, que era el control de la explicación del EXP-0016.
- Spearman sobre todo el dominio está dominado por la geometría de la restricción dura (ver arriba): las comparaciones de H-LOC-1 por Spearman deben leerse con esa línea base.
- Solo capas y señales definidas en el spec; no se exploró ninguna otra.
