# EXP-0018 — resultados (2026-10-20)

Exploratorio. `scientific_accepted=false`. Spec v2 firmado el 2026-10-20; criterios, umbrales (0.85 / 0.90, 4 de 5), β = 1, q = 0.5 y guardias no se movieron. Los brazos con A\* leen la referencia: son `oracle_ceiling`, no candidatos, no entran a `claims.jsonl`. Bloque en Kaggle T4×2 (`rodrimandub/af-pikans-exp-0018-ceiling`): 10 de 10 trabajos, 1.42 h (esperado ≈ 1.2 h, tope 3 h). La rama N-H no se corrió (el spec la reserva para cuando H pase H-CEIL-1). Datos: `results/ceiling/ceiling/`, análisis en `results/ceiling_analysis.json` (`analyze_ceiling.py`).

**Control.** La rama B reproduce bit a bit el L-BFGS de la etapa A del EXP-0016 en las 8 semillas que existían (diferencia relativa 0 en Burgers, ≤ 1.5e-16 en Allen–Cahn); Allen–Cahn 604 y 605 entrenaron su Adam aquí.

## Razón de error por brazo (error global tras L-BFGS / error de B, por semilla)

| | Burgers (301–305) | Allen–Cahn (601–605) |
|---|---|---|
| **P** foco suave por A\* | 1.12, 1.02, 0.99, 1.07, 1.03 · mediana **1.03** · mejora 1/5 | 1.38, 0.56, 0.76, 1.21, 1.05 · mediana **1.05** · mejora 2/5 |
| **H** foco duro (50 % de bases) | 1.51, 2.88, 0.46, 1.26, 1.52 · mediana **1.51** · mejora 1/5 | 2.35, 1.70, 1.36, 1.77, 0.59 · mediana **1.70** · mejora 1/5 |
| N, escalas permutadas (control) | 0.82, 0.93, 0.73, 0.45, 1.26 · mediana **0.82** · mejora 4/5 | 1.27, 1.93, 1.67, 1.23, 1.37 · mediana **1.37** · mejora 0/5 |
| O, escalas por ocupación (exploratorio) | 0.92, 1.07, 0.94, 0.40, 1.10 · mediana 0.94 · mejora 3/5 | 1.34, 1.52, 1.44, 1.04, 0.80 · mediana 1.34 · mejora 1/5 |
| P frente a N | mediana 1.34 · P mejor en 1/5 | mediana 0.77 · P mejor en 4/5 |

Guardias: región crítica (razón, media geométrica) — P 1.05 y 0.94, H 1.30 y 1.42; reloj de L-BFGS — P 1.03 y 0.97, H 1.29 y 1.09; todos los parámetros finitos. Descriptivo: el 20 % superior de A\* concentra 0.30–0.38 (Burgers) y 0.60–0.63 (Allen–Cahn) de la responsabilidad total, como en el EXP-0017.

## Veredicto según los criterios firmados

- **H-CEIL-1 (el techo existe): no se sostiene.** P: mediana de r = 1.03 y 1.05, mejora en 1/5 y 2/5 (se exigía ≤ 0.85 y 4/5). H: 1.51 y 1.70, con la región crítica empeorando 30 % y 42 % (falla la guardia g1).
- **H-CEIL-2** no aplica (ningún brazo pasó H-CEIL-1).
- Regla de decisión del spec: **H-CEIL-1 falla en ambos brazos → se cierra la familia «foco por responsabilidad por base» (estática, β = 1, q = 0.5)**; se reporta como techo negativo. El brazo O queda como la única señal sin explorar.

## Lectura de los controles (descriptiva)

- Escalar la fase de refinamiento con el error real (P) no mueve el error respecto del control (razón ≈ 1.03), y congelar la mitad de las bases menos responsables (H) empeora el error y el frente. **Ni siquiera con el error verdadero en la mano hay techo** en esta construcción.
- El control N (las mismas escalas repartidas al azar, sin información) en Burgers *habría cumplido* el criterio numérico de H-CEIL-1 por sí solo en ese problema (mediana 0.82, mejora 4/5), y en Allen–Cahn lo contrario (1.37, 0/5). Es consistente con el ruido medido en la calibración (desviación típica de la razón ≈ 0.7 en Burgers) y muestra por qué el criterio exigía **ambos** problemas y por qué el brazo permutado era necesario: un cambio arbitrario de precondicionador mueve el error en cualquier dirección.
- El escalado heterogéneo no es un efecto robusto: mejora Burgers (N y O: GM 0.79 y 0.83) y empeora Allen–Cahn (GM 1.47 y 1.20).

## Límites

- A\* es estática (se calcula una vez en el estado de partida; L-BFGS se mueve mucho después): un nulo no descarta una A\* que se refresque.
- n = 5 por problema; la potencia del diseño es ≈ 53 % si el efecto real fuera una reducción de 40 % y ≈ 11 % si fuera de 20 %; un nulo no descarta efectos pequeños. Los p exactos por semilla (P: 1.00 y 0.69; H: 0.88 y 0.88) se reportan y no deciden (con n = 5 el mínimo posible tras Holm es 0.125).
- Un solo β, un solo q, solo capas 0–2 y solo dos problemas. N-H no se corrió.
- Los brazos P/H/N/O parten del estado final de Adam; el momento de traspaso no se varió.
