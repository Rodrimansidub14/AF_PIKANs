# DFG 2D-1 Re=20 — las tres cohortes, con sus recetas lado a lado

Lectura de manifiestos y resultados existentes (2026-10-04); no se corrió nada. Exploratorio. Todas puntuadas con la fórmula de fuerzas σ·n (corregida el 2026-09-30) y, para campo, contra el FEM Taylor–Hood L3 (referencia de trabajo aceptada con Cl absoluto; hecha por un agente). Referencia de QoI: FEATFLOW (Cd 5.5795, Δp 0.11752, Cl 0.010619).

## Recetas

| | **Cohorte A: 301–305** | **Cohorte B: 501–505** | **Cohorte C: 801–805** |
|---|---|---|---|
| Origen | `baseline-close-2026-09-27/dfg-re20-gpu-r3` y `r4`; bundle EXP-0007 | kernel `af-pikans-dfg-re20-competitive-pilot-t4x2`; refinamiento `dfg-lbfgs-refinement` | `ns2d-cg-20260930/dfg_daniel.py` y `dfg_long.py` |
| Brazos | solo RBF-PIKAN 2-16-16-3 (11 bases, centros fijos) | RBF-PIKAN 2-16-16-3 **y** tanh-MLP 5×64 (pareados por semilla) | MLP 4×50 tanh y RBF16 (pareados) |
| Muestreo | Sobol fijo; 16 384 interiores, 512 puntos de borde; rechazo dentro del cilindro | igual (misma familia de runner) | LHS de la tesis de Daniel: 22 345 interiores, 318 de borde |
| Pérdida / salidas | residuos con los QoI de FEATFLOW; salida natural do-nothing | igual | salidas (u,v,p) crudas, pérdida sin normalizar, variante `corrected` |
| Adam | lr 1e-3, lote 1 024 interiores / 256 de borde, tope de 1 200 s por semilla (máx. 50 000 pasos declarados; los pasos reales **no verificados aquí**) | igual (1 200 s por par de brazos) | 20 000 pasos, lote completo, lr 1e-3 |
| L-BFGS | continuación sobre el pool completo, 540 s por semilla (1 643 pasos en total en las 5 semillas) | continuación de 540 s por par (~520 pasos por brazo y semilla) | strong-Wolfe, lr 1, tope de 600 s |
| Precisión | float32 | float32 | float32 |
| Referencia de campo | FEM L3 (puntuado a posteriori) | FEM L3 (puntuado a posteriori) | FEM L3 |
| Estado | corridas completas; r3 con harness que no cerró válido, r4 válido | completa | completa; L-BFGS terminó por tope de tiempo |

## Resultados tras L-BFGS (n = 5 semillas por brazo; media ± sd)

| | Cd (error rel.) | Δp (error rel.) | Cl (error abs.) | u (L2 vs FEM L3) | v | p |
|---|---|---|---|---|---|---|
| A · RBF 301–305 | 5.124 ± 0.053 (**8.2 %**) | 9.9 % | 0.083 | 0.086 ± 0.012 | 0.085 ± 0.010 | 0.275 ± 0.042 |
| B · RBF 501–505 | 5.233 ± 0.052 (**6.2 %**) | 7.3 % | 0.049 | 0.070 ± 0.004 | 0.069 ± 0.006 | 0.246 ± 0.023 |
| B · MLP 5×64 501–505 | 5.286 ± 0.284 (**5.3 %** ± 5.1) | 5.3 % ± 7.1 | 0.024 | 0.047 ± 0.069 | 0.052 ± 0.049 | 0.123 ± 0.215 |
| C · RBF16 801–805 | Cd error 34.7 % ± 6.6 | 41.2 % | 0.067 | 0.191 ± 0.039 | 0.277 | 0.305 |
| C · MLP 4×50 801–805 | Cd error 69.5 % ± 16 | 71.3 % | 0.143 | 0.421 ± 0.087 | 0.526 | 0.612 |

Antes de L-BFGS, la cohorte A tenía Cd con 28.9 % de error y Δp con 33.9 %: L-BFGS es lo que lleva los QoI a menos del 10 %.

## Lectura

1. **Las cohortes A y B sí están cerca de la referencia en los QoI** (5–8 % en Cd), y la B tiene un par pareado RBF–MLP. La C no converge: usa la receta de la tesis de Daniel (LHS, 4×50, ancho 16, lote completo, L-BFGS limitado a 600 s) y queda a 35–70 % en Cd.
2. **El campo está menos bien resuelto que los QoI.** u y v en 5–9 % para el RBF, pero **p en ~25 %**. El Cd es una integral sobre el cilindro y puede acertar con un campo de presión impreciso.
3. **RBF contra MLP en la cohorte B:** el MLP queda mejor en media (u 0.047 contra 0.070; p 0.12 contra 0.25) pero con una dispersión enorme entre semillas (sd mayor que la media en u y p: hay semillas muy buenas y otras malas, **no investigado**); el RBF es mucho más estable (sd 0.004 en u). No hay ventaja demostrada del RBF en error medio.
4. **A y B no son comparables entre sí** como una sola cohorte: mismas familias de muestreo y runner, pero se entrenaron en sesiones distintas, la A no tiene MLP y la A usa L-BFGS de 540 s por semilla mientras que la B lo hace por par de brazos. Se mantienen separadas.
5. **Qué falta para usar DFG como prueba de mecanismo:** un control único y pareado (RBF y MLP, mismas semillas y misma receta, L-BFGS a plateau), con presión mejor resuelta. Hoy el más cercano es la cohorte B.

## Decisión pendiente (humano)

¿DFG entra como prueba del mecanismo, o queda como prueba de estrés para la tesis? Recomendación: **prueba de estrés**, con la cohorte B como referencia exploratoria y sin ampliarla hasta tener un mecanismo definido sobre Burgers y Allen–Cahn.
