# Pasos locales 1, 2 y 4 — 2026-09-30

Solo lectura sobre artefactos archivados + cálculo en CPU. Sin entrenamiento nuevo, sin tocar `evals/`, `registry/` ni umbrales. Nada acepta referencias ni claims.

## Paso 1 — Allen–Cahn contra `Allen_Cahn.mat` (referencia de terceros, DeepXDE)

`score_ac_vs_mat.py` → `ac-vs-mat-per-seed.csv`, `ac-vs-mat-summary.json`. Métrica de DeepXDE (L2 relativo sin pesos, malla 201×101), 5 semillas por brazo, mismos checkpoints archivados.

| Brazo | Endpoint | L2 relativo vs `.mat` (media ± sd, n=5) |
|---|---|---|
| RBF-PIKAN | Adam 600 s (tope de tiempo) | `1.6565e-2 ± 2.440e-3` |
| RBF-PIKAN | + L-BFGS 240 s | `9.184e-3 ± 2.678e-3` |
| MLP tanh 4×20 | Adam 600 s | `7.869e-2 ± 1.758e-2` |
| MLP tanh 4×20 | + L-BFGS 240 s | `2.296e-2 ± 6.633e-3` |

Diferencia pareada RBF−MLP: Adam `−6.21e-2 ± 1.59e-2`, tras L-BFGS `−1.38e-2 ± 8.15e-3`; RBF gana 5/5 en ambos endpoints. Los números coinciden con los de la referencia candidata FD8192 (RBF `1.6507e-2` / `9.164e-3`) dentro de ~0.4 %: con referencia de terceros el resultado no cambia.

Notas: (a) el "MLP DeepXDE" de esas corridas tiene 4 capas ocultas de 20, mientras la demo oficial usa 3×20; la re-corrida usa 3×20. (b) Estos endpoints siguen siendo de tiempo fijo, no de pasos fijos.

## Paso 2a — la fórmula de fuerzas de `dfg.py` contra el campo FEM L3

`validate_force_formula_fem.py` → `force-formula-fem-validation.json`. Sin dolfinx: se evalúan el campo P2 de velocidad y P1 de presión isoparamétricos desde `mesh.h5` + `fem_coefficients.npz` sobre el contorno del cilindro (2048 puntos) y se pasa por `qoi_from_cylinder`.

| Fórmula | Cd sobre el campo FEM L3 | Error vs FEATFLOW (5.5795) |
|---|---|---|
| σ·n (`dfg.py`, vigente) | 5.5583 | **0.38 %** |
| tangencial (scorer antiguo) | 1.7003 | **69.5 %** |

Δp con σ·n: 0.11693 (0.50 %). Cl con σ·n: 0.0175 (error absoluto 0.007): **Cl no queda validado** — mi evaluador isoparamétrico extrapola en los puntos que caen sobre el borde curvo (coordenada de referencia mínima −0.26), y Cl es casi cero. Para Cd y Δp la validación es limpia.

**Conclusión: el scorer tangencial es incorrecto** (falla 70 % sobre un campo exacto); σ·n es correcto.

## Paso 2b — re-puntuación de todos los checkpoints DFG que hay en disco

`rescore_dfg_checkpoints.py` → `dfg-rescore-per-checkpoint.csv`, `dfg-rescore-summary.json`. Una sola fórmula (σ·n), 2048 puntos, float64. Se muestra también la fórmula antigua para ver la discrepancia.

| Cohorte | n | Cd σ·n | Error Cd | Cd fórmula antigua | Δp error |
|---|---|---|---|---|---|
| 301–305 RBF, antes de L-BFGS | 5 | 3.967 ± 0.180 | 28.9 % ± 3.2 | 1.075 ± 0.095 | 33.9 % |
| 301–305 RBF, tras L-BFGS | 5 | 5.124 ± 0.053 | **8.2 % ± 1.0** | 1.463 ± 0.106 | 9.9 % |
| 501–505 RBF, tras L-BFGS | 5 | 5.233 ± 0.052 | 6.2 % ± 0.9 | 1.498 ± 0.047 | 7.3 % |
| 501–505 MLP 5×64, tras L-BFGS | 5 | 5.286 ± 0.284 | 5.3 % ± 5.1 | 1.553 ± 0.155 | 5.3 % |

Hallazgos:

1. **Las filas de Cd/Cl de 301–305 en `scorecard-current.csv` (Cd `1.0752` antes y `1.4629` tras L-BFGS, "80.7 % / 73.8 % de error") están calculadas con la fórmula tangencial incorrecta.** Los valores `1.0752 ± 0.0946` y `1.4629 ± 0.1055` se reproducen exactamente con la fórmula antigua. Con σ·n el error de arrastre de esa cohorte es 28.9 % → 8.2 %.
2. **Las filas de 501–505 sí están bien**: Cd `5.2334 ± 0.0518` y `5.2860 ± 0.2836`, Cl `0.0144 ± 0.0722` se reproducen con σ·n. La corrección que hice ayer ("el 6 % probablemente es de un evaluador obsoleto") **era errónea**: es el 74 % el que está mal.
3. Δp no depende de esa fórmula; sus filas son correctas.
4. El L-BFGS de 301–305 mejoró Cd de 29 % a 8 % y Δp de 34 % a 10 %: el entrenamiento funcionó, lo que estaba roto era la medición.

Pendiente humano: decidir cómo se marcan/sustituyen esas filas del scorecard (no las edité).

## Paso 4 — prueba de capacidad (regresión sobre el campo FEM L3)

`capacity_fit_dfg.py` → `capacity-fit-dfg.json`. Misma arquitectura 2-16-16-3 con 11 gaussianas por arista (4 032 parámetros), regresión supervisada 4 000 pasos de Adam sobre nodos FEM, 20 % de nodos retenidos:

| | u | v | p (gauge: media) |
|---|---|---|---|
| L2 relativo, nodos retenidos | **1.10 %** | 4.93 % | **1.05 %** |

**La arquitectura actual SÍ puede representar el campo DFG** (≈1 % en u y p con solo 4 000 pasos). Mi hipótesis previa de que la red era demasiado pequeña para dibujar el cilindro queda **refutada** para representación; los errores de campo de 7–9 % (u, v) y 25 % (p) tras el entrenamiento físico son de optimización/pérdida, no de capacidad. Solo se completó la variante actual: las variantes más anchas eran demasiado lentas en CPU y no hacen falta para responder la pregunta.
