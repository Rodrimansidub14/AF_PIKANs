# EXP-0028 Fase 1c (NS2d-CG) — presupuesto, puntos del GN y estimación en T4 (2026-10-23)

NO LANZADO. Medido en CPU (2 hilos, float32); T4 estimado con un supuesto declarado abajo.

## Receta (fijada; sin tocar el worker de NS2d-CG)
RBF16 (2-16-16-3, 11 bases, 4 032 parámetros), pools de `NSProblem` (domain 8 192, inlet 170, outlet 342, other 1 536), pérdida de 8 términos con peso 1, Adam lr 1e-3, lote completo, float32, **20 000 pasos**, luego L-BFGS strong-Wolfe **500 iteraciones** (`sweep.lbfgs_stage`, chunk 50). Precondicionador: recálculo antes del paso 1, cada 1 000 pasos (pasos 0, 1 000, …, 19 000 = 20 recálculos) y uno al arrancar L-BFGS = **21 recálculos por corrida** de J/E/I. Semillas 4001-4010, brazos B, F, J, E, I = 50 corridas.

## Puntos del Gauss-Newton: fijados en 2048 por clave
`gram_preconditioning.subset` toma hasta 2048 puntos equiespaciados de cada clave: domain 8 192 → 2 048; inlet (170), outlet (342) y other (1 536) quedan completas. Total **4 096 puntos, los 8 términos con sus pesos (1)**. Es el N = 2048 de 1-B/1-C y del firmado en EXP-0023. Costo medido (CPU): E con 128/clave (512 puntos) 24-26 s por recálculo; con 2048/clave (4 096 puntos) 171 s (`budget_vector.json`, máquina cargada) y 102 s en una segunda medición con la máquina libre (`fase1c_cpu/timing_ns.json`, chunk 16): 8× los puntos cuestan ~4-7× el tiempo. En T4 el recálculo pesa ≤ 10 % de cada 1 000 pasos de Adam (abajo), así que no se recorta a 1 024/clave (ahorraría ~3 % del total y se saldría de la convención 0023).

## Mediciones en CPU (`fase1c_cpu/timing_ns.json`, 2 hilos, máquina libre)
| Concepto | CPU | T4 |
|---|---|---|
| Adam, s/paso lote completo | 2.40 (6.98 con la máquina cargada, `budget_vector.json`) | **0.0577 medido** (kernel NS del 2026-09-30: 1 154.5 s / 20 000 pasos, RBF16 semilla 701) |
| L-BFGS, s/iteración | 3.59 | ≈ 0.086 (escalado por la razón Adam CPU/T4 = 41.5×) → 500 it. ≈ 43 s |
| Recálculo J / E / I (4 096 pts, chunk 16) | 120 / 102 / 81 s | **estimado** 2.0-6.0 / 2.0-5.1 / 1.6-4.1 s |

**Supuesto declarado:** el recálculo (Jacobianos con `torch.func`, lanzamientos pequeños de kernels) se acelera en la T4 entre 20× y 50× respecto del CPU de 2 hilos (el rango pedido); la razón medida para Adam es 41.5×. Si el recálculo estuviera limitado por lanzamientos y la ganancia fuera menor de 20×, cada recálculo sería más caro; con 10× serían 8-12 s y cada corrida J/E/I subiría ~100 s (≈ 7 %), sin cambiar el reparto en bloques. **Cómo medirlo de verdad:** kernel `rodrimandub/af-pikans-exp-0028-phase1c-timing` (BLOCK `t`, ≈ 5-15 min): corre `run_phase1c.py --check-id` y `--timing` en la T4 y deja `check_B_ID_ns.json` y `timing_ns.json` (s/paso de Adam, s/iteración de L-BFGS, s por recálculo J/E/I con chunk 8 y 16, memoria pico CUDA). Cada kernel de producción repite ambos pasos al inicio y no despacha nada si B ≠ ID.

## Estimación por corrida y total en T4 (receta completa)
Por corrida: Adam 20 000 × 0.0577 = 1 155 s; F/J/E/I +5 % en el paso (aplicar P; supuesto); L-BFGS ≈ 43 s; arranque + calentamiento + puntuación ≈ 15 s; recálculos 21 × (J 6.0 / E 5.1 / I 4.1 s con 20×; ÷2.5 con 50×).

| Brazo | s/corrida (20×) | s/corrida (50×) |
|---|---|---|
| B | 1 213 | 1 213 |
| F | 1 271 | 1 271 |
| J | 1 397 | 1 321 |
| E | 1 378 | 1 314 |
| I | 1 356 | 1 305 |
| suma por semilla | **6 616** | **6 424** |

10 semillas: 18.4 / 17.8 horas-GPU = **9.2 / 8.9 h de pared en T4×2**. Con el tope de 4 h por lanzamiento (despacho hasta 3.4 h, kill a 3.9 h) no cabe en uno: **4 lanzamientos** por bloques de semillas completas (15 corridas ≈ 2.7-2.8 h de pared + 5-10 min de pruebas previas; margen ~0.5 h bajo el corte de 3.4 h):

| Kernel (slug) | Semillas | Corridas | Pared estimada |
|---|---|---|---|
| `rodrimandub/af-pikans-exp-0028-phase1c-timing` | (solo B == ID y tiempos) | 0 | 5-15 min |
| `rodrimandub/af-pikans-exp-0028-phase1c` | 4001-4003 | 15 | 2.8-3.0 h |
| `rodrimandub/af-pikans-exp-0028-phase1c-b2` | 4004-4006 | 15 | 2.8-3.0 h |
| `rodrimandub/af-pikans-exp-0028-phase1c-b3` | 4007-4009 | 15 | 2.8-3.0 h |
| `rodrimandub/af-pikans-exp-0028-phase1c-b4` | 4010 | 5 | 1.1-1.3 h |

Total ≈ 9.8 h de pared y 4 cuotas de sesión. Si la T4 resultara ≥ 20 % más lenta que lo estimado, el corte de despacho (3.4 h) deja sin empezar las últimas corridas de la última semilla del bloque (ausencia de medición, reanudable en un lanzamiento posterior; los kernels no pisan lo ya escrito si se les da `X28_SEEDS`).

## Alternativa T1 (decisión del investigador al firmar 1c)
Los otros cuatro problemas del banco usan Adam al 10 % del control (convención T1). Para NS eso es `--recipe t1` (Adam 2 000; 3 recálculos): ≈ 115 + 43 + 15 + 3×5 ≈ 190 s por corrida, 50 corridas ≈ 2.6 horas-GPU = **≈ 1.3 h de pared en un solo lanzamiento**. No es el defecto: el spec dice «Adam 20 000» para NS y el worker lo usa; con 2 000 pasos el modelo de NS aún está lejos de converger (la campaña NS reportó el error a 20 000), así que el efecto de F/J/E/I podría no ser comparable con la receta de la línea base. Se deja la elección al investigador; el comando de lanzamiento usa la receta completa.

## Pruebas hechas en CPU (1 semilla, pasos reducidos)
- `fase1c_cpu/check_B_ID_ns.json`: Adam 20 pasos, pool submuestreado a 512 por clave, 5 iteraciones de L-BFGS: **B coincide bit a bit con ID** (parámetros finales, pérdida cada 100 pasos, puntuación en Adam y final, pérdida y términos finales, mismo número de iteraciones de L-BFGS) **y con `sweep.adam_stage`** (el lazo del control). `pass: true`.
- Humo de los 5 brazos (semilla 4001, Adam 30, L-BFGS 50, recálculo cada 15, 64 puntos por clave, pool a 256): corren de principio a fin y escriben el JSON; los números no significan nada (pasos reducidos).
- Humo del lanzador en modo bundle (`BUNDLE_DIR=kaggle/dataset_1c`, `X28_TINY=1`, sin acceso al repo): prueba B == ID, 5 trabajos por 2 ranuras, `ledger.json` completo. Un `--timing` con `--smoke-pool` no se corrió en el lanzador tiny (sí a pool completo, arriba).
