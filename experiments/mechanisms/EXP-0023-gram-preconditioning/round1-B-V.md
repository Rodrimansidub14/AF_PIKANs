# EXP-0023 · ronda 1-B · etapa V — resultado (2026-10-21)

Kernel `rodrimandub/af-pikans-exp-0023-v-block` (T4×2, 1 680 s, **30 de 30 corridas completas y finitas**). Semillas 101–105 (bloque V, disjunto de T1), mismo presupuesto y receta que T1 (Adam 2 000 / 4 000, L-BFGS 500 it., P normalizado), B de comparador pareado. Resultados: `results/v-block/exp0023_vblock/`; análisis: `results/v-block/t1_block_analysis.json` (clave `passes_V_median_le_0.90_and_4of5`; la línea impresa dice «T1» por una etiqueta del script). Autorizado en el chat (2026-10-21). Exploratorio: `scientific_accepted=false`.

Criterio firmado para V: mediana de la razón ≤ 0.90 **y** mejora en ≥ 4 de 5 semillas, en Burgers **y** en Allen–Cahn.

| Razón tras L-BFGS vs B (mediana; mejora/5; por semilla) | Burgers | Allen–Cahn | Pasa V |
|---|---|---|---|
| E | **5.70; 1/5** [0.16, 4.38, 5.70, 11.08, 6.63] | 0.140; 5/5 [0.13, 0.14, 0.13, 0.14, 0.22] | **no** |
| F | 0.490; **3/5** [0.11, 0.44, 1.99, 0.49, 2.20] | 0.390; 5/5 [0.59, 0.39, 0.33, 0.56, 0.27] | **no** (Burgers 3/5 < 4) |

Error L2 relativo mediano tras L-BFGS: Burgers B 2.1e-3, E 1.4e-2, F 2.7e-3; Allen–Cahn B 4.8e-2, E 6.8e-3, F 2.1e-2. Relojes ×1.00–1.10 (dentro de la guardia).

## Lectura
- **Allen–Cahn replica:** E ≈ 0.14 (5/5) y F ≈ 0.39–0.44 (5/5) en T1 y en V. La mejora en Allen–Cahn es consistente en 10 semillas de dos bloques disjuntos.
- **Burgers no replica para E:** en T1 la mediana era 0.43 (4/5); en V es 5.70 (1/5). Cuatro de las cinco semillas de V empeoran (hasta 11×). F en Burgers mantiene la mediana (0.49) pero solo mejora 3/5 (T1: 4/5), con dos semillas peores que B (1.99, 2.20).
- **Ningún brazo cumple V:** no hay finalista de la ronda 1-B y, por la regla 3 del protocolo, nada pasa a C. Esto es lo que ya anticipaba la dependencia del problema vista con la idea A: el efecto es sólido en Allen–Cahn y no es estable en Burgers.
- Se mantiene la observación de T1 (F ≈ E en Allen–Cahn en dirección, aunque E es ≈ 3× mejor en error): no se puede atribuir la ganancia a la representación entrenada; y la variabilidad por semilla en Burgers (ya documentada en el EXP-0024) domina.
- Caveats: n = 5 por bloque; la regla de salida firmada (EXP-0024) habla de «no pasar T1»; aquí T1 se pasó y V no, un caso que la regla no nombra literalmente. La decisión de si esto activa la salida es del investigador.
