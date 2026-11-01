# EXP-0023 · ronda 1-B · nivel T1 — resultado (2026-10-21)

Kernel `rodrimandub/af-pikans-exp-0023-t1-block` (T4×2, 2 860 s de sesión, **50 de 50 corridas completas**, todas finitas, L-BFGS por tope de 500 iteraciones). Resultados: `results/t1-block/exp0023_t1block/`; análisis: `results/t1-block/t1_block_analysis.json` (`analyze_block.py`). Convención T1 + Enmienda 1 (P normalizado); semillas 1–5; B es el comparador pareado de la misma sesión. Autorizado por el investigador en el chat (2026-10-21). Exploratorio: `scientific_accepted=false`.

Criterio firmado (regla de salida): mediana de la razón error(brazo)/error(B) tras L-BFGS ≤ 0.90 en Burgers **y** en Allen–Cahn.

| Razón tras L-BFGS (mediana; mejora/5; por semilla) | Burgers | Allen–Cahn | Reloj vs B | Pasa T1 |
|---|---|---|---|---|
| J (solo Jacobi) | 5.54; 1/5 | 0.884; 3/5 | ×1.08–1.09 | no |
| **E** (bloque por arista) | **0.432; 4/5** [4.17, 0.04, 0.50, 0.43, 0.38] | **0.136; 5/5** [0.14, 0.20, 0.12, 0.21, 0.11] | ×1.09–1.10 | **sí** |
| I (bloque por entrada) | 181.9; 0/5 | 1.450; 2/5 | ×1.10–1.15 | no |
| **F** (Gram fija, uniforme) | **0.489; 4/5** [0.39, 0.14, 1.53, 0.62, 0.49] | **0.439; 5/5** [0.21, 0.56, 0.52, 0.23, 0.44] | ×1.00–1.03 | **sí** |

Error L2 relativo mediano tras L-BFGS: Burgers B 1.9e-3, E 7e-4, F 1.0e-3, J 1.6e-2, I 0.34; Allen–Cahn B 5.1e-2, E 7.0e-3, F 2.2e-2, J 4.7e-2, I 7e-2 (mediana; I es bimodal: 7e-4 a 0.51).

## Lectura (sin ir más allá de los datos)
- **E y F pasan T1 en los dos problemas.** E es el más fuerte en Allen–Cahn (5/5, ≈ 7× menos error); en Burgers la mejora es más variable (razones de 0.04 a 4.17: una semilla empeora 4×). Reloj dentro de la guardia (≤ 1.2×).
- **J no pasa** y empeora Burgers: no es un simple reescalado diagonal (consistente con la fila «H-PRE-1 sí, H-PRE-2 no» no siendo el caso para E).
- **F (Gram fija de la base, sin activaciones entrenadas) pasa casi igual que E.** Si esto se sostiene en V, apunta a que la ganancia viene de corregir el solape de la base RBF y no de la representación entrenada (la fila «H-PRE-1 y H-PRE-2, no H-PRE-3» del spec 0023: se reportaría como hallazgo sobre la base, no como mecanismo de representación). T1 no puede separar E de F: con n = 5 las medianas 0.43 vs 0.49 y 0.14 vs 0.44 no permiten afirmar que difieran.
- **I** es inestable (error ≥ 0.05 en las 5 semillas de Burgers); I no pasa.
- Caveats: n = 5 y potencia limitada; la sensibilidad numérica por semilla en Burgers ya documentada (EXP-0024) también aplica; el comparador B de Burgers tiene una semilla mala (1.27e-2). No es evidencia de mecanismo, y la aprobación de una prueba de bloque no es preregistro de confirmación.

## Siguiente paso según el protocolo (decisión del investigador)
Regla 3: pasa a V (semillas 100–199, mismo presupuesto) el brazo (o brazos) que pasó T1, con criterio mediana ≤ 0.90 y mejora en ≥ 4 de 5 semillas en ambos problemas; a C pasa un solo finalista. Aquí E y F cumplen T1 (y E y F tienen 4/5 en Burgers y 5/5 en AC, por lo que cumplirían también el conteo de V si se repitiera con estas semillas).
