# DFG con receta larga (2026-10-01) — resultados

Kernel `rodrimandub/af-pikans-dfg-long-20261001`: COMPLETE, sesión 12 010 s (3.34 h) de tope 14 400 s, 10/10 corridas `rc=0`. Exploratorio; `scientific_accepted=false`. Por semilla: `dfg-long-per-seed.csv`; artefactos en `outputs/dfgl/results/`.

Contrato: DFG Re=20, variante `corrected`, MLP 4×50 (7 953 parám.) vs RBF-PIKAN ancho 16 (4 032), float32, misma piscina por semilla (801–805), 20 000 Adam lr 1e-3 + L-BFGS strong-Wolfe (lr 1, hist. 100) con tope de 600 s. Puntuado vs FEM L3 (referencia de trabajo) y σ·n.

| Brazo | Etapa | u (L2 rel.) | v | p | error Cd | Cl abs. | error Δp |
|---|---|---|---|---|---|---|---|
| MLP 4×50 | fin de Adam | 0.671 ± 0.092 | 0.818 | 0.876 | 98.2 % | 0.019 | 100 % |
| MLP 4×50 | + L-BFGS | 0.421 ± 0.087 | 0.526 | 0.612 | 69.5 % | 0.143 | 71.3 % |
| RBF16 | fin de Adam | 0.461 ± 0.015 | 0.600 | 0.653 | 78.3 % | 0.070 | 81.9 % |
| RBF16 | + L-BFGS | **0.191 ± 0.039** | 0.277 | 0.305 | **34.7 % ± 6.6** | 0.067 | 41.2 % |

RBF gana 5/5 en u tras L-BFGS (0.166–0.258 vs 0.269–0.474).

## Lectura
- Mejora mucho respecto a la config de Daniel (2000 Adam), pero **no es competitivo**: las cohortes previas 301–305 / 501–505 llegaban a 7–9 % en u y 5–8 % en Cd. Con esta arquitectura pequeña, float32 y 20k Adam, Cd sigue 35 % (RBF) y 70 % (MLP) lejos.
- **El L-BFGS terminó siempre por tope de tiempo** (`lbfgs_time_cap`, RBF ~4 000 it., MLP ~11 500 it.), no por plateau: sigue mejorando; el resultado está limitado por presupuesto, no convergido. El RBF hace 3× menos iteraciones por segundo y aun así gana.
- Esta corrida **no reemplaza** las cohortes DFG previas como baseline; es la comparación pareada con la configuración de la tesis de Daniel bajo receta larga.
- Cuota tras la corrida: 3.55 h restantes de 30 h (renueva 2026-10-03).
