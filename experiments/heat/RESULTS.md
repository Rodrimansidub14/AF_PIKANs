# Heat 1D — resultados del baseline (EXP-0008)

Exploratorio. `scientific_accepted=false`, `reference_accepted=false`. Estado frente al baseline v1 firmado el 2026-10-03: ver `evals/baselines/baseline_v1_exploratory.md`.

**Contrato.** u_t = 0.1 u_xx en [0,1]², solución exacta sin(πx)e^(−0.1π²t), residuo escalado por 1.1. RBF-PIKAN `rbf_fixed`, float64, lote completo de
4 096 puntos interiores, Adam 20 000 pasos (lr 1e-3) y L-BFGS strong-Wolfe (lr 1, historia 100, bloques de 25 iteraciones con 125 evaluaciones).
Semillas 901–905, pools Sobol pareados. Puntuado en la malla analítica 512×256.

| Etapa | L2 relativo (media ± sd, n=5) |
|---|---|
| Fin de Adam | 2.197e-04 ± 1.80e-04 |
| + L-BFGS completo | **1.857e-06 ± 1.81e-07** |

**Lectura.** El L-BFGS elimina la dispersión entre semillas y llega a un piso de ≈1.9e-6 en ~50 iteraciones, limitado por la discretización
(4 096 puntos), no por la optimización. Un mecanismo solo puede mostrar efecto medible en la etapa Adam o en el costo para llegar al piso.
El valor archivado 2.06e-6 (L-BFGS truncado) queda supersedido.

**Comparadores.** DeepXDE α=0.1 (no pareado): 5.81e-4; B-spline S2 con L-BFGS: 1.416e-6 (seguimiento parcial).

**Artefactos.** `EXP-0008-baseline-v2-competitive/results/heat-lbfgs-redo-20260930/`, figura `EXP-0008-baseline-v2-competitive/figures/heat-lbfgs-stages-and-curves.png`, checkpoints con hashes en `checkpoints/`.
**Siguiente experimento.** EXP-0011 (RAD sobre este control), pre-registrado y firmado el 2026-10-03.
