# Burgers 1D (PINNacle) — resultados del baseline (EXP-0008)

Exploratorio. `scientific_accepted=false`, `reference_accepted=false`. Estado frente al baseline v1 firmado el 2026-10-03: ver `evals/baselines/baseline_v1_exploratory.md`.

**Contrato.** u_t + u u_x = (0.01/π) u_xx en [−1,1]×[0,1], IC −sin(πx), BC homogéneas duras; referencia Cole–Hopf (Gauss–Hermite 128) 1025×129, evaluador del verificador.
8 192 puntos Sobol por semilla compartidos entre brazos, Adam 20 000 pasos (lr 1e-3, float32) y L-BFGS strong-Wolfe. Semillas 301–305.

| Brazo | Parámetros | Fin de Adam | L-BFGS float32 (600 s) | L-BFGS float64 (900 s) | **Grande**: 16 384 pts, 24k Adam, L-BFGS f64 1 800 s |
|---|---|---|---|---|---|
| MLP 5×100 (PINNacle Vanilla) | 40 801 | 1.446e-02 ± 3.75e-03 | 5.661e-04 ± 5.70e-04 | 4.059e-04 ± 3.37e-04 | 1.050e-04 ± 2.23e-05 |
| RBF-PIKAN ancho 16 | 3 648 | 2.376e-02 ± 4.31e-02 | 1.868e-03 ± 2.42e-03 | — | — |
| RBF-PIKAN ancho 48 | 29 376 | 1.435e-02 ± 9.44e-03 | 3.867e-04 ± 2.24e-04 | 2.900e-04 ± 1.61e-04 | **7.165e-05 ± 4.80e-05** |

**Contra la referencia FEM de PINNacle** (101×11; comparación directa con lo publicado, 1.33e-2 ± 5.3e-5 con L-BFGS): mlp 1.330e-02 ± 7.93e-05; rbf16 1.382e-02 ± 1.10e-03; rbf48 1.327e-02 ± 2.07e-05. Es el piso de la referencia: no discrimina entre modelos.

**Lectura.** El MLP reproduce el Vanilla publicado (1.45e-2) al final de Adam. RBF48 empata con el MLP tras L-BFGS (pareado float64 −1.2e-4 ± 4.1e-4, RBF gana 3/5): no hay ventaja demostrada.
float64 mejora ~25–30 % sobre float32; todas las corridas de float64 terminaron por tope de tiempo. RBF16 es peor (~3×).
La corrida grande (16 384 puntos, 24 000 pasos de Adam, L-BFGS float64 1 800 s; sesión de 4.6 h) baja ~4× más: MLP 1.05e-4 ± 2.2e-5, RBF48 **7.16e-5 ± 4.8e-5**; pareado RBF48 − MLP −3.3e-5 ± 6.4e-5 (RBF gana 4/5): ventaja pequeña, sin significancia con n=5. Todas terminaron por tope de tiempo de L-BFGS.

**Artefactos.** `EXP-0008-baseline-v2-competitive/results/burgers-close-20261003/`, `burgers-f64-lbfgs-20261003/`, figura `EXP-0008-baseline-v2-competitive/figures/burgers-l2-by-stage.png`.
**Límites.** Sin SOTA de métodos especializados reproducido; el control del baseline es RBF48.
