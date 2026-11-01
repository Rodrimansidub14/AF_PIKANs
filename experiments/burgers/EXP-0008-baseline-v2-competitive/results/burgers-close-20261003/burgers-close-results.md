# Burgers 1D (PINNacle, Cole–Hopf) — cierre del baseline (2026-10-03)

Kernel `rodrimandub/af-pikans-burgers-close-20261003`: COMPLETE, sesión 5 078 s (1.41 h), 15/15 corridas. Exploratorio; `scientific_accepted=false`. Contrato: ν=0.01/π, hard IC/BC, 8 192 puntos Sobol por semilla compartidos entre brazos, float32, 20 000 Adam (lr 1e-3) + L-BFGS strong-Wolfe (tope 600 s; todas terminaron por plateau), semillas 301–305, referencia Cole–Hopf 1025×129 (evaluador del verificador).

| Brazo | Parámetros | Fin de Adam | Tras L-BFGS |
|---|---|---|---|
| MLP 5×100 (PINNacle Vanilla) | 40 801 | 1.446e-2 ± 3.75e-3 | 5.66e-4 ± 5.7e-4 |
| RBF-PIKAN ancho 16 | 3 648 | 2.38e-2 ± 4.3e-2 | 1.87e-3 ± 2.4e-3 |
| RBF-PIKAN ancho 48 | 29 376 | 1.435e-2 ± 9.4e-3 | **3.87e-4 ± 2.2e-4** |

Lectura: el MLP reproduce el Vanilla publicado de PINNacle (1.45e-2) al final de Adam. RBF48 empata con el MLP (pareado tras L-BFGS −1.8e-4 ± 6.8e-4, RBF gana 2/5): no hay ventaja demostrada. RBF16 es peor. Por semilla: `burgers-close-per-seed.csv`.
