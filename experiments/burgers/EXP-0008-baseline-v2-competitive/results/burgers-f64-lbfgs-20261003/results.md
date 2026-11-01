# Burgers 1D — Adam float32 + L-BFGS float64 (2026-10-03)

Kernel `rodrimandub/af-pikans-burgers-f64-lbfgs-20261003`: COMPLETE, sesión 7 963 s (2.2 h), 10/10 corridas. Mismo contrato y mismas semillas (301–305) que `burgers-close`: Adam idéntico en float32 (20 000 pasos), después L-BFGS strong-Wolfe en float64 con tope de 900 s (todas terminaron por tope de tiempo, no por plateau). Referencia Cole–Hopf 1025×129. Exploratorio.

| Brazo | Fin de Adam | L-BFGS float32 (600 s, plateau) | L-BFGS float64 (900 s, tope) |
|---|---|---|---|
| MLP 5×100 | 1.446e-2 ± 3.8e-3 | 5.66e-4 ± 5.7e-4 | 4.06e-4 ± 3.4e-4 |
| RBF48 | 1.435e-2 ± 9.4e-3 | 3.87e-4 ± 2.2e-4 | **2.90e-4 ± 1.6e-4** |

Diferencia pareada RBF48 − MLP en float64: −1.16e-4 ± 4.1e-4 (RBF gana 3/5): sigue siendo empate dentro del ruido. float64 mejora ~25–30 % respecto a float32, no un orden de magnitud; el tope de tiempo de L-BFGS sigue activo, así que una continuación más larga puede bajar más (`burgers-big`, en curso).
