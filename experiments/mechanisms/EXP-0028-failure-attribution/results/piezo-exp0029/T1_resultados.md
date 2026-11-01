# T1: E frente a S, arranque en caliente desde B (paso 4000), 1000 pasos

| problema | semilla | l2re_u E → S | l2re_phi E → S | cociente u | cociente phi | pérdida final E → S | saltos E / S | s/paso E → S | eventos P4 |
|---|---|---|---|---|---|---|---|---|---|
| converse | 28 | 0.869 → 0.922 | 0.022 → 0.021 | 1.06 | 0.95 | 0.48 → 1.15 | 1 / 2 | 0.481 → 0.478 | 2 |
| converse | 29 | 0.734 → 0.747 | 0.024 → 0.030 | 1.02 | 1.27 | 0.524 → 0.465 | 2 / 2 | 0.477 → 0.486 | 2 |
| converse | 30 | 0.935 → 0.786 | 0.013 → 0.026 | 0.84 | 2.05 | 0.596 → 0.444 | 2 / 1 | 0.481 → 0.480 | 1 |
| direct | 28 | 0.144 → 0.202 | 0.097 → 0.157 | 1.40 | 1.63 | 0.0229 → 0.0435 | 4 / 3 | 0.363 → 0.366 | 3 |
| direct | 29 | 0.484 → 0.207 | 0.501 → 0.182 | 0.43 | 0.36 | 652 → 975 | 3 / 3 | 0.364 → 0.367 | 3 |
| direct | 30 | 0.343 → 0.359 | 0.327 → 0.344 | 1.05 | 1.05 | 0.0866 → 0.42 | 3 / 3 | 0.371 → 0.370 | 3 |

Converso: mediana phi 1.27 (máx 2.05) → NO cumple; mediana u 1.02 (máx 1.06) → cumple.
Directo: mediana u 1.05 (máx 1.40) → NO cumple. Saltos S−E máx 1 → cumple.
Veredicto del proxy: S NO pasa todos los criterios (n = 3 semillas converso, 3 directo; horizonte corto: da signo, no tamaño).