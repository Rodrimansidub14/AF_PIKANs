# EXP-0028 Fase 2 (piezo real) - propiedades a la inicializacion con el problema del paper, sin entrenar, CPU float64

Semillas 20260728-20260730 (n = 3 por problema). Celda = mediana [min, max]. Medido con `measure_properties_piezo.py` (PiezoProblem real del paper + `piezo_residual_terms`; RBF-PIKAN 4-16-16-8 de 11 bases con las entradas/salidas del paper; pools del paper: 2 x 2048 interiores + 256 por frontera; GN con 2048 por clave; pesos fijos del paper; enrutamiento en el converso). Columna `stand-in`: el piezo propio anterior (`piezo_standin` de fase2/properties_terms.md: 2 entradas, escalas SC_*, pesos 1, sin fase ni enrutamiento). Columna `banco`: rango de los 6 problemas del banco (AC x2, Burgers, KPP, NS, DFG).

| propiedad | converso | directo | stand-in anterior | banco (6 problemas) |
|---|---|---|---|---|
| `n_outputs` | 8 [8, 8] | 8 [8, 8] | 8 [8, 8] | 1 a 3 |
| `n_loss_terms` | 16 [16, 16] | 17 [17, 17] | 16 [16, 16] | 1 a 13 |
| `n_loss_terms_nonzero` | 16 [16, 16] | 9 [9, 9] | - | - |
| `resid_conc_top10_sq` | 0.299 [0.271, 0.389] | 0.298 [0.242, 0.312] | 0.588 [0.509, 0.707] | 0.208 a 0.622 |
| `resid_conc_top10` | 0.188 [0.174, 0.228] | 0.196 [0.171, 0.199] | 0.342 [0.311, 0.403] | 0.16 a 0.345 |
| `resid_rms0` | 19.3 [11.9, 30.1] | 11.5 [4.27, 16] | 795 [524, 1.15e+03] | 0.0329 a 1.12 |
| `bc_over_pde` | 0.000192 [0.000142, 0.000976] | 0.00147 [0.00018, 0.0055] | 2.76e-07 [8.28e-08, 1.32e-06] | 0 a 593 |
| `gn_trace_bc_over_pde` | 0.000289 [0.000168, 0.00032] | 0.00231 [0.00223, 0.0025] | 1.29e-07 [9.62e-08, 1.98e-07] | 0 a 37.7 |
| `term_loss_ratio_max_min` | 2.52e+05 [2.11e+05, 1.3e+07] | 1.31e+302 [1.74e+301, 2.55e+302] | 1.04e+09 [7.94e+07, 6.97e+09] | 1.5e+03 a 1.65e+07 |
| `term_loss_ratio_max_min_nonzero` | 2.52e+05 [2.11e+05, 1.3e+07] | 1.37e+05 [7.63e+04, 7.57e+06] | - | - |
| `output_scale_ratio` | 2.18e+10 [2.18e+10, 2.18e+10] | 4.23e+10 [4.23e+10, 4.23e+10] | 2e+10 [2e+10, 2e+10] | 1 a 5 |
| `output_scale_ratio_uv_phi` | 4.23e+08 [4.23e+08, 4.23e+08] | 4.85e+06 [4.85e+06, 4.85e+06] | 3.33e+08 [3.33e+08, 3.33e+08] | - |
| `condB_median` | 8.32e+03 [8.07e+03, 8.35e+03] | 8.64e+03 [8.54e+03, 8.8e+03] | 8.68e+03 [8.41e+03, 9.27e+03] | 8.48e+03 a 1.01e+04 |
| `condB_min` | 1.98e+03 [1.8e+03, 1.99e+03] | 2.67e+03 [2.58e+03, 2.73e+03] | 2.05e+03 [1.89e+03, 2.24e+03] | 1.99e+03 a 4.15e+03 |
| `edge_trace_log10_p95_p5` | 3.59 [3.58, 3.88] | 3.82 [3.81, 4.02] | - | 1.68 a 3.46 |
| `edge_trace_log10_max_min` | 4.62 [4.56, 4.74] | 5.08 [5.08, 5.12] | - | 2.7 a 3.99 |
| `layer_trace_log10_ratio` | 2.37 [2.28, 2.59] | 1.46 [1.33, 1.58] | - | 1.91 a 3.04 |
| `loss0` | 373 [143, 906] | 131 [18.3, 255] | 6.32e+05 [2.75e+05, 1.33e+06] | 0.378 a 1.25 |
| `terms_vs_loss_relerr` | 0 [0, 1.99e-16] | 1.94e-16 [0, 2.16e-16] | 1.54e-16 [0, 3.17e-16] | 6.07e-08 a 8.54e-08 |

Perdida inicial por termino (mediana, converso | directo):

- `loss0_bc_bottom_sigma_yy`: 0.00331 | 0
- `loss0_bc_bottom_tau_xy`: 0.0029 | 0
- `loss0_bc_left_Dn`: 0.00901 | 0
- `loss0_bc_right_Dn`: 0.0109 | 0
- `loss0_bc_right_sigma_xx`: 0.0669 | -
- `loss0_bc_right_tau_xy`: 0.0019 | -
- `loss0_bc_right_traction_x`: - | 0
- `loss0_bc_right_traction_y`: - | 0
- `loss0_bc_top_sigma_yy`: 0.00223 | 0
- `loss0_bc_top_tau_xy`: 0.00365 | 0
- `loss0_bc_top_zero_net_charge`: - | 0.1
- `loss0_constitutive_D_x`: 0.0106 | 0.000127
- `loss0_constitutive_D_y`: 1.23 | 0.0282
- `loss0_constitutive_sigma_xx`: 0.357 | 0.0647
- `loss0_constitutive_sigma_yy`: 1.02 | 0.0722
- `loss0_constitutive_tau_xy`: 37.9 | 0.0287
- `loss0_equilibrium_x`: 0.0202 | 0.317
- `loss0_equilibrium_y`: 336 | 131
- `loss0_gauss`: 0.018 | 0.0281

## Lectura frente al stand-in anterior y al banco (n = 3 semillas del paper: 20260728-20260730; descriptivo; la medición previa con 4001-4003 dio lo mismo en orden de magnitud: `cpu_measure/properties_piezo_seeds4001-4003.csv`)

- **El stand-in se equivocaba en lo que dependía de escalas y pesos**, como advertía `cruce/predictions_DRAFT.md`. Con el problema real: borde/PDE en la pérdida **2e-4** (converso) y **1.5e-3** (directo) en vez de 2.8e-7; traza GN borde/PDE **2.9e-4** (converso) y **2.3e-3** (directo) en vez de 1.3e-7; disparidad entre términos **2.5e5** (converso, 16 términos no nulos) y **1.4e5** (directo, 9 términos no nulos; los 8 restantes son bordes duros con residuo exactamente 0) en vez de 1e9. Siguen lejos del borde-dominado de NS/DFG (borde/PDE 87-593): el borde pesa poco en ambos, como en AC/Burgers/KPP.
- **Concentración del residuo** (fracción del cuadrado en el 10 % de puntos): 0.30 en ambos, ya dentro del rango del banco (0.21-0.62), no el 0.59 del stand-in (que se parecía a KPP). La perdida inicial es dominada por un término: `equilibrium_y` (≈ 95 % de la pérdida inicial en ambos).
- **cond(B)** no separa (8.1e3-8.8e3, igual que el banco). **Dispersión de trazas entre aristas** (log10 p95/p5): 3.6 (converso) y 3.8 (directo) frente a 6.6 del stand-in y 1.7-3.5 del banco: sigue algo por encima del rango, pero ya no a 3 décadas de distancia; `edge_trace_log10_max_min` ≈4.6/5.1 frente a 2.7-4.0 (extrapola). Razón de escalas de salida físicas 2e10 (converso) y 4e10 (directo), igual que el stand-in y fuera del banco (1-5); es una propiedad de las unidades físicas (la red trabaja con salidas adimensionales y los bordes duros reescalan), y se conserva como extrapolación.
- **Acoplamiento**: 8 salidas, 16 (converso) / 17 (directo) términos, sin cambio. En el directo 8 términos de borde son identicamente 0 por construcción (tracciones y aislamiento en las salidas): solo la carga neta cero y los 8 términos de interior pesan.
- Conclusión práctica para `predictions.md`: de las propiedades que cambiaron, borde/PDE y la concentración ahora caen **del lado de AC/Burgers/KPP o intermedias**, no extremas; la dispersión de trazas entre aristas y la razón de escalas de salida siguen fuera del rango del banco (J, E, I extrapolan en ese eje). Las dos formulaciones difieren poco entre sí en estas propiedades (directo: más disparidad entre capas de trazas, menos acoplamiento activo), así que las propiedades del banco **no separan** converso de directo.
