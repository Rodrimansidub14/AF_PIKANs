# EXP-0030, bloque D: física (exploratorio, post hoc; sin umbrales)

## D1. Términos de pérdida sobre el *pool de entrenamiento* (con peso; mediana de los pasos > 19 000)

Razón por término: mediana entre semillas (rango). **No es el residual sobre la malla reservada** (ver D2). Hardware de M distinto al de E: la razón M/E está confundida con el hardware.

### converse

| término | E/B | M/E |
|---|---|---|
| constitutive_sigma_xx | 1.07 (0.84-1.25) | 1.58 (0.84-1.58) |
| constitutive_sigma_yy | 0.35 (0.33-0.83) | 2.09 (1.47-2.23) |
| constitutive_tau_xy | 0.49 (0.41-2.15) | 1.71 (1.55-1.81) |
| constitutive_D_x | 8.64 (1.43-14.67) | 0.17 (0.10-0.27) |
| constitutive_D_y | 1.22 (0.87-1.69) | 0.65 (0.57-0.72) |
| equilibrium_x | 0.17 (0.03-1.07) | 1.94 (1.65-16.03) |
| equilibrium_y | 0.09 (0.07-0.29) | 1.26 (0.53-3.40) |
| gauss | 1.24 (1.11-1.68) | 0.65 (0.56-0.73) |
| bc_top_sigma_yy | 0.90 (0.53-26.68) | 0.01 (0.01-0.06) |
| bc_top_tau_xy | 1.06 (0.14-4.15) | 1.49 (0.13-4.23) |
| bc_bottom_sigma_yy | 1.84 (0.18-41.97) | 0.01 (0.01-0.08) |
| bc_bottom_tau_xy | 0.23 (0.05-0.51) | 3.83 (2.00-4.76) |
| bc_right_sigma_xx | 1.35 (0.10-118.14) | 29.12 (3.05-29.33) |
| bc_right_tau_xy | 0.17 (0.10-1.61) | 3.16 (1.71-14.45) |
| bc_right_Dn | 0.64 (0.22-7.68) | 0.92 (0.76-1.27) |
| bc_left_Dn | 2.28 (0.26-14.63) | 0.27 (0.26-0.37) |

### direct

| término | E/B | M/E |
|---|---|---|
| constitutive_sigma_xx | 0.51 (0.16-0.72) | 1.08 (0.93-3.61) |
| constitutive_sigma_yy | 1.01 (0.46-1.23) | 0.76 (0.74-2.11) |
| constitutive_tau_xy | 0.51 (0.19-1.56) | 0.93 (0.56-2.66) |
| constitutive_D_x | 0.73 (0.44-1.29) | 1.20 (0.56-2.39) |
| constitutive_D_y | 0.66 (0.52-1.12) | 1.10 (0.68-1.34) |
| equilibrium_x | 0.30 (0.11-0.42) | 2.37 (0.67-5.88) |
| equilibrium_y | 0.49 (0.12-0.59) | 1.39 (0.72-2.99) |
| gauss | 0.41 (0.23-1.34) | 1.21 (0.33-3.36) |
| bc_top_sigma_yy | 0 en todas las corridas | 0 en todas las corridas |
| bc_top_tau_xy | 0 en todas las corridas | 0 en todas las corridas |
| bc_bottom_sigma_yy | 0 en todas las corridas | 0 en todas las corridas |
| bc_bottom_tau_xy | 0 en todas las corridas | 0 en todas las corridas |
| bc_right_traction_x | 0 en todas las corridas | 0 en todas las corridas |
| bc_right_traction_y | 0 en todas las corridas | 0 en todas las corridas |
| bc_right_Dn | 0 en todas las corridas | 0 en todas las corridas |
| bc_left_Dn | 0 en todas las corridas | 0 en todas las corridas |
| bc_top_equipotential | 0 en todas las corridas | 0 en todas las corridas |
| bc_top_zero_net_charge | 0.58 (0.12-2.53) | 5.17 (4.18-16.67) |

## D2. Residual por ecuación sobre la malla reservada (RMS del residual guardado en el estado final, `resid_grid`)

### converse

| ecuación o borde | E/B | M/E |
|---|---|---|
| constitutive_sigma_xx | 1.01 (0.95-1.12) | 1.14 (0.85-1.24) |
| constitutive_sigma_yy | 0.84 (0.79-1.78) | 1.12 (0.98-1.12) |
| constitutive_tau_xy | 1.09 (0.28-3.41) | 1.74 (0.86-2.46) |
| constitutive_D_x | 3.16 (2.51-6.49) | 0.26 (0.14-0.58) |
| constitutive_D_y | 1.18 (0.79-1.25) | 0.82 (0.79-0.86) |
| equilibrium_x | 0.41 (0.15-5.32) | 2.01 (1.42-3.61) |
| equilibrium_y | 0.37 (0.07-3.13) | 1.15 (0.89-1.26) |
| gauss | 1.20 (1.12-1.28) | 0.81 (0.79-0.86) |
| bc_top_sigma_yy | 0.56 (0.52-53.89) | 0.20 (0.02-0.30) |
| bc_top_tau_xy | 0.54 (0.17-3.54) | 2.34 (0.48-3.34) |
| bc_bottom_sigma_yy | 0.72 (0.53-49.50) | 0.18 (0.02-0.20) |
| bc_bottom_tau_xy | 0.38 (0.12-4.70) | 2.54 (2.07-4.43) |
| bc_right_sigma_xx | 0.93 (0.16-40.51) | 6.22 (1.44-7.65) |
| bc_right_tau_xy | 0.25 (0.13-10.72) | 3.92 (3.19-7.01) |
| bc_right_Dn | 0.96 (0.17-5.16) | 1.12 (1.03-1.73) |
| bc_left_Dn | 2.00 (0.43-4.49) | 0.60 (0.48-0.85) |

### direct

| ecuación o borde | E/B | M/E |
|---|---|---|
| constitutive_sigma_xx | 0.71 (0.62-0.72) | 1.03 (0.84-7.42) |
| constitutive_sigma_yy | 0.63 (0.56-11.01) | 1.27 (0.15-21.43) |
| constitutive_tau_xy | 0.78 (0.68-15.16) | 1.15 (0.20-3.20) |
| constitutive_D_x | 0.96 (0.82-1.17) | 1.07 (0.92-1.52) |
| constitutive_D_y | 0.60 (0.47-12.26) | 1.13 (0.13-12.05) |
| equilibrium_x | 1.00 (1.00-1.01) | 1.00 (0.99-1.00) |
| equilibrium_y | 1.47 (1.22-2.02) | 1.10 (0.45-1.19) |
| gauss | 1.75 (0.50-8.61) | 1.23 (0.12-2.78) |
| bc_top_sigma_yy | 0 en todas las corridas | 0 en todas las corridas |
| bc_top_tau_xy | 0 en todas las corridas | 0 en todas las corridas |
| bc_bottom_sigma_yy | 0 en todas las corridas | 0 en todas las corridas |
| bc_bottom_tau_xy | 0 en todas las corridas | 0 en todas las corridas |
| bc_right_traction_x | 0 en todas las corridas | 0 en todas las corridas |
| bc_right_traction_y | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) |
| bc_right_Dn | 0 en todas las corridas | 0 en todas las corridas |
| bc_left_Dn | 0 en todas las corridas | 0 en todas las corridas |
| bc_top_zero_net_charge | 1.13 (0.79-1.50) | 1.28 (0.58-1.28) |

### Prueba de cordura: recalcular `terms_end` desde el estado guardado (CPU, float64)

| problema | brazo | semilla | estado carga sin claves de más o de menos | pérdida recalculada | pérdida guardada (loss_end) | máx. diferencia relativa por término |
|---|---|---|---|---|---|---|
| conv | M | 28 | sí | 0.0952462 | 0.0952462 | 3.32e-13 |
| dire | E | 29 | sí | 0.0164999 | 0.0164999 | 3.37e-12 |

## D3. Error por campo, flujo e interfaz (valores del JSON; mediana entre semillas)

| problema | brazo | n | l2re u | l2re v | l2re φ | flujo rel. L2: σxx / σyy / τ / Dx / Dy | interface_gap |
|---|---|---|---|---|---|---|---|
| conv | B | 5 | 0.135 | 0.115 | 0.046 | 0.98 / 1.02 / 1.72 / 35.02 / 0.01 | 0 |
| conv | E | 5 | 0.077 | 0.067 | 0.051 | 1.01 / 1.03 / 1.04 / 38.55 / 0.01 | 0 |
| conv | M | 3 | 0.166 | 0.207 | 0.044 | 1.01 / 1.00 / 1.21 / 15.54 / 0.01 | 0 |
| dire | B | 3 | 0.204 | 0.228 | 0.247 | 0.71 / 1.00 / 0.17 / 2.49 / 0.39 | 0 |
| dire | E | 3 | 0.198 | 0.220 | 0.216 | 0.46 / 1.00 / 0.14 / 2.18 / 0.44 | 0 |
| dire | M | 3 | 0.178 | 0.192 | 0.215 | 0.47 / 1.00 / 0.14 / 2.65 / 0.36 | 0 |

## D4. Chequeos de admisibilidad (valores, sin umbral)

Consistencia constitutiva = error relativo L2 entre las tensiones y D que predice la red y las reconstruidas desde los gradientes de u, v y φ, sobre la malla reservada (5 magnitudes, en el orden que guarda `consistency_grid/rel_l2`). Carga neta: término `bc_top_zero_net_charge` del pool (directo). Trazas de borde: media de los términos `bc_*` del pool. Simetrías de geometría y polarización: **no se calcularon**; la lista no se cerró (pendiente de la spec).

| problema | brazo | n | consistencia constitutiva (5 magnitudes) | media de términos de borde | carga neta cero |
|---|---|---|---|---|---|
| conv | B | 5 | 0.96 / 1.00 / 1.00 / 1.59 / 0.17 | 7.99e-07 | n/a |
| conv | E | 5 | 1.01 / 1.00 / 1.00 / 1.04 / 0.20 | 8.61e-07 | n/a |
| conv | M | 3 | 0.97 / 1.00 / 1.00 / 1.05 / 0.17 | 6.21e-07 | n/a |
| dire | B | 3 | 1.05 / 1.00 / 1.00 / 2.69 / 0.78 | 0.00e+00 | 8.04e-08 |
| dire | E | 3 | 0.79 / 1.00 / 1.00 / 2.24 / 0.84 | 0.00e+00 | 7.74e-08 |
| dire | M | 3 | 0.75 / 1.00 / 1.00 / 2.63 / 0.93 | 0.00e+00 | 4.00e-07 |