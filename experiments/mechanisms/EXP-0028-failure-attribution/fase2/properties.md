# EXP-0028 Fase 2 - propiedades a la inicializacion (sin entrenar, CPU, float32)

Semillas 4001-4010 (n por problema: ac_d1e-3=10, ac_d2e-4=10, burgers=10, kpp=10, ns=10). Celda = mediana [min, max] entre semillas; '-' = no aplica. Modelo y puntos identicos al entrenamiento de EXP-0023 (RBF16 en AC/KPP/NS, RBF48 en Burgers, 11 bases, ancho 0.35; NS con ancho oculto 16, no fijado por el spec). Generado por `make_table.py` desde `properties.csv`; medido con `measure_properties.py`.

## (1) Condicionamiento de los bloques por arista (B_n + 1e-3 I) y de P

| propiedad | ac_d1e-3 | ac_d2e-4 | burgers | kpp | ns |
|---|---|---|---|---|---|
| `condB_min` | 3.43e+03 [2.95e+03, 3.81e+03] | 3.43e+03 [2.96e+03, 3.81e+03] | 3.24e+03 [2.68e+03, 3.46e+03] | 4.15e+03 [3.8e+03, 4.72e+03] | 2.04e+03 [1.86e+03, 2.18e+03] |
| `condB_median` | 9.26e+03 [8.66e+03, 9.64e+03] | 9.26e+03 [8.66e+03, 9.63e+03] | 8.48e+03 [8.42e+03, 8.54e+03] | 1.01e+04 [9.85e+03, 1.02e+04] | 7.21e+03 [6.88e+03, 8.08e+03] |
| `condB_max` | 1.09e+04 [1.08e+04, 1.09e+04] | 1.09e+04 [1.08e+04, 1.09e+04] | 1.08e+04 [1.07e+04, 1.09e+04] | 1.1e+04 [1.09e+04, 1.1e+04] | 1.09e+04 [1.08e+04, 1.09e+04] |
| `condB_l0_median` | 5.78e+03 [5.51e+03, 6.11e+03] | 5.77e+03 [5.49e+03, 6.09e+03] | 4.89e+03 [4.69e+03, 5.16e+03] | 6e+03 [5.54e+03, 6.39e+03] | 2.95e+03 [2.77e+03, 3.2e+03] |
| `condB_l1_median` | 9.33e+03 [8.78e+03, 9.67e+03] | 9.33e+03 [8.78e+03, 9.67e+03] | 8.51e+03 [8.45e+03, 8.57e+03] | 1.01e+04 [9.9e+03, 1.02e+04] | 7.12e+03 [6.79e+03, 7.99e+03] |
| `condB_l2_median` | 1.06e+04 [1.05e+04, 1.07e+04] | 1.06e+04 [1.05e+04, 1.07e+04] | 1.03e+04 [1.02e+04, 1.05e+04] | 1.08e+04 [1.08e+04, 1.09e+04] | 1.03e+04 [1.01e+04, 1.04e+04] |
| `P_sv_min` | 0.303 [0.302, 0.304] | 0.303 [0.302, 0.304] | 0.304 [0.304, 0.306] | 0.302 [0.302, 0.303] | 0.303 [0.302, 0.304] |
| `P_sv_max` | 31.6 [31.6, 31.6] | 31.6 [31.6, 31.6] | 31.6 [31.6, 31.6] | 31.6 [31.6, 31.6] | 31.6 [31.6, 31.6] |
| `P_cond_median` | 96.2 [93.1, 98.2] | 96.2 [93.1, 98.2] | 92.1 [91.7, 92.4] | 100 [99.2, 101] | 84.9 [83, 89.9] |
| `n_edges` | 304 | 304 | 2.45e+03 | 304 | 336 |
| `n_edges_dead` | 0 | 0 | 0 | 0 | 0 |

## (2) Gram uniforme de las 11 bases (covariable de F)

| propiedad | ac_d1e-3 | ac_d2e-4 | burgers | kpp | ns |
|---|---|---|---|---|---|
| `gram_cond_raw` | 2.48e+05 | 2.48e+05 | 2.48e+05 | 2.48e+05 | 2.48e+05 |
| `gram_cond_reg` | 4.38e+03 | 4.38e+03 | 4.38e+03 | 4.38e+03 | 4.38e+03 |

## (3) Concentracion del residuo inicial (0.10 = uniforme)

| propiedad | ac_d1e-3 | ac_d2e-4 | burgers | kpp | ns |
|---|---|---|---|---|---|
| `resid_conc_top10` | 0.288 [0.256, 0.31] | 0.289 [0.256, 0.31] | 0.16 [0.158, 0.198] | 0.345 [0.283, 0.699] | 0.195 [0.168, 0.235] |
| `resid_conc_top10_sq` | 0.428 [0.394, 0.444] | 0.429 [0.394, 0.445] | 0.208 [0.203, 0.305] | 0.622 [0.457, 0.976] | 0.308 [0.242, 0.409] |
| `resid_rms0` | 0.901 [0.886, 0.98] | 0.9 [0.885, 0.979] | 1.12 [1.11, 1.19] | 0.607 [0.409, 0.949] | 0.0775 [0.0633, 0.0953] |

## (4) Numeros adimensionales del PDE (constantes)

| propiedad | ac_d1e-3 | ac_d2e-4 | burgers | kpp | ns |
|---|---|---|---|---|---|
| `damkohler` | 5e+03 | 2.5e+04 | - | 1e+04 | - |
| `peclet` | - | - | 314 | 204 | 100 |
| `front_width` | 0.0141 | 0.00632 | 0.00318 | 0.0245 | - |
| `front_over_rbf` | 0.0404 | 0.0181 | 0.00909 | 0.07 | - |
| `rbf_width_x` | 0.35 | 0.35 | 0.35 | 0.35 | 0.7 |

## (5) Borde frente a PDE en la perdida inicial

| propiedad | ac_d1e-3 | ac_d2e-4 | burgers | kpp | ns |
|---|---|---|---|---|---|
| `bc_over_pde` | - | - | - | - | 87.2 [38, 203] |
| `loss0_pde` | - | - | - | - | 0.00602 [0.004, 0.00908] |
| `loss0_bc` | - | - | - | - | 0.506 [0.316, 0.909] |

## (6) Salidas y ecuaciones acopladas

| propiedad | ac_d1e-3 | ac_d2e-4 | burgers | kpp | ns |
|---|---|---|---|---|---|
| `n_outputs` | 1 | 1 | 1 | 1 | 3 |
| `n_coupled_eqs` | 1 | 1 | 1 | 1 | 3 |
| `n_loss_terms` | 1 | 1 | 1 | 1 | 8 |
| `model_width` | 16 | 16 | 48 | 16 | 16 |

## (7) Perdida y gradiente iniciales

| propiedad | ac_d1e-3 | ac_d2e-4 | burgers | kpp | ns |
|---|---|---|---|---|---|
| `loss0` | 0.812 [0.785, 0.96] | 0.81 [0.783, 0.959] | 1.25 [1.23, 1.42] | 0.378 [0.167, 0.9] | 0.515 [0.324, 0.913] |
| `grad_norm0` | 2.62 [0.261, 6.06] | 2.62 [0.256, 6.08] | 2.39 [1.95, 9.99] | 20 [0.79, 36.1] | 7.77 [2.45, 13.3] |
| `grad_norm0_coef` | 2.61 [0.258, 6.05] | 2.61 [0.253, 6.07] | 2.38 [1.94, 9.97] | 20 [0.789, 36] | 7.76 [2.44, 13.3] |

## Notas y definiciones

- cond(B) = cond de (B/m + 1e-3 I), el bloque que P factoriza (Enmienda 1); el B crudo es casi singular. Techo teorico ~ (11+1e-3)/1e-3 = 1.1e4, de modo que condB_max ya esta saturado en todos los problemas; lo que discrimina es la mediana y el minimo (NS el mas bajo, KPP el mas alto).
- Pool de bloques: 2048 puntos equiespaciados del pool de PDE (gp.subset), como el mecanismo. NS usa un residuo vectorial local (3 ecuaciones; comprobado contra prob.loss, error relativo 9e-8), porque gn_blocks solo admite residuo escalar. Solo PDE, sin terminos de borde (convencion actual del mecanismo).
- Numeros adimensionales (L = 1, T = 1): AC damkohler = a L^2/d; Burgers peclet = U L/nu (U=1); KPP damkohler = r L^2/D y peclet = c L/D (c = 5 sqrt(rD/6)); NS Re = U L/nu = 100. front_width: AC sqrt(d/a), Burgers nu/U, KPP 1/k. front_over_rbf = front_width / ancho RBF en x.
- AC d=1e-3 y 2e-4 comparten pool, semilla y modelo: difieren solo en d (y en el residuo). Sus propiedades de red casi coinciden; la diferencia esta en damkohler y front_over_rbf.
- AC/Burgers/KPP: IC y BC duras, sin terminos de borde en la perdida: bc_over_pde no aplica.
- NS: ancho oculto 16 elegido por mi (el spec no lo fija); `--ns-width` lo cambia.
- Hooks pendientes: ver HOOKS_PENDIENTES en measure_properties.py (DFG y piezo).
