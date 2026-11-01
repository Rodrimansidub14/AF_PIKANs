# EXP-0028 Fase 2 (ampliacion) - propiedades con `residual_terms` (PDE + borde + pesos), sin entrenar, CPU

Semillas 4001-4010, n = 10 por problema. Celda = mediana [min, max]. Generado por `make_table_terms.py` desde `properties_terms.csv`, medido con `measure_properties_terms.py`.
Modelo, semilla, pools y puntos del GN identicos a la fase 1c (`run_phase1c.py`, `run_phase1c_dfg.py` importados sin modificar; GN = 2048 por clave del pool, bloques por arista, chunk 16).
- `ns_terms`: NS2d-CG repetido con los 8 terminos (3 PDE + 5 de borde, peso 1); la fase 2 original (`properties.md`, columna `ns`) usaba solo PDE para cond(B) (por eso `pde_condB_*` debe coincidir con ella: 7.2e3 frente a 7.21e3).
- `dfg`: DFG 2D-1 Re20 con `DFGTerms` (13 terminos: 3 PDE + inlet 2 + paredes y cilindro 6 + salida 2; pesos 1 y 1/2), modelo `DFGRBFPIKAN` con `output_scales` dentro del forward, pools completos de `generate_dfg_pools` (interior 16 384, 512 por frontera).
- `piezo_standin`: **NO es el problema del paper.** Es la reimplementacion propia `experiments/piezo/EXP-0009-piezo-indirect-rbf/piezo_pinn.py` (2 entradas xi=x/L, eta=y/H; 8 salidas; bordes duros en u, v, phi; escalas SC_*; terminos constitutivos + equilibrio + libres de traccion; pesos 1, sin pesos adaptativos, sin enrutado stop-gradient, sin caracteristicas de fase), RBF16 de 11 bases, float64, 2048 interiores + 256 por borde (receta del paper). El adaptador `piezo_terms` esta en `measure_properties_terms.py`; reproduce `loss_fn` de `piezo_pinn.py` (error relativo ~1e-16). Sirve para medir propiedades de la fisica y del reparto de escalas, no para predecir el codigo de Daniel.

| propiedad | ns_terms | dfg | piezo_standin |
|---|---|---|---|
| `condB_l0_median` | 2.99e+03 [2.66e+03, 3.09e+03] | 3.64e+03 [3.12e+03, 3.72e+03] | 4.38e+03 [3.18e+03, 5.13e+03] |
| `condB_l1_median` | 8.48e+03 [8e+03, 8.75e+03] | 9.64e+03 [9.48e+03, 9.84e+03] | 8.13e+03 [7.86e+03, 8.9e+03] |
| `condB_l2_median` | 1.04e+04 [1.03e+04, 1.05e+04] | 1.08e+04 [1.07e+04, 1.08e+04] | 1.04e+04 [1.02e+04, 1.06e+04] |
| `condB_min` | 2e+03 [1.92e+03, 2.19e+03] | 1.99e+03 [1.88e+03, 2.14e+03] | 2.05e+03 [1.89e+03, 2.24e+03] |
| `condB_median` | 8.56e+03 [8.11e+03, 8.78e+03] | 9.69e+03 [9.58e+03, 9.91e+03] | 8.68e+03 [8.41e+03, 9.27e+03] |
| `condB_max` | 1.09e+04 [1.08e+04, 1.09e+04] | 1.1e+04 [1.09e+04, 1.1e+04] | 1.09e+04 [1.08e+04, 1.09e+04] |
| `P_sv_min` | 0.303 [0.302, 0.304] | 0.302 [0.302, 0.302] | 0.303 [0.302, 0.304] |
| `P_sv_max` | 31.6 [31.6, 31.6] | 31.6 [31.6, 31.6] | 31.6 [31.6, 31.6] |
| `P_cond_median` | 92.5 [90.1, 93.7] | 98.4 [97.9, 99.5] | 93.2 [91.7, 96.3] |
| `n_edges` | 336 | 336 | 416 |
| `n_edges_dead` | 0 | 0 | 0 |
| `pde_condB_min` | 2.04e+03 [1.86e+03, 2.18e+03] | 1.89e+03 [1.57e+03, 2.02e+03] | 2.05e+03 [1.89e+03, 2.24e+03] |
| `pde_condB_median` | 7.21e+03 [6.88e+03, 8.08e+03] | 7.4e+03 [7.05e+03, 8.05e+03] | 8.5e+03 [8.16e+03, 9.23e+03] |
| `pde_condB_max` | 1.09e+04 [1.08e+04, 1.09e+04] | 1.09e+04 [1.08e+04, 1.09e+04] | 1.09e+04 [1.08e+04, 1.09e+04] |
| `pde_P_cond_median` | 84.9 [83, 89.9] | 86 [84, 89.7] | 92.2 [90.4, 96.1] |
| `gn_trace_bc_over_pde` | 7.74 [6.3, 8.08] | 37.7 [30.1, 45.7] | 1.29e-07 [9.62e-08, 1.98e-07] |
| `gn_trace_pde` | 26.8 [25.9, 34.2] | 5.57 [4.54, 7.12] | 2.55e+09 [1.66e+09, 3.49e+09] |
| `gn_trace_bc` | 208 [200, 216] | 208 [200, 216] | 328 [312, 337] |
| `gram_cond_raw` | 2.48e+05 | 2.48e+05 | 2.48e+05 |
| `gram_cond_reg` | 4.38e+03 | 4.38e+03 | 4.38e+03 |
| `resid_conc_top10` | 0.195 [0.168, 0.235] | 0.206 [0.18, 0.226] | 0.342 [0.311, 0.403] |
| `resid_conc_top10_sq` | 0.308 [0.242, 0.409] | 0.334 [0.269, 0.39] | 0.588 [0.509, 0.707] |
| `resid_rms0` | 0.0775 [0.0633, 0.0953] | 0.0329 [0.027, 0.0419] | 795 [524, 1.15e+03] |
| `n_pool_points` | 8.19e+03 | 1.64e+04 | 2.05e+03 |
| `resid_conc_all_sq_top10` | 0.985 [0.97, 0.994] | 0.999 [0.997, 0.999] | 0.984 [0.978, 0.99] |
| `loss0_terms` | 0.515 [0.324, 0.913] | 0.606 [0.474, 0.891] | 6.32e+05 [2.75e+05, 1.33e+06] |
| `loss0_pde` | 0.00602 [0.004, 0.00908] | 0.00108 [0.000731, 0.00175] | 6.32e+05 [2.75e+05, 1.33e+06] |
| `loss0_bc` | 0.506 [0.316, 0.909] | 0.606 [0.472, 0.891] | 0.125 [0.102, 0.813] |
| `bc_over_pde` | 87.2 [38, 203] | 593 [315, 1.22e+03] | 2.76e-07 [8.28e-08, 1.32e-06] |
| `n_loss_terms` | 8 | 13 | 16 |
| `term_loss_ratio_max_min` | 1.5e+03 [159, 4.59e+03] | 1.65e+07 [4.46e+06, 2.62e+08] | 1.04e+09 [7.94e+07, 6.97e+09] |
| `loss0` | 0.515 [0.324, 0.913] | 0.606 [0.474, 0.891] | 6.32e+05 [2.75e+05, 1.33e+06] |
| `terms_vs_loss_relerr` | 6.07e-08 [6e-10, 1.53e-07] | 8.54e-08 [2.47e-09, 1.94e-07] | 1.54e-16 [0, 3.17e-16] |
| `grad_norm0` | 7.77 [2.45, 13.3] | 6.05 [1.73, 11.5] | 2.74e+07 [1.58e+07, 6.45e+07] |
| `grad_norm0_coef` | 7.76 [2.44, 13.3] | 6.04 [1.73, 11.5] | 2.72e+07 [1.57e+07, 6.41e+07] |
| `model_width` | 16 | 16 | 16 |
| `n_outputs` | 3 | 3 | 8 |
| `n_coupled_eqs` | 3 | 3 | 8 |
| `output_scale_ratio` | 1 | 5 | 2e+10 |
| `peclet` | 100 | 20 | - |
| `rbf_width_x` | 0.35 | 0.35 | 0.35 |
| `output_scale_ratio_uv_phi` | - | - | 3.33e+08 |

**Perdida inicial por termino (w_t * media r_t^2, pool completo)**

| termino | ns_terms | dfg | piezo_standin |
|---|---|---|---|
| `loss0_momentum_x` | 0.00058 [0.000204, 0.00185] | 3.59e-05 [1.85e-05, 6.09e-05] | - |
| `loss0_momentum_y` | 0.00166 [0.000705, 0.00284] | 0.000386 [0.000182, 0.000644] | - |
| `loss0_continuity` | 0.00399 [0.00117, 0.00663] | 0.000721 [0.000204, 0.00131] | - |
| `loss0_inlet_u` | 0.464 [0.233, 0.835] | 0.556 [0.354, 0.803] | - |
| `loss0_inlet_v` | 0.00789 [0.000279, 0.0352] | 0.00419 [0.000391, 0.0201] | - |
| `loss0_outlet_p` | 0.0155 [0.000182, 0.0368] | - | - |
| `loss0_other_u` | 0.0159 [0.00106, 0.0709] | - | - |
| `loss0_other_v` | 0.00643 [0.000302, 0.0412] | - | - |
| `loss0_lower_wall_u` | - | 0.00787 [0.000912, 0.0378] | - |
| `loss0_lower_wall_v` | - | 0.00384 [0.000284, 0.0173] | - |
| `loss0_upper_wall_u` | - | 0.00747 [0.000134, 0.0297] | - |
| `loss0_upper_wall_v` | - | 0.00301 [8.91e-05, 0.0266] | - |
| `loss0_cylinder_u` | - | 0.00779 [0.000173, 0.0426] | - |
| `loss0_cylinder_v` | - | 0.0034 [0.000201, 0.0213] | - |
| `loss0_outlet_sx` | - | 0.0155 [0.000181, 0.0368] | - |
| `loss0_outlet_sy` | - | 3.36e-08 [2.69e-09, 1.17e-07] | - |
| `loss0_sx` | - | - | 4.89e+04 [2.13e+04, 1.03e+05] |
| `loss0_sz` | - | - | 5.82e+05 [2.53e+05, 1.23e+06] |
| `loss0_tau` | - | - | 5.8 [2.28, 23.5] |
| `loss0_Dx` | - | - | 0.011 [0.000369, 0.0792] |
| `loss0_Dy` | - | - | 115 [52.9, 250] |
| `loss0_div1` | - | - | 64.8 [20.3, 257] |
| `loss0_div2` | - | - | 95.5 [29.1, 357] |
| `loss0_divD` | - | - | 60.3 [25, 164] |
| `loss0_right_sx` | - | - | 0.00363 [0.000172, 0.15] |
| `loss0_right_tau` | - | - | 0.0137 [0.00452, 0.283] |
| `loss0_right_Dx` | - | - | 0.00915 [6.09e-05, 0.0742] |
| `loss0_left_Dx` | - | - | 0.0125 [9.16e-05, 0.0912] |
| `loss0_bot_sz` | - | - | 0.0093 [0.000253, 0.069] |
| `loss0_bot_tau` | - | - | 0.0156 [0.00131, 0.273] |
| `loss0_top_sz` | - | - | 0.0107 [0.000922, 0.0713] |
| `loss0_top_tau` | - | - | 0.0166 [0.00441, 0.243] |

## Definiciones

- `condB_*`, `P_*`: como la fase 2 original (B/m + 1e-3 I por arista), pero con B = suma sobre TODOS los terminos (con pesos); `pde_condB_*` solo con los terminos de interior.
- `gn_trace_bc_over_pde`: traza total de los bloques GN de los terminos de borde / los de interior (cuanto del precondicionador lo dicta el borde).
- `resid_conc_top10(_sq)`: fraccion de la masa de sqrt(sum_k r_k^2) (o r^2) en el 10 % de puntos de interior con mayor valor, pool completo; `resid_conc_all_sq_top10`: con la contribucion w_t r^2/N_t de todos los puntos de todos los terminos (muy alta = la perdida inicial la dicta un grupo pequeno de puntos/terminos).
- `bc_over_pde`, `loss0_*`: perdida inicial por terminos con pesos, pool completo. `term_loss_ratio_max_min`: max/min de la perdida por termino (disparidad de escalas entre terminos). `terms_vs_loss_relerr`: adaptador frente a la perdida de entrenamiento (en float32 ~1e-7 es redondeo).
- `output_scale_ratio`: max/min de las escalas caracteristicas de las salidas (unidades mezcladas); piezo: u 3e-7, v 3.3e-5, phi 100 V, tensiones 6e3, D 1e-5; `output_scale_ratio_uv_phi` = 100 V / 3e-7 m. DFG: U_MEAN/PRESSURE_SCALE; NS: 1 (sin escalas).
- `peclet` = Re = U L / nu (NS: 100; DFG: U_MEAN D / nu = 0.2*0.1/0.001 = 20). Piezo: no aplica (sin adveccion).
- `edge_trace_*`, `layer_trace_*`: solo 3 semillas, ver `properties_edge_scale.csv`.
