# Preguntas diagnósticas Q1–Q6 — tablas descriptivas (EXP-0013)

Exploratorio. n = 5 semillas por celda; medianas e intervalos por semilla, no inferencia. `audit/` y `err/` usan la referencia; el resto no.

## Q1 — ¿Dónde está el error y dónde está el residuo?

Mediana por (problema, brazo, etapa) del Spearman residuo–error, solape de los 5 % superiores, y concentración (masa / fracción de puntos) del error y del residuo en la región steep5 (5 % de puntos con mayor gradiente de la referencia). Un solape esperado por azar es 0.05.

| problem       | arm                          | stage       |   n |   spearman_res_err_median |   top5_overlap_median |   steep5_err_conc |   steep5_res_conc |
|:--------------|:-----------------------------|:------------|----:|--------------------------:|----------------------:|------------------:|------------------:|
| allen_cahn_1d | mlp                          | adam_final  |   5 |                     0.464 |                 0     |             0.345 |             2.099 |
| allen_cahn_1d | mlp                          | lbfgs_final |   5 |                     0.355 |                 0.16  |             0.023 |             3.31  |
| allen_cahn_1d | rbf                          | adam_final  |   5 |                     0.313 |                 0.05  |             0.013 |             1.034 |
| allen_cahn_1d | rbf                          | lbfgs_final |   5 |                     0.167 |                 0.169 |             0.035 |             1.323 |
| burgers_1d    | mlp                          | adam_final  |  15 |                     0.301 |                 0.163 |            10.107 |            15.152 |
| burgers_1d    | mlp                          | lbfgs_final |  15 |                     0.052 |                 0.194 |            14.989 |            18.844 |
| burgers_1d    | rbf                          | adam_final  |  20 |                     0.471 |                 0.393 |            13.839 |            17.437 |
| burgers_1d    | rbf                          | lbfgs_final |  20 |                     0.067 |                 0.271 |            18.89  |            18.511 |
| dfg_2d        | mlp                          | adam_final  |  20 |                     0.299 |                 0.512 |             5.884 |             5.999 |
| dfg_2d        | mlp                          | lbfgs_final |  20 |                     0.361 |                 0.549 |             5.534 |             9.254 |
| dfg_2d        | rbf                          | adam_final  |  25 |                     0.448 |                 0.593 |             6.826 |            11.515 |
| dfg_2d        | rbf                          | lbfgs_final |  25 |                     0.434 |                 0.472 |             5.982 |            13.628 |
| heat_1d       | mlp                          | adam_final  |   5 |                     0.065 |                 0.028 |             0.162 |             1.841 |
| heat_1d       | rbf                          | adam_final  |  10 |                     0.311 |                 0.155 |             0.002 |             0.222 |
| heat_1d       | rbf                          | lbfgs_final |   5 |                    -0.004 |                 0.073 |             0.497 |             1.063 |
| heat_1d       | rbf(ctl-replicate)           | adam_final  |   5 |                     0.262 |                 0.124 |             0.001 |             0.161 |
| heat_1d       | rbf+rad                      | adam_final  |   5 |                     0.022 |                 0.066 |             0.168 |             0.866 |
| heat_1d       | rbf+rad                      | lbfgs_final |   5 |                     0.009 |                 0.077 |             0.141 |             1.049 |
| heat_1d       | rbf+rad(adam)->uniform-lbfgs | lbfgs_final |   5 |                    -0.049 |                 0.022 |             0.395 |             1.2   |
| ns2d_cg       | mlp                          | adam_final  |   5 |                     0.167 |                 0.149 |             1.467 |             9.102 |
| ns2d_cg       | rbf                          | adam_final  |   8 |                    -0.282 |                 0.191 |             1.696 |             9.632 |
| piezo_2d      | mlp                          | adam_final  |   5 |                    -0.574 |                 0     |             2.043 |             0.75  |
| piezo_2d      | rbf                          | adam_final  |   5 |                    -0.399 |                 0     |             1.789 |             1.119 |

## Q2 — ¿Capacidad u optimización?

Spearman entre la pérdida final registrada y el error (dentro de cada problema y etapa, sin quitar el efecto de brazo ni campaña): si la pérdida ordena el error, la optimización de la pérdida limita el resultado; si no, la pérdida no es un buen indicador del error.

| problem       | stage       | loss_column          |   n |   spearman_loss_vs_error |
|:--------------|:------------|:---------------------|----:|-------------------------:|
| allen_cahn_1d | adam_final  | train/adam_loss_end  |  10 |                    0.879 |
| allen_cahn_1d | lbfgs_final | train/adam_loss_end  |  10 |                    0.867 |
| burgers_1d    | adam_final  | train/adam_loss_end  |  35 |                    0.515 |
| burgers_1d    | lbfgs_final | train/adam_loss_end  |  35 |                   -0.011 |
| dfg_2d        | adam_final  | train/adam_loss_end  |  30 |                    0.939 |
| dfg_2d        | adam_final  | train/lbfgs_loss_end |  30 |                    0.82  |
| dfg_2d        | lbfgs_final | train/adam_loss_end  |  30 |                    0.822 |
| dfg_2d        | lbfgs_final | train/lbfgs_loss_end |  30 |                    0.933 |
| heat_1d       | adam_final  | train/adam_loss_end  |  25 |                    0.492 |
| ns2d_cg       | adam_final  | train/adam_loss_end  |  13 |                    0.247 |

Capacidad (RBF): ver `q2_capacity_vs_optimization.csv` (rango efectivo del jacobiano por parámetro, bases muertas, parámetros). Evidencia externa a este estudio: el ajuste directo del RBF de 4 032 parámetros al campo FEM de DFG llega a 1–5 % (`local-20260930/capacity-fit-dfg.json`), por lo que el límite en DFG es de optimización y no de capacidad.

## Q3 — ¿Qué cambia de Adam a L-BFGS?

Pareado por (campaña, brazo, semilla). `err_ratio_median` = error tras L-BFGS / error al final de Adam; el CV es la dispersión relativa entre semillas del error en cada etapa.

| problem       | cgroup                     | variant   | arm_w   |   n |   err_ratio_median |   res_rms_ratio |   res_gini_delta |   res_top5_delta |   spearman_delta |   err_cv_adam_final |   err_cv_lbfgs_final |
|:--------------|:---------------------------|:----------|:--------|----:|-------------------:|----------------:|-----------------:|-----------------:|-----------------:|--------------------:|---------------------:|
| allen_cahn_1d | ac-rerun-20260930          |           | mlp     |   5 |             0.0928 |          0.0693 |          -0.0528 |          -0.0693 |          -0.1298 |              0.5066 |               0.4934 |
| allen_cahn_1d | ac-rerun-20260930          |           | rbf@16  |   5 |             0.0717 |          0.1847 |          -0.0034 |           0.0857 |          -0.135  |              0.2614 |               0.3555 |
| burgers_1d    | burgers-big-20261003       |           | mlp     |   5 |             0.0182 |          0.055  |          -0.0287 |          -0.1799 |          -0.2587 |              0.3714 |               0.2139 |
| burgers_1d    | burgers-big-20261003       |           | rbf@48  |   5 |             0.0082 |          0.0353 |          -0.0538 |          -0.144  |          -0.5154 |              0.3633 |               0.6773 |
| burgers_1d    | burgers-close-20261003     |           | mlp     |   5 |             0.0243 |          0.1202 |           0.159  |           0.1649 |          -0.1785 |              0.2612 |               1.0107 |
| burgers_1d    | burgers-close-20261003     |           | rbf@16  |   5 |             0.12   |          0.2972 |           0.0085 |           0.0225 |          -0.2813 |              1.8099 |               1.2924 |
| burgers_1d    | burgers-close-20261003     |           | rbf@48  |   5 |             0.0292 |          0.2181 |          -0.0185 |           0.0327 |          -0.4033 |              0.6572 |               0.5801 |
| burgers_1d    | burgers-f64-lbfgs-20261003 |           | mlp     |   5 |             0.018  |          0.1125 |           0.1793 |           0.1643 |          -0.2327 |              0.2612 |               0.8317 |
| burgers_1d    | burgers-f64-lbfgs-20261003 |           | rbf@48  |   5 |             0.0205 |          0.2004 |           0.0096 |           0.0388 |          -0.3983 |              0.6572 |               0.5596 |
| dfg_2d        | cohort-A                   |           | rbf     |   5 |             0.5212 |          0.6417 |          -0.123  |           0.0397 |          -0.1373 |              0.128  |               0.1488 |
| dfg_2d        | cohort-B                   |           | mlp     |   5 |             0.0931 |          0.4807 |          -0.0157 |           0.0749 |          -0.0586 |              0.6507 |               1.5714 |
| dfg_2d        | cohort-B                   |           | rbf     |   5 |             0.4496 |          0.5366 |          -0.1417 |           0.0194 |          -0.1223 |              0.0659 |               0.0776 |
| dfg_2d        | dfg-daniel-cfg-20260930    | corrected | mlp     |   5 |             0.9919 |          0.928  |           0.0962 |           0.2627 |           0.0439 |              0.0134 |               0.0259 |
| dfg_2d        | dfg-daniel-cfg-20260930    | corrected | rbf     |   5 |             0.9161 |          1.0855 |           0.0632 |           0.099  |          -0.068  |              0.0271 |               0.0242 |
| dfg_2d        | dfg-daniel-cfg-20260930    | literal   | mlp     |   5 |             0.9597 |          1.0024 |           0.1191 |           0.2795 |           0.0221 |              0.0387 |               0.0134 |
| dfg_2d        | dfg-daniel-cfg-20260930    | literal   | rbf     |   5 |             0.9447 |          1.0332 |           0.0545 |           0.1528 |           0.0581 |              0.0216 |               0.0175 |
| dfg_2d        | dfg-long-20261001          | corrected | mlp     |   5 |             0.6734 |          0.9767 |           0.0805 |           0.1394 |           0.0612 |              0.0907 |               0.195  |
| dfg_2d        | dfg-long-20261001          | corrected | rbf     |   5 |             0.4306 |          1.2838 |          -0.0867 |           0.0573 |           0.1579 |              0.0268 |               0.1675 |
| heat_1d       | EXP-0011                   |           | rbf+rad |   5 |             0.9559 |          0.9446 |          -0.007  |          -0.0145 |          -0.0127 |              1.0984 |               0.158  |
| heat_1d       | heat-S2-control            |           | rbf     |   5 |             0.0115 |          0.3853 |           0.0487 |           0.3281 |          -0.3023 |              0.8199 |               0.097  |

## Q4 — Modos de falla (reglas fijas en `make_cards.py`; no causales)

| problem       | arm                          | stage       |   n |   budget_limited |   concentrated_error |   residual_aligned |   plateau_adam |   low_jacobian_rank_per_param |   null_state |   residual_misaligned |
|:--------------|:-----------------------------|:------------|----:|-----------------:|---------------------:|-------------------:|---------------:|------------------------------:|-------------:|----------------------:|
| allen_cahn_1d | mlp                          | adam_final  |   5 |                1 |                    4 |                  1 |              0 |                             0 |            0 |                     0 |
| allen_cahn_1d | mlp                          | lbfgs_final |   5 |                1 |                    5 |                  0 |              0 |                             0 |            0 |                     0 |
| allen_cahn_1d | rbf                          | adam_final  |   5 |                0 |                    5 |                  1 |              3 |                             0 |            0 |                     0 |
| allen_cahn_1d | rbf                          | lbfgs_final |   5 |                0 |                    5 |                  0 |              0 |                             0 |            0 |                     0 |
| burgers_1d    | mlp                          | adam_final  |  15 |               10 |                   10 |                  2 |              6 |                            15 |            0 |                     0 |
| burgers_1d    | mlp                          | lbfgs_final |  15 |               10 |                   11 |                  0 |              0 |                            15 |            0 |                     0 |
| burgers_1d    | rbf                          | adam_final  |  20 |               10 |                   16 |                  7 |              9 |                            15 |            0 |                     0 |
| burgers_1d    | rbf                          | lbfgs_final |  20 |               10 |                   17 |                  0 |              0 |                            15 |            0 |                     0 |
| dfg_2d        | mlp                          | adam_final  |  20 |                5 |                   16 |                  1 |              3 |                            20 |           11 |                     1 |
| dfg_2d        | mlp                          | lbfgs_final |  20 |                5 |                   16 |                  2 |              0 |                            19 |           10 |                     0 |
| dfg_2d        | rbf                          | adam_final  |  25 |                5 |                   15 |                  3 |              4 |                             0 |            0 |                     0 |
| dfg_2d        | rbf                          | lbfgs_final |  25 |                5 |                   15 |                  9 |              0 |                             0 |            0 |                     0 |
| heat_1d       | mlp                          | adam_final  |   5 |                0 |                    0 |                  1 |              1 |                             0 |            0 |                     3 |
| heat_1d       | rbf                          | adam_final  |  10 |                0 |                    0 |                  2 |              6 |                             0 |            0 |                     2 |
| heat_1d       | rbf                          | lbfgs_final |   5 |                0 |                    0 |                  0 |              0 |                             0 |            0 |                     4 |
| heat_1d       | rbf(ctl-replicate)           | adam_final  |   5 |                0 |                    0 |                  1 |              3 |                             0 |            0 |                     1 |
| heat_1d       | rbf+rad                      | adam_final  |   5 |                0 |                    0 |                  0 |              4 |                             2 |            0 |                     3 |
| heat_1d       | rbf+rad                      | lbfgs_final |   5 |                0 |                    0 |                  0 |              0 |                             2 |            0 |                     4 |
| heat_1d       | rbf+rad(adam)->uniform-lbfgs | lbfgs_final |   5 |                0 |                    0 |                  0 |              0 |                             2 |            0 |                     5 |
| ns2d_cg       | mlp                          | adam_final  |   5 |                0 |                    0 |                  0 |              3 |                             5 |            0 |                     1 |
| ns2d_cg       | rbf                          | adam_final  |   8 |                0 |                    1 |                  0 |              4 |                             5 |            0 |                     3 |
| piezo_2d      | mlp                          | adam_final  |   5 |                0 |                    0 |                  0 |              0 |                             5 |            0 |                     5 |
| piezo_2d      | rbf                          | adam_final  |   5 |                0 |                    0 |                  0 |              0 |                             5 |            0 |                     5 |

Nota: `low_jacobian_rank_per_param` (umbral 0.002) se activa en casi todos los modelos grandes y no discrimina; no se usa como evidencia.

## Q5 — Anchura de la estructura frente al espaciado de las bases RBF de la capa 0 (dimensión 0 = x)

Razón < 1: la estructura del propio modelo es más fina que el espaciado de las bases de la capa 0. Las capas ocultas pueden afinar la estructura (la capa 0 no es el único lugar de resolución); es un indicio, no una prueba.

| problem       | stage       |   scale/ratio_struct_to_spacing_dim0 |   scale/ratio_struct_to_width_dim0 |   struct/width_dim0 |   err/all_rms_normalized |   region/steep5/err_concentration |
|:--------------|:------------|-------------------------------------:|-----------------------------------:|--------------------:|-------------------------:|----------------------------------:|
| allen_cahn_1d | adam_final  |                               0.2404 |                             0.1374 |              0.0481 |                   0.0274 |                            0.0131 |
| allen_cahn_1d | lbfgs_final |                               0.241  |                             0.1377 |              0.0482 |                   0.0026 |                            0.0347 |
| burgers_1d    | adam_final  |                               0.0664 |                             0.0379 |              0.0133 |                   0.0084 |                           13.8393 |
| burgers_1d    | lbfgs_final |                               0.0658 |                             0.0376 |              0.0132 |                   0.0002 |                           18.89   |
| dfg_2d        | adam_final  |                               1.4777 |                             0.8444 |              0.3251 |                   0.9979 |                            6.8257 |
| dfg_2d        | lbfgs_final |                               0.5115 |                             0.2923 |              0.1125 |                   0.4323 |                            5.9825 |
| heat_1d       | adam_final  |                               6.063  |                             6.063  |              0.6366 |                   0.0001 |                            0.0022 |
| heat_1d       | lbfgs_final |                               6.063  |                             6.063  |              0.6366 |                   0      |                            0.3498 |
| ns2d_cg       | adam_final  |                               4.0995 |                             2.3425 |              1.6398 |                   0.502  |                            1.6965 |
| piezo_2d      | adam_final  |                               9.4145 |                             5.3797 |              0.0941 |                   0.1391 |                            1.7894 |

## Q6 — ¿Anticipa alguna señal de Adam cuánto ganará L-BFGS?

Spearman (descriptivo, sin corrección de múltiples pruebas ni bloques de semilla) entre una señal medida al final de Adam y log10(error L-BFGS / error Adam) pareado. Valores negativos = más ganancia de L-BFGS.

| problem       | adam_stage_signal             |   n |   spearman_with_lbfgs_gain |
|:--------------|:------------------------------|----:|---------------------------:|
| allen_cahn_1d | train/adam_log_slope_last10   |  10 |                     -0.333 |
| allen_cahn_1d | train/adam_rel_gain_last20pct |  10 |                     -0.345 |
| allen_cahn_1d | res/all/gini                  |  10 |                      0.067 |
| allen_cahn_1d | res/all/top5_mass             |  10 |                      0.018 |
| allen_cahn_1d | align/spearman_res_gradnorm   |  10 |                      0.285 |
| allen_cahn_1d | jac/effective_rank_per_param  |  10 |                      0.018 |
| burgers_1d    | train/adam_log_slope_last10   |  35 |                     -0.327 |
| burgers_1d    | train/adam_rel_gain_last20pct |  35 |                      0.37  |
| burgers_1d    | res/all/gini                  |  35 |                      0.235 |
| burgers_1d    | res/all/top5_mass             |  35 |                      0.146 |
| burgers_1d    | align/spearman_res_gradnorm   |  35 |                      0.443 |
| burgers_1d    | jac/effective_rank_per_param  |  35 |                      0.059 |
| dfg_2d        | train/adam_log_slope_last10   |  30 |                     -0.32  |
| dfg_2d        | train/adam_rel_gain_last20pct |  30 |                     -0.268 |
| dfg_2d        | res/all/gini                  |  45 |                     -0.738 |
| dfg_2d        | res/all/top5_mass             |  45 |                     -0.784 |
| dfg_2d        | align/spearman_res_gradnorm   |  45 |                      0.422 |
| dfg_2d        | jac/effective_rank_per_param  |  45 |                     -0.556 |
| heat_1d       | train/adam_log_slope_last10   |  10 |                      0.212 |
| heat_1d       | train/adam_rel_gain_last20pct |  10 |                      0.176 |
| heat_1d       | res/all/gini                  |  10 |                     -0.042 |
| heat_1d       | res/all/top5_mass             |  10 |                      0.442 |
| heat_1d       | align/spearman_res_gradnorm   |  10 |                      0.794 |
| heat_1d       | jac/effective_rank_per_param  |  10 |                     -0.382 |

