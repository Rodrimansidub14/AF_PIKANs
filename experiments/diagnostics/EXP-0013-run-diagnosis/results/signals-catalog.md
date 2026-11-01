# Catálogo de señales y etiquetas (EXP-0013)

Generado automáticamente por `catalog.py` a partir de las columnas calculadas. **Regla:** las señales (`res/`, `struct/`, `align/`, `basis/`, `scale/`, `jac/`, `param/`, `gradterm/`, problema, `train/`) no usan la referencia; las etiquetas (`err/`, `region/`) y la auditoría (`audit/`) sí, y solo sirven como variable a explicar.

Notas: (1) los checkpoints float32 se promueven a float64 para calcular señales; (2) las señales de bases solo existen para modelos RBF; (3) `gradterm/` solo existe con ≥ 2 componentes residuales; (4) la anchura de estructura en la dirección delgada de piezo no es físicamente interpretable (ver ficha).

| Columna | Origen | Definición | ¿Usa referencia? |
|---|---|---|---|
| `ac/energy_increase_frac` | signals | sobrepaso de |u|>1, número de cruces por cero, energía de Ginzburg–Landau y fracción de pasos con aumento de energía | no |
| `ac/energy_t0` | signals | sobrepaso de |u|>1, número de cruces por cero, energía de Ginzburg–Landau y fracción de pasos con aumento de energía | no |
| `ac/energy_t_end` | signals | sobrepaso de |u|>1, número de cruces por cero, energía de Ginzburg–Landau y fracción de pasos con aumento de energía | no |
| `ac/interface_count_t_end` | signals | sobrepaso de |u|>1, número de cruces por cero, energía de Ginzburg–Landau y fracción de pasos con aumento de energía | no |
| `ac/interface_count_t_mid` | signals | sobrepaso de |u|>1, número de cruces por cero, energía de Ginzburg–Landau y fracción de pasos con aumento de energía | no |
| `ac/overshoot_max` | signals | sobrepaso de |u|>1, número de cruces por cero, energía de Ginzburg–Landau y fracción de pasos con aumento de energía | no |
| `align/spearman_res_gradnorm` | signals | correlación de Spearman entre |residuo| y |∇u| (o |Δu|) del propio modelo | no |
| `align/spearman_res_laplacian` | signals | correlación de Spearman entre |residuo| y |∇u| (o |Δu|) del propio modelo | no |
| `align/top5res_in_top5grad` | signals | fracción del 5 % superior de |residuo| que cae en el 5 % superior de |∇u| | no |
| `basis/lN/coef_basis_gini` | signals | norma de coeficientes; fracción de aristas con norma < 1e-3·máx; Gini de la norma por base; fracción de activaciones tanh saturadas (|h|>0.99) | no |
| `basis/lN/coef_norm` | signals | norma de coeficientes; fracción de aristas con norma < 1e-3·máx; Gini de la norma por base; fracción de activaciones tanh saturadas (|h|>0.99) | no |
| `basis/lN/dead_fraction` | signals | fracción de pares (arista, base) cuya activación máxima sobre la malla es < 0.01 (bases muertas) | no |
| `basis/lN/eff_bases` | signals | número efectivo de bases exp(H) de la ocupación | no |
| `basis/lN/occ_mean` | signals | ocupación de bases RBF de la capa l: media y mínima de la activación media por base | no |
| `basis/lN/occ_min` | signals | ocupación de bases RBF de la capa l: media y mínima de la activación media por base | no |
| `basis/lN/small_edge_fraction` | signals | norma de coeficientes; fracción de aristas con norma < 1e-3·máx; Gini de la norma por base; fracción de activaciones tanh saturadas (|h|>0.99) | no |
| `basis/lN/tanh_saturation` | signals | norma de coeficientes; fracción de aristas con norma < 1e-3·máx; Gini de la norma por base; fracción de activaciones tanh saturadas (|h|>0.99) | no |
| `burgers/antisym_err` | signals | antisimetría u(−x,t)+u(x,t), ux máx tardío, anchura mínima del frente, posición media del frente | no |
| `burgers/front_pos_abs_mean_late` | signals | antisimetría u(−x,t)+u(x,t), ux máx tardío, anchura mínima del frente, posición media del frente | no |
| `burgers/front_width_late_min` | signals | antisimetría u(−x,t)+u(x,t), ux máx tardío, anchura mínima del frente, posición media del frente | no |
| `burgers/ux_max_late` | signals | antisimetría u(−x,t)+u(x,t), ux máx tardío, anchura mínima del frente, posición media del frente | no |
| `dfg/flux_imbalance_rel` | signals | caudal en entrada, salida y x=1.1, y desbalance relativo salida−entrada | no |
| `dfg/flux_inlet` | signals | caudal en entrada, salida y x=1.1, y desbalance relativo salida−entrada | no |
| `dfg/flux_mid` | signals | caudal en entrada, salida y x=1.1, y desbalance relativo salida−entrada | no |
| `dfg/flux_outlet` | signals | caudal en entrada, salida y x=1.1, y desbalance relativo salida−entrada | no |
| `gradterm/cos_*` | signals | coseno entre gradientes de componentes residuales; razón máx/mín de normas | no |
| `gradterm/norm_*` | signals | norma del gradiente ∇θ de ½·media(r_c²) por componente residual c (1 024 puntos) | no |
| `heat/decay_rate` | signals | energía ∫u²dx, tasa de decaimiento ajustada y su razón con el valor de la PDE (−2απ²), |u|máx (propiedades de la PDE, no de la solución exacta) | no |
| `heat/decay_rate_over_expected` | signals | energía ∫u²dx, tasa de decaimiento ajustada y su razón con el valor de la PDE (−2απ²), |u|máx (propiedades de la PDE, no de la solución exacta) | no |
| `heat/energy_t0` | signals | energía ∫u²dx, tasa de decaimiento ajustada y su razón con el valor de la PDE (−2απ²), |u|máx (propiedades de la PDE, no de la solución exacta) | no |
| `heat/energy_t1` | signals | energía ∫u²dx, tasa de decaimiento ajustada y su razón con el valor de la PDE (−2απ²), |u|máx (propiedades de la PDE, no de la solución exacta) | no |
| `heat/u_absmax` | signals | energía ∫u²dx, tasa de decaimiento ajustada y su razón con el valor de la PDE (−2απ²), |u|máx (propiedades de la PDE, no de la solución exacta) | no |
| `jac/cond` | signals | espectro del jacobiano ∂u/∂θ en 256 puntos fijos: rango efectivo exp(H(σ/Σσ)), por parámetro, número de condición, σmax | no |
| `jac/effective_rank` | signals | espectro del jacobiano ∂u/∂θ en 256 puntos fijos: rango efectivo exp(H(σ/Σσ)), por parámetro, número de condición, σmax | no |
| `jac/effective_rank_per_param` | signals | espectro del jacobiano ∂u/∂θ en 256 puntos fijos: rango efectivo exp(H(σ/Σσ)), por parámetro, número de condición, σmax | no |
| `jac/n_points` | signals | espectro del jacobiano ∂u/∂θ en 256 puntos fijos: rango efectivo exp(H(σ/Σσ)), por parámetro, número de condición, σmax | no |
| `jac/sigma_max` | signals | espectro del jacobiano ∂u/∂θ en 256 puntos fijos: rango efectivo exp(H(σ/Σσ)), por parámetro, número de condición, σmax | no |
| `param/absmax` | signals | cuenta, norma, máximo y fracción pequeña (<1e-3) de los parámetros | no |
| `param/count` | signals | cuenta, norma, máximo y fracción pequeña (<1e-3) de los parámetros | no |
| `param/norm` | signals | cuenta, norma, máximo y fracción pequeña (<1e-3) de los parámetros | no |
| `param/small_fraction` | signals | cuenta, norma, máximo y fracción pequeña (<1e-3) de los parámetros | no |
| `piezo/sigma_xx_rms` | signals | magnitudes del modelo (RMS de deformación, u, v, σxx, flecha en la punta) | no |
| `piezo/strain_rms` | signals | magnitudes del modelo (RMS de deformación, u, v, σxx, flecha en la punta) | no |
| `piezo/tip_v_model` | signals | magnitudes del modelo (RMS de deformación, u, v, σxx, flecha en la punta) | no |
| `piezo/u_rms` | signals | magnitudes del modelo (RMS de deformación, u, v, σxx, flecha en la punta) | no |
| `piezo/v_rms` | signals | magnitudes del modelo (RMS de deformación, u, v, σxx, flecha en la punta) | no |
| `res/all/entropy_norm` | signals | entropía de Shannon normalizada de r² (o e²) sobre los puntos (1 = uniforme) | no (res/) / sí (err/) |
| `res/all/gini` | signals | coeficiente de Gini de |residuo| (0 = uniforme, 1 = concentrado en un punto); en `err/` es el de |error| | no (res/) / sí (err/) |
| `res/all/kurtosis` | signals | curtosis (exceso) de |residuo| | no |
| `res/all/max` | signals | percentil / máximo de |residuo| | no |
| `res/all/p50` | signals | percentil / máximo de |residuo| | no |
| `res/all/p95` | signals | percentil / máximo de |residuo| | no |
| `res/all/p99` | signals | percentil / máximo de |residuo| | no |
| `res/all/rms` | signals | RMS del residuo de la PDE (por punto, escalado por la escala del contrato) sobre la malla del problema | no |
| `res/all/top1_mass` | signals | fracción de la masa Σ r² (o Σ e²) contenida en el 1 % / 5 % de puntos con mayor valor | no (res/) / sí (err/) |
| `res/all/top5_mass` | signals | fracción de la masa Σ r² (o Σ e²) contenida en el 1 % / 5 % de puntos con mayor valor | no (res/) / sí (err/) |
| `res/<componente>/entropy_norm` | signals | entropía de Shannon normalizada de r² (o e²) sobre los puntos (1 = uniforme) | no (res/) / sí (err/) |
| `res/<componente>/gini` | signals | coeficiente de Gini de |residuo| (0 = uniforme, 1 = concentrado en un punto); en `err/` es el de |error| | no (res/) / sí (err/) |
| `res/<componente>/kurtosis` | signals | curtosis (exceso) de |residuo| | no |
| `res/<componente>/max` | signals | percentil / máximo de |residuo| | no |
| `res/<componente>/p50` | signals | percentil / máximo de |residuo| | no |
| `res/<componente>/p95` | signals | percentil / máximo de |residuo| | no |
| `res/<componente>/p99` | signals | percentil / máximo de |residuo| | no |
| `res/<componente>/rms` | signals | RMS del residuo de la PDE (por punto, escalado por la escala del contrato) sobre la malla del problema | no |
| `res/<componente>/top1_mass` | signals | fracción de la masa Σ r² (o Σ e²) contenida en el 1 % / 5 % de puntos con mayor valor | no (res/) / sí (err/) |
| `res/<componente>/top5_mass` | signals | fracción de la masa Σ r² (o Σ e²) contenida en el 1 % / 5 % de puntos con mayor valor | no (res/) / sí (err/) |
| `res/n_points` | signals | número de puntos de la malla | no |
| `res/t_centroid` | signals | centroide de la masa del residuo en x o t | no |
| `res/t_quartile1_mass` | signals | fracción de la masa del residuo en cada cuartil de tiempo (problemas 1D-tiempo) | no |
| `res/t_quartile2_mass` | signals | fracción de la masa del residuo en cada cuartil de tiempo (problemas 1D-tiempo) | no |
| `res/t_quartile3_mass` | signals | fracción de la masa del residuo en cada cuartil de tiempo (problemas 1D-tiempo) | no |
| `res/t_quartile4_mass` | signals | fracción de la masa del residuo en cada cuartil de tiempo (problemas 1D-tiempo) | no |
| `res/x_centroid` | signals | centroide de la masa del residuo en x o t | no |
| `scale/ratio_struct_to_spacing_<dim>` | signals | anchura de estructura del modelo dividida por el espaciado/ancho físico de bases de la capa 0 (< 1 = estructura más fina que la base) | no |
| `scale/ratio_struct_to_width_<dim>` | signals | anchura de estructura del modelo dividida por el espaciado/ancho físico de bases de la capa 0 (< 1 = estructura más fina que la base) | no |
| `scale/spacing_phys_<dim>` | signals | espaciado y ancho de las bases RBF de la capa 0 en unidades físicas por dimensión de entrada | no |
| `scale/width_phys_<dim>` | signals | espaciado y ancho de las bases RBF de la capa 0 en unidades físicas por dimensión de entrada | no |
| `struct/grad_norm_max` | signals | estructura de la solución del propio modelo: máx. y p95 de |∇u|, máx. |Δu|, anchura de estructura 2·max|u|/max|∂u/∂xi| por dimensión | no |
| `struct/grad_norm_p95` | signals | estructura de la solución del propio modelo: máx. y p95 de |∇u|, máx. |Δu|, anchura de estructura 2·max|u|/max|∂u/∂xi| por dimensión | no |
| `struct/laplacian_abs_max` | signals | estructura de la solución del propio modelo: máx. y p95 de |∇u|, máx. |Δu|, anchura de estructura 2·max|u|/max|∂u/∂xi| por dimensión | no |
| `struct/value_absmax` | signals | estructura de la solución del propio modelo: máx. y p95 de |∇u|, máx. |Δu|, anchura de estructura 2·max|u|/max|∂u/∂xi| por dimensión | no |
| `struct/width_<dim>` | signals | estructura de la solución del propio modelo: máx. y p95 de |∇u|, máx. |Δu|, anchura de estructura 2·max|u|/max|∂u/∂xi| por dimensión | no |
| `train/adam_log_slope_last10` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/adam_loss_end` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/adam_loss_start` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/adam_n_logged` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/adam_rel_gain_last20pct` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/adam_seconds` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/adam_steps` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/lbfgs_iterations` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/lbfgs_log_slope_last10` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/lbfgs_loss_end` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/lbfgs_loss_start` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/lbfgs_n_logged` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/lbfgs_seconds` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/lbfgs_to_adam_loss_ratio` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/refresh_seconds` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `train/seconds` | signals | señales de entrenamiento registradas (pérdida final, pendiente log, ganancia relativa del último 20 %, razón L-BFGS/Adam, iteraciones, estado de parada, tiempos) | no |
| `audit/err_mass_in_top5res` | labels | AUDITORÍA: alineación residuo–error (Spearman, solape de 5 % superiores, masa de error en el 5 % superior del residuo) | sí |
| `audit/res_mass_in_top5err` | labels | AUDITORÍA: alineación residuo–error (Spearman, solape de 5 % superiores, masa de error en el 5 % superior del residuo) | sí |
| `audit/spearman_res_err` | labels | AUDITORÍA: alineación residuo–error (Spearman, solape de 5 % superiores, masa de error en el 5 % superior del residuo) | sí |
| `audit/spearman_res_err_in_steep5` | labels | AUDITORÍA: alineación residuo–error (Spearman, solape de 5 % superiores, masa de error en el 5 % superior del residuo) | sí |
| `audit/top5_overlap` | labels | AUDITORÍA: alineación residuo–error (Spearman, solape de 5 % superiores, masa de error en el 5 % superior del residuo) | sí |
| `err/all_rms_normalized` | labels | ETIQUETA: error relativo L2 por componente, plano y normalizado, y concentración del error | sí |
| `err/entropy_norm` | labels | entropía de Shannon normalizada de r² (o e²) sobre los puntos (1 = uniforme) | no (res/) / sí (err/) |
| `err/flat_rel_l2` | labels | ETIQUETA: error relativo L2 por componente, plano y normalizado, y concentración del error | sí |
| `err/gini` | labels | coeficiente de Gini de |residuo| (0 = uniforme, 1 = concentrado en un punto); en `err/` es el de |error| | no (res/) / sí (err/) |
| `err/<comp>_rel_l2` | labels | ETIQUETA: error relativo L2 por componente, plano y normalizado, y concentración del error | sí |
| `err/top1_mass` | labels | fracción de la masa Σ r² (o Σ e²) contenida en el 1 % / 5 % de puntos con mayor valor | no (res/) / sí (err/) |
| `err/top5_mass` | labels | fracción de la masa Σ r² (o Σ e²) contenida en el 1 % / 5 % de puntos con mayor valor | no (res/) / sí (err/) |
| `region/<región>/err_concentration` | labels | ETIQUETA: masa y concentración del error y del residuo en regiones definidas con la referencia (steep5, interfaz, cerca del cilindro, capas, …) | sí |
| `region/<región>/err_mass` | labels | ETIQUETA: masa y concentración del error y del residuo en regiones definidas con la referencia (steep5, interfaz, cerca del cilindro, capas, …) | sí |
| `region/<región>/point_fraction` | labels | ETIQUETA: masa y concentración del error y del residuo en regiones definidas con la referencia (steep5, interfaz, cerca del cilindro, capas, …) | sí |
| `region/<región>/res_concentration` | labels | ETIQUETA: masa y concentración del error y del residuo en regiones definidas con la referencia (steep5, interfaz, cerca del cilindro, capas, …) | sí |
| `region/<región>/res_mass` | labels | ETIQUETA: masa y concentración del error y del residuo en regiones definidas con la referencia (steep5, interfaz, cerca del cilindro, capas, …) | sí |
