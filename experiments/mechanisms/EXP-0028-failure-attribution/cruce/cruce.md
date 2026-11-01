# EXP-0028: cruce efecto x propiedades (DESCRIPTIVO, n = 6 problemas)

Generado por `cruce.py` (lee `results/phase1_scored.json`, `phase1c_scored.json`, `phase1c_dfg_scored.json`, `fase2/properties*.csv`). No hay p-values como evidencia; los intervalos son bootstrap sobre los 6 problemas (no es un intervalo de confianza fiable: con n = 6 y empates, el percentil no contiene a veces al estimador; se da el rango leave-one-out en `spearman_efecto_propiedad.csv`). Todo esto describe **6 puntos**, no es una regresion ni una prueba de causalidad por propiedad.

## 1. Mediciones nuevas de la fase 2 (CPU, semillas 4001-4010)

- **DFG medido** con el adaptador `DFGTerms` (13 terminos: 3 PDE + 10 de borde, pesos 1 y 1/2), modelo `DFGRBFPIKAN` con `output_scales`, pools completos de `generate_dfg_pools`.
- **NS repetido con `residual_terms`** (8 terminos con borde). Control: `pde_condB_median` = 7.2e3, igual que la fase 2 original (solo PDE); con borde sube a 8.6e3. `terms_vs_loss_relerr` ~1e-7 (redondeo float32): el adaptador reproduce la perdida del entrenamiento.
- Script nuevo `fase2/measure_properties_terms.py` (+ `make_table_terms.py`, `measure_edge_scale_scalar.py`); salidas `fase2/properties_terms.csv/.md`, `fase2/properties_edge_scale.csv` (escala entre aristas, solo 3 semillas). No se tocaron `properties.csv` ni `measure_properties.py`.
- **Piezo (stand-in)**: no existe `PiezoProblem` en `src/`; el codigo de Daniel (`pinn_piezo`) no esta en el arbol. Se uso la reimplementacion propia `experiments/piezo/EXP-0009-piezo-indirect-rbf/piezo_pinn.py` (8 salidas, bordes duros, 16 terminos de perdida de peso 1, RBF16 float64, 2048 + 256 por borde) con un adaptador local; ver las salvedades en `predictions_DRAFT.md`.

Propiedades por problema (mediana entre semillas; `-` = no aplica):

| propiedad | AC 1e-3 | AC 2e-4 | Burgers | KPP | NS (con borde) | DFG | piezo (stand-in) |
|---|---|---|---|---|---|---|---|
| concentracion r^2, top 10 % (0.10 = uniforme) | 0.428 | 0.429 | 0.208 | 0.622 | 0.308 | 0.334 | 0.588 |
| RMS residuo inicial | 0.901 | 0.9 | 1.12 | 0.607 | 0.0775 | 0.0329 | 795 |
| cond(B) mediana | 9.26e+03 | 9.26e+03 | 8.48e+03 | 1.01e+04 | 8.56e+03 | 9.69e+03 | 8.68e+03 |
| cond(B) minima | 3.43e+03 | 3.43e+03 | 3.24e+03 | 4.15e+03 | 2e+03 | 1.99e+03 | 2.05e+03 |
| borde/PDE en la perdida inicial | 0 | 0 | 0 | 0 | 87.2 | 593 | 2.76e-07 |
| traza GN borde/PDE | 0 | 0 | 0 | 0 | 7.74 | 37.7 | 1.29e-07 |
| n salidas | 1 | 1 | 1 | 1 | 3 | 3 | 8 |
| n terminos de perdida | 1 | 1 | 1 | 1 | 8 | 13 | 16 |
| razon de escalas de salida (unid. mezcladas) | 1 | 1 | 1 | 1 | 1 | 5 | 2e+10 |
| max/min perdida por termino | - | - | - | - | 1.5e+03 | 1.65e+07 | 1.04e+09 |
| Peclet / Re | - | - | 314 | 204 | 100 | 20 | - |
| Damkohler | 5e+03 | 2.5e+04 | - | 1e+04 | - | - | - |
| ancho de frente / ancho RBF | 0.0404 | 0.0181 | 0.00909 | 0.07 | - | - | - |
| ancho del modelo | 16 | 16 | 48 | 16 | 16 | 16 | 16 |
| log10 p95/p5 traza(B) entre aristas (3 sem.) | 3.2 | 3.2 | 1.68 | 3.46 | 2.23 | 2.54 | 6.63 |
| log10 max/min traza entre capas (3 sem.) | 2.63 | 2.63 | 2.75 | 3.04 | 1.91 | 2.29 | 4.17 |

AC, Burgers y KPP tienen IC y BC duras: no hay terminos de borde, y `bc_over_pde` = 0 por construccion (no es una medicion). Para DFG: `bc_over_pde` = 593 (rango entre semillas 315-1220) y la traza GN esta dominada ~38 veces por el borde. En el stand-in del piezo la perdida inicial la dicta casi un solo termino constitutivo (sz 5.8e5 de 6.3e5 en total) por la disparidad de escalas; el borde pesa 1e-7. Valores `edge_trace_*` y `layer_trace_*` solo con semillas 4001-4003. NS y DFG no tienen Damkohler ni `front_over_rbf` (sin frente prescrito).

Otras observaciones de la medicion: `condB_max` esta saturado (techo 1.1e4) en todos, no discrimina; `gram_cond` es identico (covariable de F no varia entre problemas, solo se midio una vez); los numeros de cond(B) de los 7 problemas estan todos en 8.5e3-1.0e4 (mediana), de modo que **cond(B) no separa nada** (el rango de 1.2x frente al efecto de 100x).

## 2. Tabla problema x brazo (razon mediana e_X/e_B pareada; mejoras k/n; clase del criterio uniforme: S sirve, D daña, i indiferente, d dependiente de semilla; s/d sin datos)

NS y DFG: metrica `l2re_flat`; todas a receta corta (Adam 2000-4000, L-BFGS 500 it.). Nota: el JSON de NS rotula "Adam 20000" pero el .md del scoring dice t1 = Adam 2000: tomo el .md. Datos completos en `tabla_problema_brazo.csv`, `tabla_wide_final.csv`, `tabla_wide_adam.csv`. Los brazos S, R, C solo existen en AC, Burgers y KPP (n = 4-5); W (oraculo) esta en el CSV, no en las figuras.

**Final (tras L-BFGS)**

| problema | F | J | E | I | S | R | C |
|---|---|---|---|---|---|---|---|
| AC 1e-3 | 0.35 (10/10, S) | 0.94 (6/10, i) | 0.16 (10/10, S) | 0.14 (6/10, d) | 1.39 (1/5, D) | 0.79 (5/5, S) | 0.78 (5/5, S) |
| AC 2e-4 | 0.40 (10/10, S) | 0.84 (8/10, S) | 0.24 (10/10, S) | 0.14 (9/10, d) | 1.05 (1/5, i) | 0.80 (5/5, S) | 0.67 (5/5, S) |
| Burgers | 0.29 (10/10, S) | 2.94 (3/10, D) | 0.24 (9/10, d) | 47.86 (0/9, D) | 12.89 (0/4, D) | 76.99 (0/4, D) | 4.43 (0/4, D) |
| KPP | 1.84 (4/10, d) | 2.38 (3/10, D) | 0.77 (6/10, d) | 1.39 (5/10, d) | 3.19 (0/5, D) | 2.56 (1/5, D) | 1.32 (1/5, D) |
| NS2d-CG | 0.83 (10/10, S) | 0.97 (6/10, i) | 0.79 (10/10, S) | 0.74 (10/10, S) | s/d | s/d | s/d |
| DFG | 0.57 (10/10, S) | 2.07 (0/10, D) | 0.81 (10/10, S) | 0.39 (10/10, S) | s/d | s/d | s/d |

**Final de Adam** (lo relevante para piezo, que no tiene L-BFGS)

| problema | F | J | E | I | S | R | C |
|---|---|---|---|---|---|---|---|
| AC 1e-3 | 0.29 (10/10, S) | 0.99 (5/10, i) | 0.28 (10/10, S) | 0.22 (6/10, d) | 1.74 (0/5, D) | 0.92 (5/5, i) | 0.79 (4/5, S) |
| AC 2e-4 | 0.47 (10/10, S) | 0.96 (6/10, i) | 0.40 (10/10, S) | 0.18 (9/10, d) | 1.35 (0/5, D) | 1.05 (0/5, i) | 0.73 (5/5, S) |
| Burgers | 0.63 (8/10, d) | 1.93 (4/10, d) | 1.14 (4/10, d) | 6.24 (1/9, D) | 5.83 (0/4, D) | 7.57 (0/4, D) | 0.94 (2/4, d) |
| KPP | 1.09 (5/10, d) | 2.72 (2/10, D) | 0.76 (7/10, d) | 1.32 (5/10, d) | 2.96 (0/5, D) | 1.17 (2/5, d) | 1.15 (1/5, D) |
| NS2d-CG | 0.68 (10/10, S) | 1.00 (5/10, i) | 0.64 (10/10, S) | 0.56 (10/10, S) | s/d | s/d | s/d |
| DFG | 0.80 (10/10, S) | 1.02 (2/10, i) | 0.83 (10/10, S) | 0.76 (10/10, S) | s/d | s/d | s/d |

Figura: `heatmap_efecto.{png,pdf}` (log10 de la razon mediana, problema x brazo, final y Adam).

## 3. Efecto frente a propiedad (Spearman, n = 6)

Archivos: `spearman_efecto_propiedad.csv` (rho, IC bootstrap, rango leave-one-out, n), `heatmap_spearman.{png,pdf}`, `dispersion_efecto_propiedad.{png,pdf}`. Propiedades definidas solo en una parte de los problemas tienen n = 3-4 (Peclet n = 4, Damkohler n = 3 y por tanto sin rho, `front_over_rbf` n = 4): no se interpretan.

Lo que muestran las cifras, con la salvedad de que **muchas propiedades son la misma particion binaria** {AC, Burgers, KPP: escalar, borde duro} frente a {NS, DFG: vectorial con terminos de borde}: `n_outputs`, `n_loss_terms`, `bc_over_pde`, `gn_trace_bc_over_pde` y (en la direccion contraria) `resid_rms0` son indistinguibles con estos datos, y `model_width` (Burgers = 48) y `grad_norm0` se confunden con otro problema cada uno.

- **F**: no hay propiedad que ordene el efecto de F con rho fiable. F sirve en 5 de 6; el unico problema donde no sirve es KPP, que tiene la mayor concentracion del residuo (0.62), la mayor cond(B) y el mayor `grad_norm0` (20); las tres separan KPP del resto, pero eso es un solo problema (una separacion perfecta de 1 contra 5 ocurre por azar con probabilidad 1/3 por propiedad). La covariable propia de F (Gram uniforme) es igual en todos.
- **E** (final): rho = +0.85 con borde/PDE (el beneficio de E es mayor donde no hay terminos de borde: AC 0.16-0.24; NS 0.79, DFG 0.81), -0.94 con el RMS inicial (misma particion). Con Peclet (n = 4, rho -1.0) solo ordena NS/DFG (Re 20-100, E sirve) frente a Burgers/KPP (Pe 200-300, E dependiente de semilla); AC no tiene Peclet. Lectura: E sirve donde la ecuacion es de reaccion-difusion con frente estacionario (AC) o estacionaria (NS, DFG) y es poco fiable en los dos problemas de transporte con frente viajero (Burgers, KPP). Esa lectura es una **hipotesis**, no esta respaldada por 6 puntos. En Adam el patron de E por borde desaparece (rho 0.27).
- **J** (final): ninguna propiedad de la tabla lo ordena (|rho| <= 0.5 salvo Peclet 0.80 con n = 4 y `model_width`). J daña en Burgers, KPP y DFG; en Burgers y DFG el daño aparece solo al pasar a L-BFGS (en Adam es dependiente de semilla/indiferente), es decir, no es el reescalado en si sino su interaccion con el segundo optimizador. KPP es el unico donde J ya daña en Adam (2.72).
- **I**: es el brazo con la separacion mas limpia: sirve en NS y DFG (Adam 0.56 y 0.76), es dependiente de semilla en AC y KPP y daña catastroficamente en Burgers (48x). La particion {NS, DFG} frente a {escalares} es perfecta para `n_outputs`, `n_loss_terms`, `bc_over_pde`, `resid_rms0`, pero son colineales: **el banco no puede decir cual de ellas es la causa** (acoplamiento de salidas, presencia de terminos de borde o simplemente que son dos problemas de Navier-Stokes). No hay un problema multi-salida sin borde ni uno escalar con borde que desempate.
- **Concentracion del residuo**: rho entre -0.4 y +0.5, con rango leave-one-out que cambia de signo en los cuatro brazos. Con estos 6 problemas no separa donde un brazo sirve de donde daña. (Burgers es el menos concentrado, 0.21, y a la vez es donde I y J dañan mas; KPP, el mas concentrado, es donde F y E no sirven; AC, intermedio, es donde todo sirve: no monotono.)
- **cond(B)**: no discrimina (ver arriba). **Escala entre aristas** (3 semillas): rho entre -0.5 y +0.5, nada interpretable; lo relevante es que el stand-in de piezo cae **fuera del rango del banco** (6.6 decadas frente a 1.7-3.5).

## 4. Que propiedades separan «sirve» de «daña» (`separacion_clase.csv`, AUC de Mann-Whitney por propiedad)

Separaciones perfectas (AUC 0 o 1) encontradas: F sirve vs KPP (concentracion, cond(B), grad_norm0, `front_over_rbf`; 1 problema contra 5); E sirve {AC1, AC2, NS, DFG} vs {Burgers, KPP} por Peclet (n = 2 + 2); I sirve {NS, DFG} vs el resto por salidas/terminos/borde/RMS (2 + 4); J: ninguna entre «daña» y el resto. Con 6 problemas, ~15 propiedades correlacionadas y un grupo de 1-2 problemas, una separacion perfecta es esperable por azar para varias propiedades (con 2 contra 4, la probabilidad por propiedad es 1/15 si el orden fuera aleatorio): **ninguna de estas separaciones es evidencia**; sirven para nombrar candidatas, que es lo que usa `predictions_DRAFT.md`.

Lectura honesta resumida:
1. El unico factor que el banco ordena con claridad es **«vectorial con terminos de borde»** (NS, DFG) frente a **«escalar con borde duro»**, y solo para I (sirve frente a no sirve) y, en menor medida, para E (sirve mas donde no hay borde). Es un factor binario con 2 problemas de un lado.
2. Donde un brazo daña (J en Burgers/KPP/DFG; I en Burgers; S, R, C en Burgers/KPP) no hay propiedad medible a la inicializacion que lo explique con n = 6; **el daño de J y R aparece en L-BFGS**, lo que apunta al entrenamiento (eje C), no al problema.
3. La concentracion del residuo, cond(B), el Damkohler/Peclet y el acoplamiento por separado **no quedan respaldados** como separadores con estos datos.

## 5. Potencia y limites

- n = 6 problemas (3 de ellos escalares con frentes, 2 de Navier-Stokes, 1 AC con dos coeficientes que comparten red y pool: AC 1e-3 y 2e-4 son casi el mismo punto, es decir, hay ~5 puntos independientes).
- Receta corta (Adam 2000-4000 float32, L-BFGS 500 it.) en todo el banco; piezo corre Adam 20 000 float64 sin L-BFGS. Los efectos de la fase 1d (receta completa) no estan.
- La razon e_X/e_B de cada problema es una mediana de 4-10 semillas pareadas con un B de la misma sesion; B no reproduce por semilla entre sesiones (fase 0, inconsistencia 2), la dependencia de semilla de Burgers/KPP puede ser en parte ruido del control.
- Las propiedades se miden a la inicializacion; el efecto es de la trayectoria entera.
- NS usa ancho oculto 16 elegido por el agente de la fase 2, no fijado por el spec (igual que la fase 1c).
