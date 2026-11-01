# Ficha de diagnóstico — dfg-long-20261001__rbf__s802__adam_final__a3c69462

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** rbf · **Semilla:** 802 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-rbf-seed-802/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.4593, p = 0.6358, u = 0.4547, v = 0.5942
- Error normalizado (RMS, todas las componentes): 0.9819; plano: 0.4593 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3408; masa en el 5 % superior 0.4903; entropía normalizada 0.8718

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3463 | 5.14 | 0.6725 | 9.983 |
| steep5 | 0.05009 | 0.347 | 6.927 | 0.6722 | 13.42 |
| wake | 0.15 | 0.09693 | 0.6464 | 0.01189 | 0.07929 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01368; p95 0.02585; máx 0.2116
- Concentración: Gini 0.6595; masa en el 1 % / 5 % superior 0.5542 / 0.8431; entropía normalizada 0.6738
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4006; con |Δu| 0.5167

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.403; solape del 5 % superior 0.6065; masa de error en el 5 % superior del residuo 0.3944; masa de residuo en el 5 % superior del error 0.6114
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 22.28 (por parámetro 0.005525); número de condición 1.237e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 1.478 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.673 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.00723
- adam_loss_end: 0.0007528
- adam_loss_start: 0.002699
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.05673
- adam_seconds: 2495
- adam_steps: 2e+04
- lbfgs_iterations: 4075
- lbfgs_log_slope_last10: -0.005367
- lbfgs_loss_end: 0.0003358
- lbfgs_loss_start: 0.0007439
- lbfgs_n_logged: 163
- lbfgs_seconds: 603.1
- lbfgs_to_adam_loss_ratio: 0.4461
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
