# Ficha de diagnóstico — dfg-long-20261001__rbf__s802__lbfgs_final__a7b7693d

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** rbf · **Semilla:** 802 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-rbf-seed-802/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1916, p = 0.2984, u = 0.1884, v = 0.2924
- Error normalizado (RMS, todas las componentes): 0.4583; plano: 0.1916 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3077; masa en el 5 % superior 0.5195; entropía normalizada 0.8678

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3487 | 5.176 | 0.8166 | 12.12 |
| steep5 | 0.05009 | 0.348 | 6.947 | 0.802 | 16.01 |
| wake | 0.15 | 0.09847 | 0.6567 | 0.0269 | 0.1794 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01757; p95 0.02106; máx 0.7702
- Concentración: Gini 0.5195; masa en el 1 % / 5 % superior 0.7678 / 0.8784; entropía normalizada 0.4686
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2155; con |Δu| 0.3581

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.537; solape del 5 % superior 0.4717; masa de error en el 5 % superior del residuo 0.3551; masa de residuo en el 5 % superior del error 0.5637
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 32.87 (por parámetro 0.008152); número de condición 6.568e+06
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.5113 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.9687 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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

- budget_limited, concentrated_error, residual_aligned

## 8. Límites y datos ausentes

- (ninguno)
