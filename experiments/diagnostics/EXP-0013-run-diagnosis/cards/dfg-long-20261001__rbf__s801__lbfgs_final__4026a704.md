# Ficha de diagnóstico — dfg-long-20261001__rbf__s801__lbfgs_final__4026a704

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** rbf · **Semilla:** 801 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-rbf-seed-801/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1682, p = 0.2514, u = 0.1659, v = 0.2358
- Error normalizado (RMS, todas las componentes): 0.3826; plano: 0.1682 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.318; masa en el 5 % superior 0.4605; entropía normalizada 0.8876

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.299 | 4.439 | 0.8093 | 12.01 |
| steep5 | 0.05009 | 0.3023 | 6.034 | 0.7889 | 15.75 |
| wake | 0.15 | 0.1162 | 0.775 | 0.02177 | 0.1452 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01846; p95 0.01981; máx 0.7987
- Concentración: Gini 0.5464; masa en el 1 % / 5 % superior 0.7899 / 0.8999; entropía normalizada 0.4788
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1394; con |Δu| 0.3869

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.598; solape del 5 % superior 0.4109; masa de error en el 5 % superior del residuo 0.3111; masa de residuo en el 5 % superior del error 0.3392
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 36.89 (por parámetro 0.00915); número de condición 3.332e+06
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.5164 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.9907 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.00295
- adam_loss_end: 0.0007527
- adam_loss_start: 0.002813
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.03256
- adam_seconds: 2496
- adam_steps: 2e+04
- lbfgs_iterations: 4000
- lbfgs_log_slope_last10: -0.008073
- lbfgs_loss_end: 0.0002759
- lbfgs_loss_start: 0.0007366
- lbfgs_n_logged: 160
- lbfgs_seconds: 600.1
- lbfgs_to_adam_loss_ratio: 0.3665
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, residual_aligned

## 8. Límites y datos ausentes

- (ninguno)
