# Ficha de diagnóstico — dfg-long-20261001__rbf__s804__lbfgs_final__acb70ec5

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** rbf · **Semilla:** 804 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-rbf-seed-804/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1808, p = 0.3006, u = 0.1776, v = 0.2548
- Error normalizado (RMS, todas las componentes): 0.4323; plano: 0.1808 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2998; masa en el 5 % superior 0.4481; entropía normalizada 0.8926

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2934 | 4.355 | 0.7918 | 11.75 |
| steep5 | 0.05009 | 0.2997 | 5.982 | 0.7511 | 14.99 |
| wake | 0.15 | 0.1167 | 0.7785 | 0.03749 | 0.25 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0154; p95 0.02073; máx 0.6364
- Concentración: Gini 0.5547; masa en el 1 % / 5 % superior 0.7115 / 0.8606; entropía normalizada 0.5377
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2491; con |Δu| 0.3921

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.6194; solape del 5 % superior 0.45; masa de error en el 5 % superior del residuo 0.3197; masa de residuo en el 5 % superior del error 0.6242
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 32.93 (por parámetro 0.008168); número de condición 4.183e+06
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.5115 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.9603 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.00398
- adam_loss_end: 0.0008448
- adam_loss_start: 0.002627
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.02821
- adam_seconds: 2498
- adam_steps: 2e+04
- lbfgs_iterations: 4075
- lbfgs_log_slope_last10: -0.006342
- lbfgs_loss_end: 0.0003069
- lbfgs_loss_start: 0.0007752
- lbfgs_n_logged: 163
- lbfgs_seconds: 600.4
- lbfgs_to_adam_loss_ratio: 0.3633
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, residual_aligned

## 8. Límites y datos ausentes

- (ninguno)
