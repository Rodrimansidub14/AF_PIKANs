# Ficha de diagnóstico — dfg-long-20261001__mlp__s803__lbfgs_final__30281a2c

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 803 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-803/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2724, p = 0.4363, u = 0.2689, v = 0.3161
- Error normalizado (RMS, todas las componentes): 0.6021; plano: 0.2724 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2993; masa en el 5 % superior 0.4229; entropía normalizada 0.9033

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2642 | 3.921 | 0.9856 | 14.63 |
| steep5 | 0.05009 | 0.266 | 5.309 | 0.9822 | 19.61 |
| wake | 0.15 | 0.1175 | 0.7834 | 0.001939 | 0.01293 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.06509; p95 0.02466; máx 4.822
- Concentración: Gini 0.6761; masa en el 1 % / 5 % superior 0.9788 / 0.9892; entropía normalizada 0.1478
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4131; con |Δu| 0.5943

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5424; solape del 5 % superior 0.3804; masa de error en el 5 % superior del residuo 0.2322; masa de residuo en el 5 % superior del error 0.9676
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 13.21 (por parámetro 0.001661); número de condición 1.421e+16

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.0061
- adam_loss_end: 0.001149
- adam_loss_start: 0.00295
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.1662
- adam_seconds: 742.9
- adam_steps: 2e+04
- lbfgs_iterations: 1.14e+04
- lbfgs_log_slope_last10: -0.00929
- lbfgs_loss_end: 0.0003173
- lbfgs_loss_start: 0.001074
- lbfgs_n_logged: 456
- lbfgs_seconds: 601.1
- lbfgs_to_adam_loss_ratio: 0.276
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, residual_aligned, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
