# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s805__adam_final__6079db5b

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 805 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-805/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9675, p = 1.009, u = 0.9664, v = 1.014
- Error normalizado (RMS, todas las componentes): 1.726; plano: 0.9675 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2923; masa en el 5 % superior 0.3448; entropía normalizada 0.9206

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3223 | 4.784 | 0.2176 | 3.23 |
| steep5 | 0.05009 | 0.295 | 5.888 | 0.229 | 4.572 |
| wake | 0.15 | 0.1628 | 1.085 | 0.01834 | 0.1223 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005468; p95 0.01476; máx 0.02395
- Concentración: Gini 0.4746; masa en el 1 % / 5 % superior 0.1522 / 0.5581; entropía normalizada 0.8349
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6078; con |Δu| 0.4176

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1662; solape del 5 % superior 0.3783; masa de error en el 5 % superior del residuo 0.1479; masa de residuo en el 5 % superior del error 0.3193
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.299 (por parámetro 0.0002891); número de condición 1.59e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.006261
- adam_loss_end: 0.002524
- adam_loss_start: 0.003378
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.1008
- adam_seconds: 71.29
- adam_steps: 2000
- lbfgs_loss_end: 0.001654
- lbfgs_loss_start: 0.002415
- lbfgs_n_logged: 10
- lbfgs_seconds: 47.38
- lbfgs_to_adam_loss_ratio: 0.6551
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
