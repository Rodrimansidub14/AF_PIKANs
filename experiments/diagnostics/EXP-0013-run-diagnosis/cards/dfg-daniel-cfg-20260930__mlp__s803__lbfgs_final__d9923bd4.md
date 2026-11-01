# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s803__lbfgs_final__d9923bd4

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 803 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-803/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9152, p = 1.024, u = 0.9125, v = 1.019
- Error normalizado (RMS, todas las componentes): 1.709; plano: 0.9152 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3027; masa en el 5 % superior 0.3252; entropía normalizada 0.9219

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2824 | 4.191 | 0.2582 | 3.832 |
| steep5 | 0.05009 | 0.2564 | 5.119 | 0.2332 | 4.655 |
| wake | 0.15 | 0.1573 | 1.049 | 0.05451 | 0.3635 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.006475; p95 0.01426; máx 0.04077
- Concentración: Gini 0.5217; masa en el 1 % / 5 % superior 0.281 / 0.6768; entropía normalizada 0.7954
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6188; con |Δu| 0.7292

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4892; solape del 5 % superior 0.7391; masa de error en el 5 % superior del residuo 0.2657; masa de residuo en el 5 % superior del error 0.5614
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.693 (por parámetro 0.0003386); número de condición 1.178e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.02194
- adam_loss_end: 0.002448
- adam_loss_start: 0.002892
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.0156
- adam_seconds: 71.35
- adam_steps: 2000
- lbfgs_loss_end: 0.001585
- lbfgs_loss_start: 0.002382
- lbfgs_n_logged: 10
- lbfgs_seconds: 47.68
- lbfgs_to_adam_loss_ratio: 0.6475
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
