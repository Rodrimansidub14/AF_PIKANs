# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s802__adam_final__e09e7ca6

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 802 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-802/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9695, p = 1.012, u = 0.9684, v = 1.017
- Error normalizado (RMS, todas las componentes): 1.731; plano: 0.9695 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2888; masa en el 5 % superior 0.3485; entropía normalizada 0.9205

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3263 | 4.843 | 0.201 | 2.984 |
| steep5 | 0.05009 | 0.2999 | 5.987 | 0.1924 | 3.84 |
| wake | 0.15 | 0.1638 | 1.092 | 0.06867 | 0.458 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.006234; p95 0.01394; máx 0.02347
- Concentración: Gini 0.4096; masa en el 1 % / 5 % superior 0.1174 / 0.4033; entropía normalizada 0.8915
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6252; con |Δu| 0.4578

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2695; solape del 5 % superior 0.3543; masa de error en el 5 % superior del residuo 0.1715; masa de residuo en el 5 % superior del error 0.2065
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.314 (por parámetro 0.000291); número de condición 1.568e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.004478
- adam_loss_end: 0.002525
- adam_loss_start: 0.00303
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.01614
- adam_seconds: 71.36
- adam_steps: 2000
- lbfgs_loss_end: 0.001657
- lbfgs_loss_start: 0.002479
- lbfgs_n_logged: 10
- lbfgs_seconds: 47.64
- lbfgs_to_adam_loss_ratio: 0.6561
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
