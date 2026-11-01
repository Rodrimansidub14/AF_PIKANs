# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s805__lbfgs_final__059c8638

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 805 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-805/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.8658, p = 0.9861, u = 0.8625, v = 1.008
- Error normalizado (RMS, todas las componentes): 1.653; plano: 0.8658 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3081; masa en el 5 % superior 0.3589; entropía normalizada 0.9135

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3177 | 4.716 | 0.2863 | 4.25 |
| steep5 | 0.05009 | 0.291 | 5.809 | 0.2378 | 4.748 |
| wake | 0.15 | 0.1526 | 1.018 | 0.02611 | 0.1742 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005516; p95 0.01287; máx 0.03376
- Concentración: Gini 0.4576; masa en el 1 % / 5 % superior 0.2429 / 0.6583; entropía normalizada 0.8115
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6908; con |Δu| 0.1321

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1614; solape del 5 % superior 0.7326; masa de error en el 5 % superior del residuo 0.2793; masa de residuo en el 5 % superior del error 0.5672
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.314 (por parámetro 0.000291); número de condición 1.44e+17

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
