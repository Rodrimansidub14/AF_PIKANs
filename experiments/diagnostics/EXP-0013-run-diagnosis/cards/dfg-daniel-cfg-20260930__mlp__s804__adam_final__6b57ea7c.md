# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s804__adam_final__6b57ea7c

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 804 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-804/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9461, p = 1.03, u = 0.9441, v = 1.012
- Error normalizado (RMS, todas las componentes): 1.726; plano: 0.9461 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.295; masa en el 5 % superior 0.3502; entropía normalizada 0.9188

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3275 | 4.861 | 0.1211 | 1.797 |
| steep5 | 0.05009 | 0.301 | 6.009 | 0.1367 | 2.728 |
| wake | 0.15 | 0.1616 | 1.078 | 0.0705 | 0.4702 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005705; p95 0.01052; máx 0.02994
- Concentración: Gini 0.3674; masa en el 1 % / 5 % superior 0.1654 / 0.4071; entropía normalizada 0.8975
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.5969; con |Δu| -0.6121

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3047; solape del 5 % superior 0.4804; masa de error en el 5 % superior del residuo 0.1782; masa de residuo en el 5 % superior del error 0.1979
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.437 (por parámetro 0.0003064); número de condición 1.435e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01081
- adam_loss_end: 0.002447
- adam_loss_start: 0.002858
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.04512
- adam_seconds: 74.52
- adam_steps: 2000
- lbfgs_loss_end: 0.001664
- lbfgs_loss_start: 0.002342
- lbfgs_n_logged: 10
- lbfgs_seconds: 48.15
- lbfgs_to_adam_loss_ratio: 0.6802
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
