# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s802__adam_final__c7f9251e

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 802 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-802/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9696, p = 1.023, u = 0.9683, v = 1.016
- Error normalizado (RMS, todas las componentes): 1.737; plano: 0.9696 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2882; masa en el 5 % superior 0.3499; entropía normalizada 0.9204

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3277 | 4.865 | 0.2196 | 3.259 |
| steep5 | 0.05009 | 0.3018 | 6.024 | 0.2052 | 4.097 |
| wake | 0.15 | 0.163 | 1.087 | 0.07121 | 0.4749 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005772; p95 0.01336; máx 0.02178
- Concentración: Gini 0.4348; masa en el 1 % / 5 % superior 0.1208 / 0.4222; entropía normalizada 0.8835
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.61; con |Δu| 0.4642

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2925; solape del 5 % superior 0.3348; masa de error en el 5 % superior del residuo 0.1676; masa de residuo en el 5 % superior del error 0.2156
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.316 (por parámetro 0.0002912); número de condición 1.623e+17

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.004374
- adam_loss_end: 0.002541
- adam_loss_start: 0.003065
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.01727
- adam_seconds: 74.72
- adam_steps: 2000
- lbfgs_loss_end: 0.001623
- lbfgs_loss_start: 0.002493
- lbfgs_n_logged: 10
- lbfgs_seconds: 47.94
- lbfgs_to_adam_loss_ratio: 0.6387
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- null_state, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
