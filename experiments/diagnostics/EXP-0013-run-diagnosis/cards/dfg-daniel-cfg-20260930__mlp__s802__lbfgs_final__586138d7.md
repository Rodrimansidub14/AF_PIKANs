# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s802__lbfgs_final__586138d7

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 802 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-802/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9095, p = 1.041, u = 0.9063, v = 1.031
- Error normalizado (RMS, todas las componentes): 1.723; plano: 0.9095 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3064; masa en el 5 % superior 0.3363; entropía normalizada 0.9197

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2954 | 4.384 | 0.2274 | 3.376 |
| steep5 | 0.05009 | 0.2685 | 5.36 | 0.241 | 4.812 |
| wake | 0.15 | 0.1587 | 1.058 | 0.04063 | 0.271 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.006118; p95 0.01388; máx 0.06038
- Concentración: Gini 0.4788; masa en el 1 % / 5 % superior 0.2496 / 0.5607; entropía normalizada 0.8301
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.715; con |Δu| 0.5204

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.266; solape del 5 % superior 0.4848; masa de error en el 5 % superior del residuo 0.1845; masa de residuo en el 5 % superior del error 0.3955
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.74 (por parámetro 0.0003445); número de condición 1.218e+17

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
