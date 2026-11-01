# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s803__adam_final__b2c04b9d

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 803 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-mlp-seed-803/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.9639, p = 1.004, u = 0.963, v = 1.001
- Error normalizado (RMS, todas las componentes): 1.714; plano: 0.9639 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2945; masa en el 5 % superior 0.3449; entropía normalizada 0.9199

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3197 | 4.746 | 0.2199 | 3.264 |
| steep5 | 0.05009 | 0.2923 | 5.835 | 0.2327 | 4.645 |
| wake | 0.15 | 0.1614 | 1.076 | 0.0109 | 0.07269 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.00528; p95 0.01405; máx 0.03069
- Concentración: Gini 0.5231; masa en el 1 % / 5 % superior 0.2306 / 0.6819; entropía normalizada 0.7892
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4274; con |Δu| 0.3584

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2522; solape del 5 % superior 0.5239; masa de error en el 5 % superior del residuo 0.1718; masa de residuo en el 5 % superior del error 0.4366
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.353 (por parámetro 0.0002958); número de condición 1.406e+17

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
