# Ficha de diagnóstico — dfg-long-20261001__mlp__s803__adam_final__7653ca3e

**Problema:** dfg_2d · **Campaña:** dfg-long-20261001 · **Brazo:** mlp · **Semilla:** 803 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-long-20261001/corrected-mlp-seed-803/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.6019, p = 0.7964, u = 0.5967, v = 0.7766
- Error normalizado (RMS, todas las componentes): 1.262; plano: 0.6019 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3095; masa en el 5 % superior 0.4263; entropía normalizada 0.8926

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3413 | 5.067 | 0.4923 | 7.308 |
| steep5 | 0.05009 | 0.3302 | 6.593 | 0.475 | 9.482 |
| wake | 0.15 | 0.1212 | 0.8085 | 0.05092 | 0.3396 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01729; p95 0.03928; máx 0.1224
- Concentración: Gini 0.5327; masa en el 1 % / 5 % superior 0.2865 / 0.6482; entropía normalizada 0.7982
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3622; con |Δu| 0.6113

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4812; solape del 5 % superior 0.5; masa de error en el 5 % superior del residuo 0.3248; masa de residuo en el 5 % superior del error 0.4688
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.24 (por parámetro 0.0009104); número de condición 2.8e+16

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

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
