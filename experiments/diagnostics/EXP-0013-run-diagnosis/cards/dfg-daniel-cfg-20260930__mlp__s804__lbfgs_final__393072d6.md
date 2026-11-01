# Ficha de diagnóstico — dfg-daniel-cfg-20260930__mlp__s804__lbfgs_final__393072d6

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** mlp · **Semilla:** 804 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-mlp-seed-804/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (mlp) · **Parámetros:** 7953 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.8628, p = 1.031, u = 0.8587, v = 0.9995
- Error normalizado (RMS, todas las componentes): 1.673; plano: 0.8628 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3057; masa en el 5 % superior 0.3562; entropía normalizada 0.9144

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3169 | 4.704 | 0.1552 | 2.304 |
| steep5 | 0.05009 | 0.2903 | 5.796 | 0.1467 | 2.928 |
| wake | 0.15 | 0.1527 | 1.018 | 0.02221 | 0.1481 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005685; p95 0.01222; máx 0.03541
- Concentración: Gini 0.5426; masa en el 1 % / 5 % superior 0.308 / 0.7544; entropía normalizada 0.767
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.6018; con |Δu| 0.5772

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3206; solape del 5 % superior 0.6022; masa de error en el 5 % superior del residuo 0.2022; masa de residuo en el 5 % superior del error 0.5657
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 2.47 (por parámetro 0.0003106); número de condición 1.409e+17

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
