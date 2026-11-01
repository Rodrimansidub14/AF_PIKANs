# Ficha de diagnóstico — burgers-f64-lbfgs-20261003__mlp__s301__lbfgs_final__92bac6f0

**Problema:** burgers_1d · **Campaña:** burgers-f64-lbfgs-20261003 · **Brazo:** mlp · **Semilla:** 301 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-f64-lbfgs-20261003/mlp-seed-301/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0002049, u = 0.0002049
- Error normalizado (RMS, todas las componentes): 0.0002049; plano: 0.0002049 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5089; masa en el 5 % superior 0.8759; entropía normalizada 0.565

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.95 | 1.403 | 0.9652 | 1.426 |
| steep5 | 0.05004 | 0.8413 | 16.81 | 0.9809 | 19.6 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.002464; p95 0.0008833; máx 0.1185
- Concentración: Gini 0.6399; masa en el 1 % / 5 % superior 0.9771 / 0.9866; entropía normalizada 0.3384
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2379; con |Δu| 0.3032

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.02296; solape del 5 % superior 0.224; masa de error en el 5 % superior del residuo 0.6761; masa de residuo en el 5 % superior del error 0.8378
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.21 (por parámetro 0.0002012); número de condición 5.63e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.08261
- adam_loss_end: 0.0008274
- adam_loss_start: 0.4189
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.1209
- adam_seconds: 223.3
- adam_steps: 2e+04
- lbfgs_iterations: 6800
- lbfgs_seconds: 905.6
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
