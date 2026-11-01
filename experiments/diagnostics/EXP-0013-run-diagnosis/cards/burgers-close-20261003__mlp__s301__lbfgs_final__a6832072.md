# Ficha de diagnóstico — burgers-close-20261003__mlp__s301__lbfgs_final__a6832072

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** mlp · **Semilla:** 301 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/mlp-seed-301/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0002375, u = 0.0002375
- Error normalizado (RMS, todas las componentes): 0.0002375; plano: 0.0002375 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4999; masa en el 5 % superior 0.7888; entropía normalizada 0.6386

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9239 | 1.365 | 0.9634 | 1.423 |
| steep5 | 0.05004 | 0.722 | 14.43 | 0.9744 | 19.47 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.00278; p95 0.001128; máx 0.1361
- Concentración: Gini 0.6195; masa en el 1 % / 5 % superior 0.9686 / 0.9821; entropía normalizada 0.3472
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2851; con |Δu| 0.3528

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) -0.01592; solape del 5 % superior 0.182; masa de error en el 5 % superior del residuo 0.4831; masa de residuo en el 5 % superior del error 0.8073
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.256 (por parámetro 0.0002024); número de condición 5.604e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.08261
- adam_loss_end: 0.0008274
- adam_loss_start: 0.4189
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.1209
- adam_seconds: 215.1
- adam_steps: 2e+04
- lbfgs_iterations: 2355
- lbfgs_seconds: 59.07
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
