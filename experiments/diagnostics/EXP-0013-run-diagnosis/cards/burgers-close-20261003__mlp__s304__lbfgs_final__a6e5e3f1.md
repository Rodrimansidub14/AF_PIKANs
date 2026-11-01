# Ficha de diagnóstico — burgers-close-20261003__mlp__s304__lbfgs_final__a6e5e3f1

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** mlp · **Semilla:** 304 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/mlp-seed-304/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0004502, u = 0.0004502
- Error normalizado (RMS, todas las componentes): 0.0004502; plano: 0.0004502 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.6063; masa en el 5 % superior 0.9507; entropía normalizada 0.5101

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9792 | 1.447 | 0.9352 | 1.382 |
| steep5 | 0.05004 | 0.9339 | 18.66 | 0.943 | 18.84 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.001623; p95 0.0009452; máx 0.1066
- Concentración: Gini 0.5939; masa en el 1 % / 5 % superior 0.932 / 0.9616; entropía normalizada 0.3462
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1887; con |Δu| 0.2715

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.121; solape del 5 % superior 0.2048; masa de error en el 5 % superior del residuo 0.7225; masa de residuo en el 5 % superior del error 0.8893
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.004 (por parámetro 0.0001962); número de condición 5.711e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.02195
- adam_loss_end: 0.0005161
- adam_loss_start: 0.4185
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.9251
- adam_seconds: 218.8
- adam_steps: 2e+04
- lbfgs_iterations: 2858
- lbfgs_seconds: 73.06
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
