# Ficha de diagnóstico — burgers-close-20261003__mlp__s302__lbfgs_final__32d986ae

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** mlp · **Semilla:** 302 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/mlp-seed-302/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.001606, u = 0.001606
- Error normalizado (RMS, todas las componentes): 0.001606; plano: 0.001606 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7593; masa en el 5 % superior 0.9944; entropía normalizada 0.4594

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9975 | 1.474 | 0.9801 | 1.448 |
| steep5 | 0.05004 | 0.9918 | 19.82 | 0.991 | 19.8 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.005383; p95 0.001232; máx 0.391
- Concentración: Gini 0.6331; masa en el 1 % / 5 % superior 0.9897 / 0.9938; entropía normalizada 0.2134
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1655; con |Δu| 0.225

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.133; solape del 5 % superior 0.2731; masa de error en el 5 % superior del residuo 0.67; masa de residuo en el 5 % superior del error 0.9784
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 7.982 (por parámetro 0.0001956); número de condición 5.611e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01164
- adam_loss_end: 0.0001252
- adam_loss_start: 0.4179
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.9432
- adam_seconds: 229.3
- adam_steps: 2e+04
- lbfgs_iterations: 2403
- lbfgs_seconds: 62.15
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
