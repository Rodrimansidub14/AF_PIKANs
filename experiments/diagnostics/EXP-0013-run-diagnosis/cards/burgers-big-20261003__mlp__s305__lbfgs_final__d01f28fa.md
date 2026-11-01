# Ficha de diagnóstico — burgers-big-20261003__mlp__s305__lbfgs_final__d01f28fa

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** mlp · **Semilla:** 305 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/mlp-seed-305/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 8.447e-05, u = 8.447e-05
- Error normalizado (RMS, todas las componentes): 8.447e-05; plano: 8.447e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4411; masa en el 5 % superior 0.3992; entropía normalizada 0.8846

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.6896 | 1.019 | 0.6637 | 0.9804 |
| steep5 | 0.05004 | 0.1806 | 3.609 | 0.6954 | 13.9 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0006098; p95 0.0008087; máx 0.01686
- Concentración: Gini 0.5436; masa en el 1 % / 5 % superior 0.6624 / 0.7992; entropía normalizada 0.6384
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.199; con |Δu| 0.2853

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.03547; solape del 5 % superior 0.1856; masa de error en el 5 % superior del residuo 0.1552; masa de residuo en el 5 % superior del error 0.4606
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.492 (por parámetro 0.0002081); número de condición 5.457e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.03397
- adam_loss_end: 0.0001756
- adam_loss_start: 0.4175
- adam_n_logged: 240
- adam_rel_gain_last20pct: 0.9064
- adam_seconds: 489.7
- adam_steps: 2.4e+04
- lbfgs_iterations: 7550
- lbfgs_seconds: 1813
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
