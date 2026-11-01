# Ficha de diagnóstico — burgers-big-20261003__mlp__s301__adam_final__c156da7d

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** mlp · **Semilla:** 301 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/mlp-seed-301/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.00587, u = 0.00587
- Error normalizado (RMS, todas las componentes): 0.00587; plano: 0.00587 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5324; masa en el 5 % superior 0.4627; entropía normalizada 0.8276

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9642 | 1.424 | 0.9694 | 1.432 |
| steep5 | 0.05004 | 0.3121 | 6.236 | 0.9172 | 18.33 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01572; p95 0.008876; máx 0.3468
- Concentración: Gini 0.5349; masa en el 1 % / 5 % superior 0.9103 / 0.9386; entropía normalizada 0.5189
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.03826; con |Δu| 0.01515

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5504; solape del 5 % superior 0.1521; masa de error en el 5 % superior del residuo 0.3451; masa de residuo en el 5 % superior del error 0.8804
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.726 (por parámetro 0.0002139); número de condición 5.276e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.1473
- adam_loss_end: 0.0009335
- adam_loss_start: 0.4189
- adam_n_logged: 240
- adam_rel_gain_last20pct: -1.342
- adam_seconds: 485.8
- adam_steps: 2.4e+04
- lbfgs_iterations: 7450
- lbfgs_seconds: 1811
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, residual_aligned, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
