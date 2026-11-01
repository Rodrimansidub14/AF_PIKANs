# Ficha de diagnóstico — ac-rerun-20260930__mlp__s603__adam_final__f22f9ea8

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** mlp · **Semilla:** 603 · **Etapa:** adam_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/mlp-seed-603/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 921 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1718, u = 0.1718
- Error normalizado (RMS, todas las componentes): 0.1718; plano: 0.1718 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7517; masa en el 5 % superior 0.6858; entropía normalizada 0.7908

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9754 | 1.587 | 0.8729 | 1.42 |
| steep5 | 0.05005 | 0.01727 | 0.345 | 0.105 | 2.099 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.008253; p95 0.01884; máx 0.04979
- Concentración: Gini 0.5386; masa en el 1 % / 5 % superior 0.1992 / 0.5189; entropía normalizada 0.856
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1032; con |Δu| 0.08759

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3733; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.0101; masa de residuo en el 5 % superior del error 0.009396
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.49 (por parámetro 0.01899); número de condición 4.848e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.00368
- adam_loss_end: 0.002188
- adam_loss_start: 0.5336
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.5005
- adam_seconds: 346.9
- adam_steps: 4e+04
- lbfgs_iterations: 1.34e+04
- lbfgs_seconds: 300.7
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
