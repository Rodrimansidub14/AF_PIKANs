# Ficha de diagnóstico — ac-rerun-20260930__mlp__s602__lbfgs_final__4d3432c3

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** mlp · **Semilla:** 602 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/mlp-seed-602/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 921 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.01215, u = 0.01215
- Error normalizado (RMS, todas las componentes): 0.01215; plano: 0.01215 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.8601; masa en el 5 % superior 0.9696; entropía normalizada 0.6418

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9995 | 1.626 | 0.707 | 1.15 |
| steep5 | 0.05005 | 0.009022 | 0.1803 | 0.2172 | 4.341 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0006134; p95 0.001177; máx 0.008792
- Concentración: Gini 0.4631; masa en el 1 % / 5 % superior 0.2076 / 0.4284; entropía normalizada 0.8808
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2432; con |Δu| 0.3297

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3546; solape del 5 % superior 0.2108; masa de error en el 5 % superior del residuo 0.2917; masa de residuo en el 5 % superior del error 0.1345
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 19.4 (por parámetro 0.02106); número de condición 7.834e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.02319
- adam_loss_end: 0.0006302
- adam_loss_start: 0.5908
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.2922
- adam_seconds: 346.8
- adam_steps: 4e+04
- lbfgs_iterations: 9652
- lbfgs_seconds: 214.2
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
