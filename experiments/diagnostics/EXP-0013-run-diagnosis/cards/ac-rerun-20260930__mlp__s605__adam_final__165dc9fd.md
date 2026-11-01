# Ficha de diagnóstico — ac-rerun-20260930__mlp__s605__adam_final__165dc9fd

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** mlp · **Semilla:** 605 · **Etapa:** adam_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/mlp-seed-605/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 921 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1441, u = 0.1441
- Error normalizado (RMS, todas las componentes): 0.1441; plano: 0.1441 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.8285; masa en el 5 % superior 0.8911; entropía normalizada 0.7211

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9983 | 1.624 | 0.8639 | 1.406 |
| steep5 | 0.05005 | 0.004989 | 0.09969 | 0.05487 | 1.096 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0045; p95 0.009738; máx 0.02872
- Concentración: Gini 0.5426; masa en el 1 % / 5 % superior 0.2296 / 0.5326; entropía normalizada 0.8505
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1498; con |Δu| 0.1001

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4873; solape del 5 % superior 0; masa de error en el 5 % superior del residuo 0.00186; masa de residuo en el 5 % superior del error 0.03564
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 18.7 (por parámetro 0.02031); número de condición 4.779e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.007776
- adam_loss_end: 0.0006437
- adam_loss_start: 0.7773
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.4473
- adam_seconds: 340.4
- adam_steps: 4e+04
- lbfgs_iterations: 1.019e+04
- lbfgs_seconds: 229.8
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
