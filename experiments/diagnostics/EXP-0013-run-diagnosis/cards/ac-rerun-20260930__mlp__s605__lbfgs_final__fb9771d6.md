# Ficha de diagnóstico — ac-rerun-20260930__mlp__s605__lbfgs_final__fb9771d6

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** mlp · **Semilla:** 605 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/mlp-seed-605/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 921 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0212, u = 0.0212
- Error normalizado (RMS, todas las componentes): 0.0212; plano: 0.0212 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.8977; masa en el 5 % superior 0.9893; entropía normalizada 0.6077

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9998 | 1.627 | 0.7697 | 1.252 |
| steep5 | 0.05005 | 0.001128 | 0.02254 | 0.1162 | 2.322 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0005319; p95 0.001052; máx 0.005571
- Concentración: Gini 0.4934; masa en el 1 % / 5 % superior 0.2444 / 0.5039; entropía normalizada 0.8618
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1429; con |Δu| 0.1776

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3337; solape del 5 % superior 0.1911; masa de error en el 5 % superior del residuo 0.3959; masa de residuo en el 5 % superior del error 0.1034
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 20.38 (por parámetro 0.02212); número de condición 6.211e+301

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
