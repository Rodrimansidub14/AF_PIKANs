# Ficha de diagnóstico — ac-rerun-20260930__mlp__s602__adam_final__c7673eea

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** mlp · **Semilla:** 602 · **Etapa:** adam_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/mlp-seed-602/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 921 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1309, u = 0.1309
- Error normalizado (RMS, todas las componentes): 0.1309; plano: 0.1309 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.8063; masa en el 5 % superior 0.8715; entropía normalizada 0.7258

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9982 | 1.624 | 0.8131 | 1.323 |
| steep5 | 0.05005 | 0.04029 | 0.8049 | 0.1169 | 2.336 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.004457; p95 0.00897; máx 0.03872
- Concentración: Gini 0.5146; masa en el 1 % / 5 % superior 0.2017 / 0.4731; entropía normalizada 0.8699
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2066; con |Δu| 0.1646

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4644; solape del 5 % superior 0.04631; masa de error en el 5 % superior del residuo 0.02699; masa de residuo en el 5 % superior del error 0.05885
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 18.79 (por parámetro 0.02041); número de condición 4.994e+301

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
