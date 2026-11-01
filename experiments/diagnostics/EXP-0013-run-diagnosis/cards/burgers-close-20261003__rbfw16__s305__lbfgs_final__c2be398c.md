# Ficha de diagnóstico — burgers-close-20261003__rbfw16__s305__lbfgs_final__c2be398c

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 16) · **Semilla:** 305 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf16-seed-305/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.00622, u = 0.00622
- Error normalizado (RMS, todas las componentes): 0.00622; plano: 0.00622 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.898; masa en el 5 % superior 0.9996; entropía normalizada 0.4646

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9927 | 1.467 | 0.3224 | 0.4763 |
| steep5 | 0.05004 | 0.9995 | 19.97 | 0.989 | 19.76 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.009866; p95 0.002315; máx 0.593
- Concentración: Gini 0.726; masa en el 1 % / 5 % superior 0.9875 / 0.9953; entropía normalizada 0.3078
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1278; con |Δu| 0.2851

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1407; solape del 5 % superior 0.4419; masa de error en el 5 % superior del residuo 0.9143; masa de residuo en el 5 % superior del error 0.9864
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.02 (por parámetro 0.003844); número de condición 6.955e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.106 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06621 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.04859
- adam_loss_end: 0.001986
- adam_loss_start: 0.2943
- adam_n_logged: 200
- adam_rel_gain_last20pct: -0.8993
- adam_seconds: 363.3
- adam_steps: 2e+04
- lbfgs_iterations: 5340
- lbfgs_seconds: 177.7
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
