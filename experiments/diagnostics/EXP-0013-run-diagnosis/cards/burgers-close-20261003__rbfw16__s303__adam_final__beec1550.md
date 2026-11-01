# Ficha de diagnóstico — burgers-close-20261003__rbfw16__s303__adam_final__beec1550

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 16) · **Semilla:** 303 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf16-seed-303/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.001302, u = 0.001302
- Error normalizado (RMS, todas las componentes): 0.001302; plano: 0.001302 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.6079; masa en el 5 % superior 0.8689; entropía normalizada 0.6285

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.7972 | 1.178 | 0.966 | 1.427 |
| steep5 | 0.05004 | 0.6696 | 13.38 | 0.9489 | 18.96 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01276; p95 0.0075; máx 0.8993
- Concentración: Gini 0.6201; masa en el 1 % / 5 % superior 0.9377 / 0.9684; entropía normalizada 0.3029
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.0937; con |Δu| 0.3414

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1962; solape del 5 % superior 0.3509; masa de error en el 5 % superior del residuo 0.5931; masa de residuo en el 5 % superior del error 0.9141
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 18.13 (por parámetro 0.00497); número de condición 4.633e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.21 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06594 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.01306
- adam_loss_end: 6.393e-05
- adam_loss_start: 0.2893
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.473
- adam_seconds: 367.6
- adam_steps: 2e+04
- lbfgs_iterations: 4158
- lbfgs_seconds: 139.6
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
