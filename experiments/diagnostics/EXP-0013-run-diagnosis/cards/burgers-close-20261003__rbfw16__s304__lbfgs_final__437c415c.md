# Ficha de diagnóstico — burgers-close-20261003__rbfw16__s304__lbfgs_final__437c415c

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 16) · **Semilla:** 304 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf16-seed-304/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.001147, u = 0.001147
- Error normalizado (RMS, todas las componentes): 0.001147; plano: 0.001147 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7226; masa en el 5 % superior 0.9933; entropía normalizada 0.4626

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9965 | 1.472 | 0.8694 | 1.284 |
| steep5 | 0.05004 | 0.9917 | 19.82 | 0.9263 | 18.51 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.002905; p95 0.001819; máx 0.1228
- Concentración: Gini 0.64; masa en el 1 % / 5 % superior 0.9267 / 0.9657; entropía normalizada 0.4349
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.177; con |Δu| 0.3162

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.0784; solape del 5 % superior 0.3401; masa de error en el 5 % superior del residuo 0.7156; masa de residuo en el 5 % superior del error 0.9121
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.64 (por parámetro 0.004287); número de condición 6.574e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.15 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06582 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.06989
- adam_loss_end: 0.00142
- adam_loss_start: 0.289
- adam_n_logged: 200
- adam_rel_gain_last20pct: -1.919
- adam_seconds: 360.8
- adam_steps: 2e+04
- lbfgs_iterations: 4584
- lbfgs_seconds: 155
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
