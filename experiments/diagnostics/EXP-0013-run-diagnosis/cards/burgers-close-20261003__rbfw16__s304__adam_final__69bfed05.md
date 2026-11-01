# Ficha de diagnóstico — burgers-close-20261003__rbfw16__s304__adam_final__69bfed05

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 16) · **Semilla:** 304 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf16-seed-304/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.009561, u = 0.009561
- Error normalizado (RMS, todas las componentes): 0.009561; plano: 0.009561 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7045; masa en el 5 % superior 0.9659; entropía normalizada 0.512

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9907 | 1.463 | 0.9713 | 1.435 |
| steep5 | 0.05004 | 0.9268 | 18.52 | 0.7735 | 15.46 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.02226; p95 0.02379; máx 0.4649
- Concentración: Gini 0.7357; masa en el 1 % / 5 % superior 0.7651 / 0.9523; entropía normalizada 0.5911
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.1123; con |Δu| 0.3778

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3921; solape del 5 % superior 0.6024; masa de error en el 5 % superior del residuo 0.8538; masa de residuo en el 5 % superior del error 0.8696
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 16.57 (por parámetro 0.004542); número de condición 6.505e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.13 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06716 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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

- concentrated_error, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
