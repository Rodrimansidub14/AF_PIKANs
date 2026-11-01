# Ficha de diagnóstico — burgers-close-20261003__rbfw16__s302__adam_final__62c84f9c

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 16) · **Semilla:** 302 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf16-seed-302/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.002103, u = 0.002103
- Error normalizado (RMS, todas las componentes): 0.002103; plano: 0.002103 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5631; masa en el 5 % superior 0.8177; entropía normalizada 0.6616

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.8685 | 1.283 | 0.9005 | 1.33 |
| steep5 | 0.05004 | 0.6968 | 13.92 | 0.8161 | 16.31 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0082; p95 0.009877; máx 0.2103
- Concentración: Gini 0.6304; masa en el 1 % / 5 % superior 0.7169 / 0.8965; entropía normalizada 0.6308
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.07968; con |Δu| 0.3746

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3022; solape del 5 % superior 0.4922; masa de error en el 5 % superior del residuo 0.6963; masa de residuo en el 5 % superior del error 0.7536
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.86 (por parámetro 0.004896); número de condición 5.447e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.182 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06614 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.08093
- adam_loss_end: 0.0002075
- adam_loss_start: 0.2907
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.05496
- adam_seconds: 378.4
- adam_steps: 2e+04
- lbfgs_iterations: 5359
- lbfgs_seconds: 187.2
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
