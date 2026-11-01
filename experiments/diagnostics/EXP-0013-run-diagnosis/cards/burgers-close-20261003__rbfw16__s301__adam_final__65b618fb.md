# Ficha de diagnóstico — burgers-close-20261003__rbfw16__s301__adam_final__65b618fb

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 16) · **Semilla:** 301 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf16-seed-301/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.005372, u = 0.005372
- Error normalizado (RMS, todas las componentes): 0.005372; plano: 0.005372 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.6471; masa en el 5 % superior 0.9437; entropía normalizada 0.525

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9747 | 1.44 | 0.9731 | 1.438 |
| steep5 | 0.05004 | 0.9259 | 18.5 | 0.9202 | 18.39 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01425; p95 0.009183; máx 0.4016
- Concentración: Gini 0.6449; masa en el 1 % / 5 % superior 0.8833 / 0.9581; entropía normalizada 0.5172
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.006094; con |Δu| 0.3201

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3305; solape del 5 % superior 0.4659; masa de error en el 5 % superior del residuo 0.8796; masa de residuo en el 5 % superior del error 0.8642
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 18.66 (por parámetro 0.005115); número de condición 4.946e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.215 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06639 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.001493
- adam_loss_end: 0.000642
- adam_loss_start: 0.293
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.6842
- adam_seconds: 382
- adam_steps: 2e+04
- lbfgs_iterations: 4559
- lbfgs_seconds: 158.3
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
