# Ficha de diagnóstico — burgers-close-20261003__rbfw16__s301__lbfgs_final__3c478fde

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 16) · **Semilla:** 301 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf16-seed-301/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 9.675e-05, u = 9.675e-05
- Error normalizado (RMS, todas las componentes): 9.675e-05; plano: 9.675e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4483; masa en el 5 % superior 0.4006; entropía normalizada 0.8801

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.7644 | 1.129 | 0.7832 | 1.157 |
| steep5 | 0.05004 | 0.2377 | 4.75 | 0.7186 | 14.36 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.001236; p95 0.001478; máx 0.05865
- Concentración: Gini 0.5459; masa en el 1 % / 5 % superior 0.7172 / 0.8337; entropía normalizada 0.5926
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2269; con |Δu| 0.284

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.0492; solape del 5 % superior 0.2084; masa de error en el 5 % superior del residuo 0.1997; masa de residuo en el 5 % superior del error 0.5244
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.44 (por parámetro 0.00478); número de condición 5.288e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.162 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.0658 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
