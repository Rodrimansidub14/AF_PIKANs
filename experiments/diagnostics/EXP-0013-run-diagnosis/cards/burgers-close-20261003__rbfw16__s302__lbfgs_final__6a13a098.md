# Ficha de diagnóstico — burgers-close-20261003__rbfw16__s302__lbfgs_final__6a13a098

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 16) · **Semilla:** 302 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf16-seed-302/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0007214, u = 0.0007214
- Error normalizado (RMS, todas las componentes): 0.0007214; plano: 0.0007214 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7156; masa en el 5 % superior 0.9932; entropía normalizada 0.438

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9963 | 1.472 | 0.9568 | 1.413 |
| steep5 | 0.05004 | 0.9915 | 19.81 | 0.9495 | 18.97 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.002596; p95 0.001419; máx 0.1173
- Concentración: Gini 0.6412; masa en el 1 % / 5 % superior 0.9404 / 0.972; entropía normalizada 0.4142
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1423; con |Δu| 0.309

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.07613; solape del 5 % superior 0.3305; masa de error en el 5 % superior del residuo 0.9028; masa de residuo en el 5 % superior del error 0.909
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 16.09 (por parámetro 0.004411); número de condición 5.69e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.163 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06581 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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
