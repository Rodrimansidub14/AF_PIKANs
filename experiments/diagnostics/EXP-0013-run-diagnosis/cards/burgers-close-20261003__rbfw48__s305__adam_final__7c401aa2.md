# Ficha de diagnóstico — burgers-close-20261003__rbfw48__s305__adam_final__7c401aa2

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 305 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf48-seed-305/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.02179, u = 0.02179
- Error normalizado (RMS, todas las componentes): 0.02179; plano: 0.02179 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.7211; masa en el 5 % superior 0.989; entropía normalizada 0.4457

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9986 | 1.475 | 0.9797 | 1.447 |
| steep5 | 0.05004 | 0.9861 | 19.7 | 0.9418 | 18.82 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01579; p95 0.009824; máx 0.5949
- Concentración: Gini 0.6139; masa en el 1 % / 5 % superior 0.9166 / 0.9619; entropía normalizada 0.4685
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.06238; con |Δu| 0.3204

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4497; solape del 5 % superior 0.412; masa de error en el 5 % superior del residuo 0.9836; masa de residuo en el 5 % superior del error 0.8656
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 16.28 (por parámetro 0.0005542); número de condición 6.681e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.193 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06651 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.09267
- adam_loss_end: 0.0005343
- adam_loss_start: 0.2543
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.3087
- adam_seconds: 806.8
- adam_steps: 2e+04
- lbfgs_iterations: 3410
- lbfgs_seconds: 193
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
