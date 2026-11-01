# Ficha de diagnóstico — burgers-close-20261003__rbfw48__s303__lbfgs_final__8dc10b8f

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 303 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/rbf48-seed-303/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.000279, u = 0.000279
- Error normalizado (RMS, todas las componentes): 0.000279; plano: 0.000279 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.5715; masa en el 5 % superior 0.9449; entropía normalizada 0.5126

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9733 | 1.438 | 0.8719 | 1.288 |
| steep5 | 0.05004 | 0.9325 | 18.63 | 0.8565 | 17.12 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.001731; p95 0.001575; máx 0.08109
- Concentración: Gini 0.6187; masa en el 1 % / 5 % superior 0.8355 / 0.9299; entropía normalizada 0.5207
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1864; con |Δu| 0.3389

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.06001; solape del 5 % superior 0.2886; masa de error en el 5 % superior del residuo 0.7234; masa de residuo en el 5 % superior del error 0.7609
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.8 (por parámetro 0.0005038); número de condición 7.127e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.153 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.06581 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.007401
- adam_loss_end: 9.245e-05
- adam_loss_start: 0.2488
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.8348
- adam_seconds: 806.9
- adam_steps: 2e+04
- lbfgs_iterations: 3110
- lbfgs_seconds: 179.3
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
