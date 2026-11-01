# Ficha de diagnóstico — burgers-big-20261003__rbfw48__s302__lbfgs_final__45f8f983

**Problema:** burgers_1d · **Campaña:** burgers-big-20261003 · **Brazo:** rbf (ancho 48) · **Semilla:** 302 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-big-20261003/rbf48-seed-302/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 29376 · **Precisión nativa:** float64 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 5.682e-05, u = 5.682e-05
- Error normalizado (RMS, todas las componentes): 5.682e-05; plano: 5.682e-05 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.4659; masa en el 5 % superior 0.5187; entropía normalizada 0.8137

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.7304 | 1.079 | 0.5357 | 0.7914 |
| steep5 | 0.05004 | 0.3844 | 7.682 | 0.4712 | 9.416 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0005575; p95 0.0008413; máx 0.02573
- Concentración: Gini 0.5588; masa en el 1 % / 5 % superior 0.5924 / 0.7786; entropía normalizada 0.6744
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.177; con |Δu| 0.2659

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.01619; solape del 5 % superior 0.1725; masa de error en el 5 % superior del residuo 0.3247; masa de residuo en el 5 % superior del error 0.4001
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 15.63 (por parámetro 0.0005322); número de condición 6.548e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 3.162 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.0658 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2544
- adam_loss_end: 0.004286
- adam_loss_start: 0.2309
- adam_n_logged: 240
- adam_rel_gain_last20pct: -2.745
- adam_seconds: 1879
- adam_steps: 2.4e+04
- lbfgs_iterations: 5700
- lbfgs_seconds: 1811
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_time_cap
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- budget_limited, concentrated_error, low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
