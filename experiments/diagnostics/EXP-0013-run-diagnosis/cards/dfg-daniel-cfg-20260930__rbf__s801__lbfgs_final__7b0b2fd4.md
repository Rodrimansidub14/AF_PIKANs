# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s801__lbfgs_final__7b0b2fd4

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 801 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-801/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5659, p = 0.8726, u = 0.5589, v = 0.6851
- Error normalizado (RMS, todas las componentes): 1.242; plano: 0.5659 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3118; masa en el 5 % superior 0.4239; entropía normalizada 0.8945

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3274 | 4.859 | 0.6264 | 9.298 |
| steep5 | 0.05009 | 0.3231 | 6.449 | 0.581 | 11.6 |
| wake | 0.15 | 0.1171 | 0.7811 | 0.01496 | 0.09976 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01518; p95 0.03095; máx 0.1669
- Concentración: Gini 0.6062; masa en el 1 % / 5 % superior 0.435 / 0.8137; entropía normalizada 0.7177
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3113; con |Δu| 0.5871

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5217; solape del 5 % superior 0.5087; masa de error en el 5 % superior del residuo 0.3225; masa de residuo en el 5 % superior del error 0.6223
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.71 (por parámetro 0.004394); número de condición 2.78e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.188 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.026 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01278
- adam_loss_end: 0.001252
- adam_loss_start: 0.002809
- adam_n_logged: 20
- adam_rel_gain_last20pct: -0.04181
- adam_seconds: 237.5
- adam_steps: 2000
- lbfgs_loss_end: 0.0009034
- lbfgs_loss_start: 0.001119
- lbfgs_n_logged: 10
- lbfgs_seconds: 131.5
- lbfgs_to_adam_loss_ratio: 0.7214
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, residual_aligned

## 8. Límites y datos ausentes

- (ninguno)
