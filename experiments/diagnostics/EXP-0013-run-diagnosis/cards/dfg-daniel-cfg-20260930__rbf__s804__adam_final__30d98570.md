# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s804__adam_final__30d98570

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 804 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-804/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5852, p = 0.8777, u = 0.5774, v = 0.8005
- Error normalizado (RMS, todas las componentes): 1.321; plano: 0.5852 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3251; masa en el 5 % superior 0.4511; entropía normalizada 0.886

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.38 | 5.641 | 0.4957 | 7.359 |
| steep5 | 0.05009 | 0.3544 | 7.076 | 0.3846 | 7.678 |
| wake | 0.15 | 0.1203 | 0.8023 | 0.05872 | 0.3916 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01428; p95 0.03286; máx 0.0906
- Concentración: Gini 0.5619; masa en el 1 % / 5 % superior 0.2804 / 0.662; entropía normalizada 0.789
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.396; con |Δu| 0.686

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4916; solape del 5 % superior 0.5609; masa de error en el 5 % superior del residuo 0.2897; masa de residuo en el 5 % superior del error 0.4888
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.1 (por parámetro 0.003497); número de condición 2.113e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.394 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.57 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.009538
- adam_loss_end: 0.001117
- adam_loss_start: 0.002627
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.05251
- adam_seconds: 237.9
- adam_steps: 2000
- lbfgs_loss_end: 0.0009451
- lbfgs_loss_start: 0.001082
- lbfgs_n_logged: 10
- lbfgs_seconds: 131.6
- lbfgs_to_adam_loss_ratio: 0.846
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
