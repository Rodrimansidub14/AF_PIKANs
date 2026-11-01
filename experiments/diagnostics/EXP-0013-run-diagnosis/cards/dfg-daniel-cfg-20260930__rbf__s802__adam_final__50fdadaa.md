# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s802__adam_final__50fdadaa

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 802 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-802/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5579, p = 0.8823, u = 0.549, v = 0.8013
- Error normalizado (RMS, todas las componentes): 1.312; plano: 0.5579 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3282; masa en el 5 % superior 0.4596; entropía normalizada 0.8832

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.389 | 5.775 | 0.4226 | 6.274 |
| steep5 | 0.05009 | 0.3649 | 7.284 | 0.3412 | 6.811 |
| wake | 0.15 | 0.121 | 0.8071 | 0.09443 | 0.6298 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01579; p95 0.03181; máx 0.09403
- Concentración: Gini 0.4882; masa en el 1 % / 5 % superior 0.2531 / 0.5425; entropía normalizada 0.838
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3928; con |Δu| 0.6431

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.6358; solape del 5 % superior 0.5826; masa de error en el 5 % superior del residuo 0.2842; masa de residuo en el 5 % superior del error 0.435
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 13.78 (por parámetro 0.003417); número de condición 2.808e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.642 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.597 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.01218
- adam_loss_end: 0.001417
- adam_loss_start: 0.002685
- adam_n_logged: 20
- adam_rel_gain_last20pct: -0.2387
- adam_seconds: 237.6
- adam_steps: 2000
- lbfgs_loss_end: 0.0009146
- lbfgs_loss_start: 0.001085
- lbfgs_n_logged: 10
- lbfgs_seconds: 131.5
- lbfgs_to_adam_loss_ratio: 0.6453
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, residual_aligned, plateau_adam

## 8. Límites y datos ausentes

- (ninguno)
