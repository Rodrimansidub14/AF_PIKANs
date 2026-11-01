# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s802__lbfgs_final__c38ef6a8

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 802 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/literal-rbf-seed-802/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.567, p = 0.8744, u = 0.5601, v = 0.6771
- Error normalizado (RMS, todas las componentes): 1.24; plano: 0.567 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3101; masa en el 5 % superior 0.4203; entropía normalizada 0.896

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3201 | 4.751 | 0.6035 | 8.958 |
| steep5 | 0.05009 | 0.318 | 6.347 | 0.5696 | 11.37 |
| wake | 0.15 | 0.1181 | 0.7876 | 0.01419 | 0.09465 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01541; p95 0.03282; máx 0.1582
- Concentración: Gini 0.6388; masa en el 1 % / 5 % superior 0.3996 / 0.8021; entropía normalizada 0.7235
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2592; con |Δu| 0.5989

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5744; solape del 5 % superior 0.5348; masa de error en el 5 % superior del residuo 0.3192; masa de residuo en el 5 % superior del error 0.622
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 16.44 (por parámetro 0.004077); número de condición 3.276e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.16 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.053 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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

- concentrated_error, residual_aligned

## 8. Límites y datos ausentes

- (ninguno)
