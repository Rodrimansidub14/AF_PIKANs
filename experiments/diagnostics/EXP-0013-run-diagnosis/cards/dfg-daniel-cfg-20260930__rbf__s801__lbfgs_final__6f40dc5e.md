# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s801__lbfgs_final__6f40dc5e

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 801 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-rbf-seed-801/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5239, p = 0.7443, u = 0.5183, v = 0.6802
- Error normalizado (RMS, todas las componentes): 1.134; plano: 0.5239 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3332; masa en el 5 % superior 0.4739; entropía normalizada 0.8749

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3646 | 5.413 | 0.624 | 9.262 |
| steep5 | 0.05009 | 0.3617 | 7.22 | 0.5768 | 11.52 |
| wake | 0.15 | 0.1013 | 0.6755 | 0.0131 | 0.08736 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01526; p95 0.03143; máx 0.1695
- Concentración: Gini 0.642; masa en el 1 % / 5 % superior 0.45 / 0.8199; entropía normalizada 0.7104
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3682; con |Δu| 0.428

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4155; solape del 5 % superior 0.4717; masa de error en el 5 % superior del residuo 0.3458; masa de residuo en el 5 % superior del error 0.6286
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 17.93 (por parámetro 0.004448); número de condición 2.271e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.162 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.133 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.02228
- adam_loss_end: 0.001109
- adam_loss_start: 0.002813
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.07046
- adam_seconds: 249.8
- adam_steps: 2000
- lbfgs_loss_end: 0.0008919
- lbfgs_loss_start: 0.001097
- lbfgs_n_logged: 10
- lbfgs_seconds: 136.6
- lbfgs_to_adam_loss_ratio: 0.804
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
