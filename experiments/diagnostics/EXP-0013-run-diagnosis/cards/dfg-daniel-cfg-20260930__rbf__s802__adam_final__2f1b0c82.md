# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s802__adam_final__2f1b0c82

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 802 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-rbf-seed-802/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5309, p = 0.7259, u = 0.5246, v = 0.7835
- Error normalizado (RMS, todas las componentes): 1.19; plano: 0.5309 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3544; masa en el 5 % superior 0.5085; entropía normalizada 0.8622

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.4352 | 6.46 | 0.5325 | 7.905 |
| steep5 | 0.05009 | 0.4073 | 8.13 | 0.4142 | 8.268 |
| wake | 0.15 | 0.1077 | 0.7184 | 0.04993 | 0.333 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01422; p95 0.03454; máx 0.09018
- Concentración: Gini 0.6022; masa en el 1 % / 5 % superior 0.2858 / 0.6973; entropía normalizada 0.7756
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3476; con |Δu| 0.4709

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4209; solape del 5 % superior 0.613; masa de error en el 5 % superior del residuo 0.3306; masa de residuo en el 5 % superior del error 0.5461
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 14.08 (por parámetro 0.003492); número de condición 2.438e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.47 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.457 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.02094
- adam_loss_end: 0.001075
- adam_loss_start: 0.002699
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.2373
- adam_seconds: 249.6
- adam_steps: 2000
- lbfgs_loss_end: 0.0009128
- lbfgs_loss_start: 0.001069
- lbfgs_n_logged: 10
- lbfgs_seconds: 136.2
- lbfgs_to_adam_loss_ratio: 0.8495
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
