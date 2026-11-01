# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s803__adam_final__dfa3d200

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 803 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-rbf-seed-803/checkpoint-adam-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5244, p = 0.7295, u = 0.5185, v = 0.735
- Error normalizado (RMS, todas las componentes): 1.158; plano: 0.5244 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3447; masa en el 5 % superior 0.4961; entropía normalizada 0.8674

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.4098 | 6.082 | 0.5434 | 8.066 |
| steep5 | 0.05009 | 0.3903 | 7.791 | 0.4659 | 9.301 |
| wake | 0.15 | 0.106 | 0.7066 | 0.02281 | 0.1521 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01552; p95 0.03408; máx 0.1315
- Concentración: Gini 0.6133; masa en el 1 % / 5 % superior 0.3519 / 0.7795; entropía normalizada 0.7446
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.413; con |Δu| 0.4934

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4123; solape del 5 % superior 0.613; masa de error en el 5 % superior del residuo 0.3569; masa de residuo en el 5 % superior del error 0.6108
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 16.24 (por parámetro 0.004029); número de condición 1.627e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.296 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.47 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.006281
- adam_loss_end: 0.0009996
- adam_loss_start: 0.002698
- adam_n_logged: 20
- adam_rel_gain_last20pct: 0.05528
- adam_seconds: 249.8
- adam_steps: 2000
- lbfgs_loss_end: 0.0008824
- lbfgs_loss_start: 0.0009905
- lbfgs_n_logged: 10
- lbfgs_seconds: 136.8
- lbfgs_to_adam_loss_ratio: 0.8827
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_complete
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- (ninguno)
