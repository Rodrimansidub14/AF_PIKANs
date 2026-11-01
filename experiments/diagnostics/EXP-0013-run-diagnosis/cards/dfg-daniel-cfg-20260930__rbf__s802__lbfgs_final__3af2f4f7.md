# Ficha de diagnóstico — dfg-daniel-cfg-20260930__rbf__s802__lbfgs_final__3af2f4f7

**Problema:** dfg_2d · **Campaña:** dfg-daniel-cfg-20260930 · **Brazo:** rbf · **Semilla:** 802 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/dfg-daniel-cfg-20260930/corrected-rbf-seed-802/checkpoint-lbfgs-final.pt` · **Familia de modelo:** dfg_daniel_recipe (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.5218, p = 0.7218, u = 0.5167, v = 0.6705
- Error normalizado (RMS, todas las componentes): 1.112; plano: 0.5218 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3428; masa en el 5 % superior 0.4767; entropía normalizada 0.8722

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.3631 | 5.39 | 0.6043 | 8.97 |
| steep5 | 0.05009 | 0.3633 | 7.252 | 0.5763 | 11.51 |
| wake | 0.15 | 0.1016 | 0.6778 | 0.01163 | 0.07754 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.01544; p95 0.03519; máx 0.1572
- Concentración: Gini 0.6576; masa en el 1 % / 5 % superior 0.3929 / 0.7963; entropía normalizada 0.7256
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.4543; con |Δu| 0.4861

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3347; solape del 5 % superior 0.4717; masa de error en el 5 % superior del residuo 0.3439; masa de residuo en el 5 % superior del error 0.5878
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 16.08 (por parámetro 0.003987); número de condición 2.99e+07
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 2.13 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 2.249 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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
