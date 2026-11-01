# Ficha de diagnóstico — ac-rerun-20260930__rbfw16__s603__adam_final__aa3a74f3

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** rbf (ancho 16) · **Semilla:** 603 · **Etapa:** adam_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/rbf16-seed-603/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0233, u = 0.0233
- Error normalizado (RMS, todas las componentes): 0.0233; plano: 0.0233 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.9037; masa en el 5 % superior 0.9857; entropía normalizada 0.6168

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9998 | 1.627 | 0.613 | 0.9974 |
| steep5 | 0.05005 | 0.000202 | 0.004036 | 0.05177 | 1.034 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0005382; p95 0.0009966; máx 0.005942
- Concentración: Gini 0.474; masa en el 1 % / 5 % superior 0.2716 / 0.5101; entropía normalizada 0.8596
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.1411; con |Δu| 0.02126

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2537; solape del 5 % superior 0.06897; masa de error en el 5 % superior del residuo 0.2119; masa de residuo en el 5 % superior del error 0.08402
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 46.49 (por parámetro 0.01274); número de condición 4.502e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 9.915 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.2411 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.1855
- adam_loss_end: 9.329e-06
- adam_loss_start: 0.09492
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.9456
- adam_seconds: 771.6
- adam_steps: 4e+04
- lbfgs_iterations: 2412
- lbfgs_seconds: 85.54
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
