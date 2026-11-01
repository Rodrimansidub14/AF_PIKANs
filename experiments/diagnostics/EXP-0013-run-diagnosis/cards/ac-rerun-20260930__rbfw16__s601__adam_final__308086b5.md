# Ficha de diagnóstico — ac-rerun-20260930__rbfw16__s601__adam_final__308086b5

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** rbf (ancho 16) · **Semilla:** 601 · **Etapa:** adam_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/rbf16-seed-601/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.02815, u = 0.02815
- Error normalizado (RMS, todas las componentes): 0.02815; plano: 0.02815 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.8623; masa en el 5 % superior 0.9863; entropía normalizada 0.614

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9992 | 1.626 | 0.4382 | 0.7129 |
| steep5 | 0.05005 | 0.0007369 | 0.01473 | 0.0353 | 0.7053 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0009441; p95 0.001955; máx 0.004971
- Concentración: Gini 0.4567; masa en el 1 % / 5 % superior 0.1192 / 0.3377; entropía normalizada 0.9085
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.1976; con |Δu| 0.1143

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3931; solape del 5 % superior 0.02365; masa de error en el 5 % superior del residuo 0.08083; masa de residuo en el 5 % superior del error 0.04141
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 48.56 (por parámetro 0.01331); número de condición 3.931e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 9.936 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.2404 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.001596
- adam_loss_end: 2.327e-05
- adam_loss_start: 0.08161
- adam_n_logged: 400
- adam_rel_gain_last20pct: 0.2501
- adam_seconds: 801.4
- adam_steps: 4e+04
- lbfgs_iterations: 2716
- lbfgs_seconds: 99.79
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
