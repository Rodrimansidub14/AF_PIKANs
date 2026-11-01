# Ficha de diagnóstico — ac-rerun-20260930__rbfw16__s602__lbfgs_final__ec845b64

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** rbf (ancho 16) · **Semilla:** 602 · **Etapa:** lbfgs_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/rbf16-seed-602/checkpoint-lbfgs-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.003674, u = 0.003674
- Error normalizado (RMS, todas las componentes): 0.003674; plano: 0.003674 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.8657; masa en el 5 % superior 0.9871; entropía normalizada 0.5891

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9998 | 1.627 | 0.7297 | 1.187 |
| steep5 | 0.05005 | 0.0005348 | 0.01069 | 0.05623 | 1.124 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.000149; p95 0.000272; máx 0.001825
- Concentración: Gini 0.4963; masa en el 1 % / 5 % superior 0.3378 / 0.5865; entropía normalizada 0.8295
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.08337; con |Δu| 0.1261

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.1957; solape del 5 % superior 0.2069; masa de error en el 5 % superior del residuo 0.4422; masa de residuo en el 5 % superior del error 0.1932
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 42.88 (por parámetro 0.01175); número de condición 4.998e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 10.38 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.241 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: -0.05847
- adam_loss_end: 1.279e-05
- adam_loss_start: 0.07435
- adam_n_logged: 400
- adam_rel_gain_last20pct: -0.08867
- adam_seconds: 822.5
- adam_steps: 4e+04
- lbfgs_iterations: 3323
- lbfgs_seconds: 125.7
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
