# Ficha de diagnóstico — ac-rerun-20260930__rbfw16__s602__adam_final__c1017b8c

**Problema:** allen_cahn_1d · **Campaña:** ac-rerun-20260930 · **Brazo:** rbf (ancho 16) · **Semilla:** 602 · **Etapa:** adam_final
**Checkpoint:** `experiments/allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints/ac-rerun-20260930/ac/rbf16-seed-602/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 3648 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.02739, u = 0.02739
- Error normalizado (RMS, todas las componentes): 0.02739; plano: 0.02739 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.9028; masa en el 5 % superior 0.9854; entropía normalizada 0.6122

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| interface | 0.6146 | 0.9997 | 1.627 | 0.5483 | 0.8922 |
| steep5 | 0.05005 | 0.0001173 | 0.002344 | 0.03792 | 0.7578 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.0006434; p95 0.001319; máx 0.005047
- Concentración: Gini 0.5071; masa en el 1 % / 5 % superior 0.2535 / 0.52; entropía normalizada 0.8541
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.2145; con |Δu| 0.000513

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3085; solape del 5 % superior 0.03744; masa de error en el 5 % superior del residuo 0.02923; masa de residuo en el 5 % superior del error 0.04457
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 46.48 (por parámetro 0.01274); número de condición 4.18e+301
- Bases (capa 0): efectivas 10.74; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_t = 10.04 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_x = 0.2412 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

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

- concentrated_error, plateau_adam

## 8. Límites y datos ausentes

- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
