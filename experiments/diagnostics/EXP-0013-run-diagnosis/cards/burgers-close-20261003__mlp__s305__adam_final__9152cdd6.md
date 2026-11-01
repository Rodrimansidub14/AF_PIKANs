# Ficha de diagnóstico — burgers-close-20261003__mlp__s305__adam_final__9152cdd6

**Problema:** burgers_1d · **Campaña:** burgers-close-20261003 · **Brazo:** mlp · **Semilla:** 305 · **Etapa:** adam_final
**Checkpoint:** `experiments/burgers/EXP-0008-baseline-v2-competitive/checkpoints/burgers-close-20261003/mlp-seed-305/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (mlp) · **Parámetros:** 40801 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.02088, u = 0.02088
- Error normalizado (RMS, todas las componentes): 0.02088; plano: 0.02088 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.6286; masa en el 5 % superior 0.9071; entropía normalizada 0.5467

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| late_time | 0.6769 | 0.9859 | 1.456 | 0.949 | 1.402 |
| steep5 | 0.05004 | 0.8886 | 17.76 | 0.8846 | 17.68 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.02035; p95 0.0156; máx 0.5601
- Concentración: Gini 0.5205; masa en el 1 % / 5 % superior 0.874 / 0.9105; entropía normalizada 0.5139
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) -0.01614; con |Δu| 0.1894

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4679; solape del 5 % superior 0.3365; masa de error en el 5 % superior del residuo 0.8841; masa de residuo en el 5 % superior del error 0.8836
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 8.797 (por parámetro 0.0002156); número de condición 5.219e+301

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.2396
- adam_loss_end: 0.002006
- adam_loss_start: 0.4175
- adam_n_logged: 200
- adam_rel_gain_last20pct: -11.89
- adam_seconds: 228.8
- adam_steps: 2e+04
- lbfgs_iterations: 2233
- lbfgs_seconds: 58.07
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_lbfgs_status: lbfgs_plateau
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- concentrated_error, low_jacobian_rank_per_param, plateau_adam

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `gradterm/*`: un solo término residual (IC/BC por transformación dura): no hay conflicto entre términos para medir
