# Ficha de diagnóstico — ns2d-cg-20260930__rbfw16__s701__adam_final__3bac6d41

**Problema:** ns2d_cg · **Campaña:** ns2d-cg-20260930 · **Brazo:** rbf (ancho 16) · **Semilla:** 701 · **Etapa:** adam_final
**Checkpoint:** `experiments/navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints/ns2d-cg-20260930/ns/rbf16-seed-701/checkpoint-adam-final.pt` · **Familia de modelo:** pikan_campaign (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.06833, p = 0.3437, u = 0.04206, v = 0.3823
- Error normalizado (RMS, todas las componentes): 0.5158; plano: 0.06833 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.3516; masa en el 5 % superior 0.4474; entropía normalizada 0.8936

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_corner | 0.171 | 0.2312 | 1.352 | 0.9838 | 5.752 |
| steep5 | 0.05004 | 0.08018 | 1.602 | 0.1826 | 3.649 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.02733; p95 0.02049; máx 0.4719
- Concentración: Gini 0.6777; masa en el 1 % / 5 % superior 0.84 / 0.9671; entropía normalizada 0.5636
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3443; con |Δu| 0.2543

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.04884; solape del 5 % superior 0.1767; masa de error en el 5 % superior del residuo 0.1252; masa de residuo en el 5 % superior del error 0.1655
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 33.09 (por parámetro 0.008207); número de condición 5.499e+06
- Bases (capa 0): efectivas 10.79; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 4.212 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 1.841 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 6. Señales de entrenamiento registradas

- adam_log_slope_last10: 0.002929
- adam_loss_end: 5.478e-05
- adam_loss_start: 0.008976
- adam_n_logged: 200
- adam_rel_gain_last20pct: 0.2154
- adam_seconds: 1155
- adam_steps: 2e+04
- meta_adam_status: adam_complete
- meta_dtype: float32
- meta_status: DRAFT_UNVERIFIED

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- (ninguno)
