# Ficha de diagnóstico — cohort-A-r3r4-adam__rbf__s303__adam_final__2687b037

**Problema:** dfg_2d · **Campaña:** cohort-A-r3r4-adam · **Brazo:** rbf · **Semilla:** 303 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-close-2026-09-27/dfg-re20-gpu-r4/download-attempts/attempt-01/pikan-c349281ede3181b8/runs/seed-303/dfg_rbf_pikan.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2728, p = 0.5149, u = 0.2682, v = 0.2118
- Error normalizado (RMS, todas las componentes): 0.618; plano: 0.2728 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2608; masa en el 5 % superior 0.2971; entropía normalizada 0.9408

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2027 | 3.009 | 0.6335 | 9.403 |
| steep5 | 0.05009 | 0.1945 | 3.883 | 0.6169 | 12.32 |
| wake | 0.15 | 0.1669 | 1.113 | 0.01854 | 0.1237 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08875; p95 0.17; máx 2.749
- Concentración: Gini 0.6625; masa en el 1 % / 5 % superior 0.5591 / 0.8423; entropía normalizada 0.6239
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1508; con |Δu| 0.2547

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4497; solape del 5 % superior 0.6609; masa de error en el 5 % superior del residuo 0.2558; masa de residuo en el 5 % superior del error 0.6279
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 48.68 (por parámetro 0.01207); número de condición 5.556e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3934 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.8107 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- meta_stop_reason: time_limit

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
