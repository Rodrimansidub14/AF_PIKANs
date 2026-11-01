# Ficha de diagnóstico — cohort-A-r3r4-adam__rbf__s302__adam_final__cf0029b0

**Problema:** dfg_2d · **Campaña:** cohort-A-r3r4-adam · **Brazo:** rbf · **Semilla:** 302 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-close-2026-09-27/dfg-re20-gpu-r3/download-attempts/attempt-01/pikan-682b323105591987/runs/seed-302/dfg_rbf_pikan.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2236, p = 0.4376, u = 0.2192, v = 0.1896
- Error normalizado (RMS, todas las componentes): 0.5248; plano: 0.2236 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2515; masa en el 5 % superior 0.2919; entropía normalizada 0.9433

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1933 | 2.869 | 0.6555 | 9.73 |
| steep5 | 0.05009 | 0.1832 | 3.657 | 0.6135 | 12.25 |
| wake | 0.15 | 0.1691 | 1.128 | 0.03498 | 0.2333 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08044; p95 0.1526; máx 2.186
- Concentración: Gini 0.6315; masa en el 1 % / 5 % superior 0.5325 / 0.8204; entropía normalizada 0.6657
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1756; con |Δu| 0.3246

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.489; solape del 5 % superior 0.5522; masa de error en el 5 % superior del residuo 0.2158; masa de residuo en el 5 % superior del error 0.4908
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 52.21 (por parámetro 0.01295); número de condición 3.708e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3995 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7927 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- meta_stop_reason: time_limit

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
