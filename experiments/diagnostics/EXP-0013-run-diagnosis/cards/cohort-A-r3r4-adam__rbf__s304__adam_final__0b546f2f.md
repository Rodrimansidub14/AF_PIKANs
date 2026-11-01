# Ficha de diagnóstico — cohort-A-r3r4-adam__rbf__s304__adam_final__0b546f2f

**Problema:** dfg_2d · **Campaña:** cohort-A-r3r4-adam · **Brazo:** rbf · **Semilla:** 304 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-close-2026-09-27/dfg-re20-gpu-r4/download-attempts/attempt-01/pikan-c349281ede3181b8/runs/seed-304/dfg_rbf_pikan.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1952, p = 0.3901, u = 0.1912, v = 0.1661
- Error normalizado (RMS, todas las componentes): 0.4651; plano: 0.1952 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2504; masa en el 5 % superior 0.3095; entropía normalizada 0.9397

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2106 | 3.126 | 0.7679 | 11.4 |
| steep5 | 0.05009 | 0.1993 | 3.979 | 0.7443 | 14.86 |
| wake | 0.15 | 0.1519 | 1.013 | 0.01342 | 0.08951 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08217; p95 0.1249; máx 3.819
- Concentración: Gini 0.6202; masa en el 1 % / 5 % superior 0.7276 / 0.8929; entropía normalizada 0.4765
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2688; con |Δu| 0.1686

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3478; solape del 5 % superior 0.5848; masa de error en el 5 % superior del residuo 0.2412; masa de residuo en el 5 % superior del error 0.7663
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 51.05 (por parámetro 0.01266); número de condición 3.794e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3907 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.8219 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- meta_stop_reason: time_limit

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
