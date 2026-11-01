# Ficha de diagnóstico — cohort-A-r3r4-adam__rbf__s301__adam_final__6dafbb97

**Problema:** dfg_2d · **Campaña:** cohort-A-r3r4-adam · **Brazo:** rbf · **Semilla:** 301 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-close-2026-09-27/dfg-re20-gpu-r3/download-attempts/attempt-01/pikan-682b323105591987/runs/seed-301/dfg_rbf_pikan.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2778, p = 0.5198, u = 0.273, v = 0.2421
- Error normalizado (RMS, todas las componentes): 0.6351; plano: 0.2778 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2522; masa en el 5 % superior 0.309; entropía normalizada 0.9399

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2198 | 3.263 | 0.7231 | 10.73 |
| steep5 | 0.05009 | 0.207 | 4.132 | 0.685 | 13.67 |
| wake | 0.15 | 0.1601 | 1.068 | 0.01291 | 0.0861 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.1065; p95 0.1699; máx 5.264
- Concentración: Gini 0.6613; masa en el 1 % / 5 % superior 0.647 / 0.8909; entropía normalizada 0.5198
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3887; con |Δu| 0.3088

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.365; solape del 5 % superior 0.6826; masa de error en el 5 % superior del residuo 0.2432; masa de residuo en el 5 % superior del error 0.8158
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 52.35 (por parámetro 0.01298); número de condición 3.969e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3983 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7911 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- meta_stop_reason: time_limit

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
