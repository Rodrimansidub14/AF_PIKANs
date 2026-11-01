# Ficha de diagnóstico — cohort-A-r3r4-adam__rbf__s305__adam_final__84f86c90

**Problema:** dfg_2d · **Campaña:** cohort-A-r3r4-adam · **Brazo:** rbf · **Semilla:** 305 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-close-2026-09-27/dfg-re20-gpu-r4/download-attempts/attempt-01/pikan-c349281ede3181b8/runs/seed-305/dfg_rbf_pikan.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2536, p = 0.517, u = 0.248, v = 0.2141
- Error normalizado (RMS, todas las componentes): 0.612; plano: 0.2536 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2455; masa en el 5 % superior 0.2627; entropía normalizada 0.9511

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1857 | 2.756 | 0.6067 | 9.006 |
| steep5 | 0.05009 | 0.1713 | 3.421 | 0.5768 | 11.51 |
| wake | 0.15 | 0.1778 | 1.185 | 0.02018 | 0.1346 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08557; p95 0.1759; máx 1.947
- Concentración: Gini 0.6359; masa en el 1 % / 5 % superior 0.458 / 0.8157; entropía normalizada 0.6859
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2789; con |Δu| 0.2688

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4721; solape del 5 % superior 0.6543; masa de error en el 5 % superior del residuo 0.2115; masa de residuo en el 5 % superior del error 0.5655
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 55.58 (por parámetro 0.01378); número de condición 3.41e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3763 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.8236 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- meta_stop_reason: time_limit

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
