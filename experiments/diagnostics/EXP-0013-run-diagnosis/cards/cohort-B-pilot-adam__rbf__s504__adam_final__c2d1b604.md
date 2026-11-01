# Ficha de diagnóstico — cohort-B-pilot-adam__rbf__s504__adam_final__c2d1b604

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** rbf · **Semilla:** 504 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-RBF-PIKAN/seed-504/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.23, p = 0.4751, u = 0.2248, v = 0.1834
- Error normalizado (RMS, todas las componentes): 0.5567; plano: 0.23 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2577; masa en el 5 % superior 0.2884; entropía normalizada 0.9441

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1973 | 2.929 | 0.7022 | 10.42 |
| steep5 | 0.05009 | 0.189 | 3.773 | 0.6772 | 13.52 |
| wake | 0.15 | 0.1653 | 1.103 | 0.01886 | 0.1258 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08359; p95 0.1478; máx 2.821
- Concentración: Gini 0.6394; masa en el 1 % / 5 % superior 0.6176 / 0.8702; entropía normalizada 0.5976
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3021; con |Δu| 0.222

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3548; solape del 5 % superior 0.5348; masa de error en el 5 % superior del residuo 0.2249; masa de residuo en el 5 % superior del error 0.6394
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 47.52 (por parámetro 0.01178); número de condición 4.676e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.4012 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.8133 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
