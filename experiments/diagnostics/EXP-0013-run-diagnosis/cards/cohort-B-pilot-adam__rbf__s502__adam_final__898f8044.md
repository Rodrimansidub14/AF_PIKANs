# Ficha de diagnóstico — cohort-B-pilot-adam__rbf__s502__adam_final__898f8044

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** rbf · **Semilla:** 502 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-RBF-PIKAN/seed-502/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2489, p = 0.5193, u = 0.2432, v = 0.196
- Error normalizado (RMS, todas las componentes): 0.606; plano: 0.2489 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2528; masa en el 5 % superior 0.2911; entropía normalizada 0.9442

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1956 | 2.903 | 0.6602 | 9.799 |
| steep5 | 0.05009 | 0.1862 | 3.718 | 0.665 | 13.28 |
| wake | 0.15 | 0.1619 | 1.08 | 0.01308 | 0.08721 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08914; p95 0.1511; máx 3.221
- Concentración: Gini 0.6411; masa en el 1 % / 5 % superior 0.6426 / 0.8706; entropía normalizada 0.5756
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2921; con |Δu| 0.2378

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.3691; solape del 5 % superior 0.6087; masa de error en el 5 % superior del residuo 0.2423; masa de residuo en el 5 % superior del error 0.7518
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 51.96 (por parámetro 0.01289); número de condición 4.012e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3897 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.839 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
