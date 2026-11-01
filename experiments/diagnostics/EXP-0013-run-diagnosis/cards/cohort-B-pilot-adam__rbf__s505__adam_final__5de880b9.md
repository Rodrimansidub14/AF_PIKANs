# Ficha de diagnóstico — cohort-B-pilot-adam__rbf__s505__adam_final__5de880b9

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** rbf · **Semilla:** 505 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-RBF-PIKAN/seed-505/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2691, p = 0.5302, u = 0.2637, v = 0.2269
- Error normalizado (RMS, todas las componentes): 0.6341; plano: 0.2691 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.249; masa en el 5 % superior 0.279; entropía normalizada 0.946

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1906 | 2.829 | 0.6046 | 8.974 |
| steep5 | 0.05009 | 0.1855 | 3.704 | 0.5542 | 11.06 |
| wake | 0.15 | 0.1658 | 1.106 | 0.01775 | 0.1183 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08289; p95 0.1656; máx 2.146
- Concentración: Gini 0.6377; masa en el 1 % / 5 % superior 0.4995 / 0.8018; entropía normalizada 0.6704
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3016; con |Δu| 0.3083

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4773; solape del 5 % superior 0.613; masa de error en el 5 % superior del residuo 0.2141; masa de residuo en el 5 % superior del error 0.5286
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 51.83 (por parámetro 0.01286); número de condición 4.902e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.4015 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.8364 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
