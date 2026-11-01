# Ficha de diagnóstico — cohort-B-pilot-adam__rbf__s503__adam_final__c2d0c379

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** rbf · **Semilla:** 503 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-RBF-PIKAN/seed-503/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2952, p = 0.5502, u = 0.2903, v = 0.2393
- Error normalizado (RMS, todas las componentes): 0.6665; plano: 0.2952 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.26; masa en el 5 % superior 0.3008; entropía normalizada 0.9397

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.21 | 3.117 | 0.6375 | 9.463 |
| steep5 | 0.05009 | 0.2024 | 4.04 | 0.5929 | 11.84 |
| wake | 0.15 | 0.1564 | 1.043 | 0.02176 | 0.1451 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.09606; p95 0.1882; máx 3.173
- Concentración: Gini 0.6767; masa en el 1 % / 5 % superior 0.5106 / 0.8146; entropía normalizada 0.6552
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2498; con |Δu| 0.3944

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.5732; solape del 5 % superior 0.5717; masa de error en el 5 % superior del residuo 0.2356; masa de residuo en el 5 % superior del error 0.6654
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 48.97 (por parámetro 0.01215); número de condición 9.603e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3968 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.8154 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- residual_aligned

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
