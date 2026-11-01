# Ficha de diagnóstico — cohort-B-pilot-adam__rbf__s501__adam_final__1718fa4b

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** rbf · **Semilla:** 501 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-RBF-PIKAN/seed-501/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2601, p = 0.5152, u = 0.255, v = 0.197
- Error normalizado (RMS, todas las componentes): 0.6077; plano: 0.2601 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2594; masa en el 5 % superior 0.2887; entropía normalizada 0.9435

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1939 | 2.878 | 0.6617 | 9.822 |
| steep5 | 0.05009 | 0.1854 | 3.7 | 0.6401 | 12.78 |
| wake | 0.15 | 0.1646 | 1.098 | 0.01911 | 0.1274 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.08669; p95 0.1587; máx 3.57
- Concentración: Gini 0.6158; masa en el 1 % / 5 % superior 0.5791 / 0.8318; entropía normalizada 0.5926
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.346; con |Δu| 0.335

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4019; solape del 5 % superior 0.5935; masa de error en el 5 % superior del residuo 0.2297; masa de residuo en el 5 % superior del error 0.6836
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 54.83 (por parámetro 0.0136); número de condición 4.044e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3992 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.8064 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
