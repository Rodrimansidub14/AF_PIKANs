# Ficha de diagnóstico — cohort-A-lbfgs__rbf__s304__lbfgs_final__50cefb8b

**Problema:** dfg_2d · **Campaña:** cohort-A-lbfgs · **Brazo:** rbf · **Semilla:** 304 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261/training/seed-304/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.07413, p = 0.2202, u = 0.06975, v = 0.07344
- Error normalizado (RMS, todas las componentes): 0.2424; plano: 0.07413 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.1935; masa en el 5 % superior 0.1989; entropía normalizada 0.9708

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1444 | 2.144 | 0.8084 | 12 |
| steep5 | 0.05009 | 0.1325 | 2.645 | 0.7562 | 15.1 |
| wake | 0.15 | 0.1798 | 1.199 | 0.0189 | 0.1261 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.05411; p95 0.05998; máx 2.625
- Concentración: Gini 0.548; masa en el 1 % / 5 % superior 0.7846 / 0.8904; entropía normalizada 0.4482
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2159; con |Δu| 0.1115

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2057; solape del 5 % superior 0.4913; masa de error en el 5 % superior del residuo 0.1359; masa de residuo en el 5 % superior del error 0.7067
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 51.08 (por parámetro 0.01267); número de condición 4.519e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3691 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7563 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
