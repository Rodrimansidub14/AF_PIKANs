# Ficha de diagnóstico — cohort-A-lbfgs__rbf__s301__lbfgs_final__19e05215

**Problema:** dfg_2d · **Campaña:** cohort-A-lbfgs · **Brazo:** rbf · **Semilla:** 301 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-rbf-current-cohort-refinement/dfg-current-cohort-refinement/run-1790606331368753261/training/seed-301/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_A (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1088, p = 0.3205, u = 0.1026, v = 0.09766
- Error normalizado (RMS, todas las componentes): 0.3504; plano: 0.1088 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2147; masa en el 5 % superior 0.1989; entropía normalizada 0.9676

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1528 | 2.269 | 0.8062 | 11.97 |
| steep5 | 0.05009 | 0.1381 | 2.757 | 0.7815 | 15.6 |
| wake | 0.15 | 0.1867 | 1.245 | 0.02385 | 0.159 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.06832; p95 0.07145; máx 3.867
- Concentración: Gini 0.5181; masa en el 1 % / 5 % superior 0.805 / 0.8997; entropía normalizada 0.3884
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1826; con |Δu| 0.1109

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2997; solape del 5 % superior 0.5522; masa de error en el 5 % superior del residuo 0.1475; masa de residuo en el 5 % superior del error 0.7898
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 52.79 (por parámetro 0.01309); número de condición 3.59e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3733 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7343 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
