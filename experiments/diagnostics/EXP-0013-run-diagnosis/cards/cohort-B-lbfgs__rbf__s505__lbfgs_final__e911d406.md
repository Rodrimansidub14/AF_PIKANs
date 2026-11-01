# Ficha de diagnóstico — cohort-B-lbfgs__rbf__s505__lbfgs_final__e911d406

**Problema:** dfg_2d · **Campaña:** cohort-B-lbfgs · **Brazo:** rbf · **Semilla:** 505 · **Etapa:** lbfgs_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/training/DFG-RBF-PIKAN/seed-505/checkpoint-lbfgs.pt` · **Familia de modelo:** dfg_cohort_B (rbf) · **Parámetros:** 4032 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.0734, p = 0.2679, u = 0.06645, v = 0.07127
- Error normalizado (RMS, todas las componentes): 0.2851; plano: 0.0734 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2071; masa en el 5 % superior 0.1308; entropía normalizada 0.9762

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1098 | 1.629 | 0.7751 | 11.51 |
| steep5 | 0.05009 | 0.09125 | 1.822 | 0.6992 | 13.96 |
| wake | 0.15 | 0.2204 | 1.47 | 0.03196 | 0.2131 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.04206; p95 0.05327; máx 1.695
- Concentración: Gini 0.4757; masa en el 1 % / 5 % superior 0.7114 / 0.8417; entropía normalizada 0.5529
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.2533; con |Δu| 0.1854

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.2417; solape del 5 % superior 0.4717; masa de error en el 5 % superior del residuo 0.0987; masa de residuo en el 5 % superior del error 0.2543
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 49.92 (por parámetro 0.01238); número de condición 8.167e+05
- Bases (capa 0): efectivas 10.75; muertas 0; aristas pequeñas 0
- ratio_struct_to_spacing_x = 0.3632 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)
- ratio_struct_to_spacing_y = 0.7445 (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- ninguno de los marcadores se activa

## 8. Límites y datos ausentes

- `train/history`: sin archivo de historia de pérdida junto al checkpoint
