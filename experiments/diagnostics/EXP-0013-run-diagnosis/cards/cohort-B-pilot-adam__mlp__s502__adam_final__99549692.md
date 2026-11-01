# Ficha de diagnóstico — cohort-B-pilot-adam__mlp__s502__adam_final__99549692

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** mlp · **Semilla:** 502 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-Tanh-MLP/seed-502/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.1686, p = 0.3721, u = 0.1641, v = 0.1153
- Error normalizado (RMS, todas las componentes): 0.4227; plano: 0.1686 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2441; masa en el 5 % superior 0.2814; entropía normalizada 0.9476

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.1916 | 2.844 | 0.6006 | 8.915 |
| steep5 | 0.05009 | 0.18 | 3.594 | 0.5989 | 11.96 |
| wake | 0.15 | 0.1642 | 1.095 | 0.01938 | 0.1293 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.05504; p95 0.1021; máx 2.459
- Concentración: Gini 0.5554; masa en el 1 % / 5 % superior 0.567 / 0.8157; entropía normalizada 0.6167
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.3055; con |Δu| 0.3582

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4538; solape del 5 % superior 0.5348; masa de error en el 5 % superior del residuo 0.2208; masa de residuo en el 5 % superior del error 0.6466
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 32.86 (por parámetro 0.00193); número de condición 7.115e+07

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
