# Ficha de diagnóstico — cohort-B-pilot-adam__mlp__s501__adam_final__89871924

**Problema:** dfg_2d · **Campaña:** cohort-B-pilot-adam · **Brazo:** mlp · **Semilla:** 501 · **Etapa:** adam_final
**Checkpoint:** `var/agent_runs/baseline-competitive-2026-09-27/kaggle-cli-fetch-competitive-pilots/dfg/dfg-re20-t4x2/campaign/DFG-Tanh-MLP/seed-501/checkpoint-final.pt` · **Familia de modelo:** dfg_cohort_B (mlp) · **Parámetros:** 17027 · **Precisión nativa:** float32 (señales en float64)

> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.

## 1. Qué hizo el modelo (resultado, etiquetas)

- Error relativo L2 por componente: flat = 0.2838, p = 0.487, u = 0.2804, v = 0.2027
- Error normalizado (RMS, todas las componentes): 0.5974; plano: 0.2838 (la medida plana mezcla unidades en piezo)
- Concentración del error: Gini 0.2687; masa en el 5 % superior 0.3257; entropía normalizada 0.9327

## 2. Dónde falla (regiones, definidas con la referencia)

| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |
|---|---|---|---|---|---|
| near_cyl | 0.06737 | 0.2148 | 3.188 | 0.4802 | 7.127 |
| steep5 | 0.05009 | 0.2085 | 4.162 | 0.4768 | 9.519 |
| wake | 0.15 | 0.145 | 0.9673 | 0.007944 | 0.05298 |

## 3. Qué ve el residuo (señales sin referencia)

- Residuo RMS 0.07081; p95 0.163; máx 1.543
- Concentración: Gini 0.7153; masa en el 1 % / 5 % superior 0.4328 / 0.7877; entropía normalizada 0.6959
- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) 0.1708; con |Δu| 0.2092

## 4. Alineación residuo–error (auditoría con etiquetas)

- Spearman(|r|, error) 0.4393; solape del 5 % superior 0.6522; masa de error en el 5 % superior del residuo 0.2741; masa de residuo en el 5 % superior del error 0.683
- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.

## 5. Capacidad y optimización

- Rango efectivo del jacobiano 30.12 (por parámetro 0.001769); número de condición 6.138e+07

## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)

- low_jacobian_rank_per_param

## 8. Límites y datos ausentes

- `basis/*`: modelo sin bases RBF (MLP)
- `train/history`: sin archivo de historia de pérdida junto al checkpoint
