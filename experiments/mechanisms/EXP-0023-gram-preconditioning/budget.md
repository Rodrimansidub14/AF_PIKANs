# EXP-0023 — presupuesto (prueba previa 6, parte CPU) — 2026-10-20

Estado: **incompleto.** Medido solo en CPU (2 hilos, float32, N = 2 048 puntos por recálculo). La medición en GPU y la equivalencia CPU/GPU (prueba 7) no se hicieron: requieren lanzar en Kaggle y falta autorización. Datos: `results/budget_cpu_ac.json`, `results/budget_cpu_burgers.json` (`measure_budget.py`).

| CPU | s por paso de Adam (B) | s por recálculo J / E / I | recálculo en pasos de Adam |
|---|---|---|---|
| Allen–Cahn RBF16 | 0.55 | 24 / 26 / 30 | ≈ 45–55 |
| Burgers RBF48 | 1.73 | 173 / 153 / 165 | ≈ 90–100 |

- El sobrecosto por paso de la reparametrización no se distingue del ruido (la CPU estaba compartida; J salió más rápido que B). Orden de magnitud: ≤ 20 %.
- **Proyección a T4 (solo orden de magnitud, no es medición):** si la razón recálculo/paso se conservara, un recálculo costaría ≈ 1 s (AC, 0.021 s/paso) y ≈ 4 s (Burgers, 0.039 s/paso). La razón en GPU puede ser muy distinta.
- Recálculos por corrida: pasos 0, 1 000, 2 000, … y el arranque de L-BFGS. Control completo: Burgers 21, Allen–Cahn 41. Convención T1 (10 % de Adam): 3 y 5.
- **Tamaño del bloque (50 corridas = 5 brazos × 5 semillas × 2 problemas), estimado con tiempos de T4 del T1-plan:** corridas completas ≈ 20 h de GPU (Burgers 20 k Adam ≈ 13 min + L-BFGS; AC 40 k ≈ 14 min + L-BFGS); a presupuesto T1 ≈ 2 h. Ambos son proyecciones.

## Puntos que el investigador debe decidir antes de lanzar (no los resolví)
1. **Presupuesto:** el spec 0023 dice «corridas completas desde la inicialización»; el T1-plan y la regla de salida hablan de «un solo bloque T1» (10 % de Adam, L-BFGS 500 it.). Con corridas completas el bloque supera el tope de 5 h de la autorización anterior.
2. **Criterio:** el 0023 firma 0.85 (H-PRE-1) y la regla de salida 0.90 (T1). Y el comparador: el 0023 es B/J/E/I/F; «sin variantes nuevas» entiendo que son esos cinco brazos.
3. **Semillas:** 0023 fija 801–805 (Burgers) y 811–815 (AC); T1 usó 1–5.
4. **Interpretaciones mías del spec** (declaradas en el docstring): recálculo también antes del paso 1; J usa el mismo ε que E (para que no explote en coeficientes casi muertos); N = 2 048 puntos del pool (igual que el 0022), equiespaciados; F usa la Gram de la base con entradas uniformes (igual para todas las aristas).
