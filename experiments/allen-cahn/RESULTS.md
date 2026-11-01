# Allen–Cahn 1D — resultados del baseline (EXP-0008)

Exploratorio. `scientific_accepted=false`, `reference_accepted=false`. Estado frente al baseline v1 firmado el 2026-10-03: ver `evals/baselines/baseline_v1_exploratory.md`.

**Contrato (DeepXDE).** u_t = 0.001 u_xx + 5(u − u³) en [−1,1]×[0,1], IC x²cos(πx), transformada dura; referencia `Allen_Cahn.mat` (DeepXDE, tercero; **aceptada por el investigador el 2026-10-01**), malla 201×101.
MLP 3×20 (921 parámetros) frente a RBF-PIKAN ancho 16; 8 000 puntos Sobol, Adam 40 000 pasos (float32) + L-BFGS (tope 300 s). Semillas 601–605.

| Brazo | Fin de Adam | + L-BFGS |
|---|---|---|
| MLP 3×20 | 1.947e-01 ± 9.86e-02 | 1.773e-02 ± 8.75e-03 |
| RBF-PIKAN 16 | 2.838e-02 ± 7.42e-03 | **2.686e-03 ± 9.55e-04** |

**Lectura.** El RBF gana en las 5 semillas, en ambas etapas. DeepXDE oficial (no pareado): 2.00e-2 ± 6.1e-3. Las referencias generadas por agente (FD8192/ETDRK4) siguen rechazadas y no cuentan.

**Artefactos.** `EXP-0008-baseline-v2-competitive/results/ac-rerun-20260930/`, figura `EXP-0008-baseline-v2-competitive/figures/ac-l2-by-stage.png`.
