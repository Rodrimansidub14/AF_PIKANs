# Navier–Stokes (NS2d-CG y DFG Re=20) — resultados del baseline (EXP-0008)

Exploratorio. `scientific_accepted=false`, `reference_accepted=false`. Estado frente al baseline v1 firmado el 2026-10-03: ver `evals/baselines/baseline_v1_exploratory.md`.

## NS2d-CG (PINNacle, backward step, Re=100)
Referencia COMSOL `ns_0_obstacle.dat` (sha256 13e1d9fa…). MLP 5×100 frente a RBF-PIKAN; Adam 20 000 pasos (float32), sin L-BFGS. Semillas 701–705 (RBF16: 701–703). L2RE plano sobre las tres salidas
(la agregación de PINNacle no es verificable en su código).

| Brazo | L2RE plano, fin de Adam |
|---|---|
| mlp (41,003 par., n=5) | 9.775e-02 ± 6.73e-03 |
| rbf16 (4,032 par., n=3) | 6.667e-02 ± 2.27e-03 |
| rbf56 (40,992 par., n=5) | 6.554e-02 ± 5.77e-03 |

PINNacle publica vanilla 0.119 y LAAF 0.0824. RBF gana pareado; sin L-BFGS corrido.

## DFG 2D-1 Re=20
- **Cohortes previas (control del v1):** RBF-PIKAN 301–305 tras L-BFGS: Cd 5.124 ± 0.053 (error 8.16 %), Δp error 9.89 %; MLP 5×64 (501–505): Cd error 5.3 %. Fórmula de fuerzas σ·n corregida el 2026-09-30. Campo contra FEM Taylor–Hood L3 (referencia de trabajo aceptada con Cl absoluto).
- **Receta de la tesis de Daniel (2 000 Adam + 1 000 L-BFGS):** ningún brazo resuelve (MLP u ≈ 0.88, RBF ≈ 0.51).
- **Receta larga (informativa, no converge; L-BFGS terminó por tope de tiempo):**

| Brazo / etapa | u (L2 rel. vs FEM L3) | error de Cd |
|---|---|---|
| mlp Adam | 6.709e-01 ± 9.24e-02 | 9.820e-01 ± 1.81e-02 |
| mlp + L-BFGS | 4.214e-01 ± 8.68e-02 | 6.947e-01 ± 1.58e-01 |
| rbf Adam | 4.609e-01 ± 1.51e-02 | 7.834e-01 ± 1.96e-02 |
| rbf + L-BFGS | 1.911e-01 ± 3.87e-02 | 3.469e-01 ± 6.59e-02 |

**Artefactos.** `EXP-0008-baseline-v2-competitive/results/{ns2d-cg-20260930,dfg-daniel-cfg-20260930,dfg-long-20261001}/`, figuras `ns2dcg-flat-l2re.png` y `dfg-long-by-stage.png`.
