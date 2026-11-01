# WP-E3 — DFG re-scoring with one σn scorer (CPU, float32)

`scientific_accepted=false`, `reference_accepted=false`. Reads checkpoints only.
Script: `../tools/exp0008_dfg_rescore.py`. Data: `dfg_rescore_sigma_n.json`.
Scorer: `src/research_pipeline/sciml/dfg.py::qoi_from_cylinder` (FEATFLOW σn traction, radial normal, 512-point contour).

## Result (mean ± sample SD, n=5 seeds; errors vs FEATFLOW Cd 5.5795, Cl 0.010619, Δp 0.11752)

| Cohort · arm · stage | Cd | Cd err | Cl | \|Cl err\| | Δp | Δp err |
|---|---|---:|---|---:|---|---:|
| 301–305 · RBF · pre-LBFGS | 3.967 ± 0.180 | 28.9 % | −0.026 ± 0.125 | 0.094 | 0.0777 ± 0.0055 | 33.9 % |
| 301–305 · RBF · post-LBFGS | 5.124 ± 0.053 | **8.2 %** | +0.039 ± 0.099 | 0.083 | 0.1059 ± 0.0028 | **9.9 %** |
| 501–505 · RBF · pre-LBFGS | 3.894 ± 0.120 | 30.2 % | +0.026 ± 0.078 | 0.056 | 0.0759 ± 0.0034 | 35.4 % |
| 501–505 · RBF · post-LBFGS | 5.233 ± 0.052 | **6.2 %** | +0.014 ± 0.072 | 0.049 | 0.1089 ± 0.0014 | **7.3 %** |
| 501–505 · MLP 5×64 · pre-LBFGS | 3.422 ± 1.775 | 38.7 % | +0.015 ± 0.041 | 0.033 | 0.0687 ± 0.0381 | 41.5 % |
| 501–505 · MLP 5×64 · post-LBFGS | 5.286 ± 0.284 | **5.3 %** | +0.010 ± 0.037 | 0.024 | 0.1113 ± 0.0084 | 5.3 % |

## What it establishes

1. **H5 supported.** With one scorer both cohorts agree: RBF post-L-BFGS Cd 5.12 vs 5.23 (8.2 % vs 6.2 %).
2. **The archived "Cd 1.463 (74 % below FEATFLOW)" of cohort 301–305 was a scorer artifact.**
   The tangential-reduction formula (`du_τ/dn`, superseded on 2026-09-28 because its tangent
   orientation disagreed with FEATFLOW's traction; failure F-009) reproduces it:
   the 501–505 checkpoints scored with that formula give Cd ≈ 1.45–1.57, the same range.
   The two formulas are equivalent only for an exact no-slip/divergence-free field; for a PINN they
   differ, and σn is the estimator adopted by the project. Δp is scorer-independent (identical under
   both), which is why Δp never showed the discrepancy.
3. The two cohorts are **different models**, not just different seeds: 501–505 use `run_campaign_source.py`
   (RBF width 0.35, centers in [−1,1], MLP 5×64) while 301–305 use `dfg.py` (`DFGRBFPIKAN`).
   Their post-L-BFGS QoIs are nevertheless close.
4. Gate status (proposed G1: Cd ≤ 10 %, Δp ≤ 10 %, |ΔCl| ≤ 0.03): Cd and Δp pass for RBF in both cohorts
   (301–305 Δp is borderline at 9.9 %); **Cl fails** (|ΔCl| 0.05–0.08, SD 0.07–0.10 across seeds);
   the MLP passes Cl (0.024). G2 (≤ 5 %) is not reached by the RBF (best 6.2 % / 7.3 %).

## Consequences

- The scorecard rows for `RBF-PIKAN_current-cohort-LBFGS` (Cd 1.46294, 73.78 %) and
  `RBF-PIKAN_pre_LBFGS_r3_r4` (Cd 1.075, 80.7 %) must be superseded, not edited (append-only):
  these σn values replace them.
- DFG no longer needs a from-scratch v2 run to reach G1 on Cd/Δp. Remaining question: Cl noise and G2,
  which may need a longer L-BFGS continuation (GPU) — decide after the Burgers/AC results.
- Failure F-012 (open) → cause **diagnosed**: scorer (tangential vs σn).
