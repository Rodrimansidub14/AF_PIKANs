# Allen-Cahn CPU reference study (candidate artifact)

Candidate only: not canonical, not scientifically accepted, and does not unlock evals.

## Contract and solver

- PDE: u_t - 0.001*u_xx - 5*(u-u^3) = 0
- Spatial nodes: [513, 1025, 2049]
- Base BDF tolerances: {'rtol': 1e-08, 'atol': 1e-10}
- Tight BDF tolerances: {'rtol': 1e-10, 'atol': 1e-12}
- Common grid: {'nx': 512, 'nt': 256, 'include_boundary': True}

## Sanity checks

- Operator sign checks: True
- All solves succeeded / values finite: True / True
- Max boundary / initial errors: 0.000e+00 / 1.694e-21
- Energy monotone within tolerance: True

## Refinement differences

| Comparison | Weighted relative field change | Maximum relative energy change |
|---|---:|---:|
| space_513_rtol_1e-08_atol_1e-10 -> space_1025_rtol_1e-08_atol_1e-10 | 1.170267e-04 | 1.301025e-03 |
| space_1025_rtol_1e-08_atol_1e-10 -> space_2049_rtol_1e-08_atol_1e-10 | 6.273861e-05 | 3.256636e-04 |
| space_2049_rtol_1e-08_atol_1e-10 -> tighter BDF | 1.264968e-08 | 2.345827e-08 |

- Last spatial field limit (<0.005): True
- Last spatial energy limit (<0.001): True
- Subsequent-change criterion: unresolved: problem.yaml does not define effect or comparison

The configured phrase 'subsequent change below 10% of effect' has no
operational definition of effect in problem.yaml. The observed ratio is
recorded, but this study does not claim that criterion passed.

## Provenance

- Python 3.12.10, NumPy 2.3.2, SciPy 1.18.0
- Generated UTC: 2026-09-23T22:39:20.378881+00:00
- Script SHA-256: 66ec42c9783e38a7f8803e90038dde42bc1789c861c1d793390c5b855f7ad820
- NPZ SHA-256: 3ed3bb0a275b7c57113a56662a6b7b035648f33b1fa44c418157407b40a20eb0
