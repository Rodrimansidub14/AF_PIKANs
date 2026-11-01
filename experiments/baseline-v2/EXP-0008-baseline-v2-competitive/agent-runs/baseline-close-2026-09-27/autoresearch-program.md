# AF_PIKANs autoresearch contract

This adapts the narrow experiment loop in karpathy/autoresearch to a four-PDE baseline program.

- Immutable per-problem evaluator, held-out points and metric definitions.
- One experiment changes exactly one training/configuration hypothesis. PDE, final test, normalization convention and evaluator never change during a campaign.
- One fixed GPU wall-clock budget per candidate, pinned before submission. Five minutes is reserved for future candidate screening; it does not replace the longer seed-confirmation budget.
- Two T4s may execute independent seeds/candidates concurrently. Every run records code/config/bundle hashes, exact Sobol pools, model and parameter count, optimizer, precision, GPU slot, actual GPU telemetry and W&B IDs.
- Score candidates only on the fixed validation score for that PDE. Keep the candidate only if its score improves the current best without breaking physical/finite-value guards. Otherwise discard it. Record crashes and timeouts as outcomes; do not hide or retry them without cause.
- Do not aggregate raw field error across PDEs. Heat, Burgers, Allen–Cahn and DFG each retain their own scorecard and physical checks.
- The fixed-RBF/Sobol control is the point of departure. RAD is not automatically kept: EXP-0006 worsened Burgers error in all five paired seeds. Sampling-policy changes must be their own iterations and compete against Sobol under equal points and time.
- A keep is an engineering selection on validation, not scientific acceptance or an SOTA claim. A selected candidate returns to a fresh multi-seed confirmation campaign before any baseline is frozen.

Current blockers before the loop can make SOTA claims: no DFG RBF run yet; the Allen–Cahn published field is provisional due missing solver provenance; and the four-problem suite plus its claim gates are not frozen in the protected verifier.
