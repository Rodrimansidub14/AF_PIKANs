# Next-problem readiness after EXP-0005

## Decision

Allen-Cahn is the next lower-cost problem for a physics-signal pilot, after the scalar trainer and trusted checkpoint evaluator are extended. Its BDF field reference can be built independently on CPU; field-error reporting must stay locked until the convergence contract is met and a human approves a verifier change.

DFG 2D-1 should follow as a separate engineering track. Its current operators are verified, but the trainer has no vector/2D path, the FEM reference and QoIs are absent, and cited reference values need reconciliation. Neither problem is ready for a scored field comparison today.

The user's authorization to continue the roadmap covers preparation and implementation work. It does not itself create a signed preregistration, a frozen baseline, an approved machine-readable campaign budget, an eval-integrity change, or scientific acceptance. The exploratory suite still declares `budget.approved: false`; `evals/baselines/README.md` says there is no frozen baseline.

## What EXP-0005 makes reusable

The architecture screen completed 18 runs and synced 18/18 to W&B. Its summaries explicitly mark scientific acceptance false. Reuse the measured training and instrumentation patterns only after the parent signal audit chooses what transfers; do not treat EXP-0005 validation scores as thresholds or transfer its PDE-specific pools, normalization, hard transforms, or loss balance.

The roadmap already records the two-T4 worker assignment and says to verify device assignment again per campaign. Current trainer code uses float64 and one device per worker; the EXP-0003/0004 precision, batch, time, and seed proposals remain drafts to revisit after the signal audit.

## Allen-Cahn 1D

**Reference status and exact blocker.** `evals/problems/allen_cahn_1d/reference.py` contains the PDE operator, hard transform, grid helper, and `REFERENCE_STATUS='pending_convergence'`; it supplies no field solution. The problem contract requires centered second-order space differences with implicit BDF time stepping, at least three resolutions, independent spatial and temporal-tolerance refinements, field change below 0.005, QoI change below 0.001, and a subsequent change below 10% of the effect. The verifier grid is fixed at 512 x 256, with diagnostic times 0.25, 0.5, 0.75, and 1.0. The interface mask `abs(u_ref)<0.8` is explicitly locked and unavailable to the controller until a reference exists.

Two convergence terms still need an operational definition before freezing results: the norm/quadrature for "field change" and the scalar QoI(s) behind "QoI change" and "fraction of effect." EXP-0003 proposes energy diagnostics, but neither the YAML nor the reference module defines the energy formula and convergence samples. DeepXDE 1.12.2 documents the matching PDE, traces, and hard transform, and loads an Allen_Cahn.mat dataset on dx=dt=0.01; that example does not satisfy this repository's independent BDF convergence requirement by itself ([official example](https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html)).

**Offline next actions.**

1. Build a CPU-only method-of-lines reference generator from the declared contract: second-order centered finite differences in x, exact Dirichlet values, and SciPy's implicit BDF integrator for `u_t=0.001u_xx+5(u-u^3)`. Save the generated fields at the fixed verifier grid, plus solver settings, versions, resolution/tolerance levels, interpolation method, and hashes under a generated reference-artifact directory in `var/agent_runs/`.
2. Choose and record nested spatial levels and separate BDF tolerance refinements. Compare each result after projection to the same 512 x 256 verifier grid; calculate the declared field-change norm and each declared QoI at fixed times. Check the configured thresholds before marking the reference ready. Do not expose the pending interface mask to training or tune from the locked final error.
3. Define energy before the sweep if it is the QoI: for this PDE the standard gradient-flow energy is `integral(0.001/2 * u_x^2 + 5/4 * (1-u^2)^2) dx`; record the quadrature and time samples. This is a proposed diagnostic definition for human review, not an accepted verifier threshold.
4. Extend the proposer path explicitly for Allen-Cahn: model hard transform, correct reaction-diffusion residual, Sobol bounds/pools, fixed-pool control, held-out diagnostic coordinates, and instrumentation. Today `model.py` and `sampling.py` reject IDs other than Heat/Burgers, `train.py` routes every non-Heat PDE through Burgers' residual, and its model transform only special-cases Heat.
5. Add a trusted CPU checkpoint evaluator that recomputes predictions and derivatives on verifier-owned coordinates. `evals/checkpoint_validation.py` currently supports Heat/Burgers only. Candidate-supplied derivatives/residual arrays cannot be trusted as grader input; the exploratory grader documents this precondition.

A residual-only signal run can precede reference convergence once trainer/evaluator support, preregistration, and campaign gates are complete. Relative L2 remains `null` with reason `reference_not_verified` until the reference is audited and the verifier is changed through the human `eval_integrity` gate.

## DFG 2D-1, Re=20

**Reference status and exact blocker.** `evals/problems/dfg_2d/reference.py` supplies the PDE residual, inlet/outlet, deterministic grid, boundary points, and critical masks; its module docstring says no FEM field is supplied, and status is `pending_fem_convergence`. The YAML requires curved-geometry Taylor-Hood P2/P1, at least three resolutions, independent mesh/quadrature/algebraic-tolerance refinements, field change below 0.005, QoI change below 0.001, and subsequent change below 10% of the effect. It requests drag, lift, and pressure difference but does not implement them.

There is a source-target conflict to resolve before freezing QoIs. The official FEATFLOW 2D-1 page matches this contract's geometry, inlet, do-nothing outlet, and asymmetric-gradient stress `sigma = nu grad(u) - p I`. It defines drag/lift from the cylinder stress integral and pressure difference as `p(0.15,0.2)-p(0.25,0.2)`, and reports Cd=5.57953523384, Cl=0.010618948146, and dp=0.11752016697 ([official benchmark](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)). The EXP-0004 source list also cites Turek et al. 2010, whose Re=20 table reports different values: Cd=6.1853331, Cl=0.009401228, dp=0.17101905 ([paper PDF](https://www.mathematik.tu-dortmund.de/papers/TurekMierkaHysingKuzmin2010.pdf)). Trace which formulation, geometry, force convention, and reference table apply before choosing values; do not treat these numbers as interchangeable.

**Offline next actions.**

1. Select and version a CPU FEM stack and exact curved-cylinder mesh procedure; current `requirements-analysis.txt` has SciPy/PyVista but declares no FEM solver. Implement the stated P2/P1 reference in a standalone, reproducible builder.
2. Run separate mesh, quadrature, and algebraic/nonlinear-tolerance refinements. Interpolate each field to the fixed 440 x 82 interior verifier grid with a declared quadrature/interpolation rule. Compute field change and each QoI change on the same levels; refine further if the YAML criteria do not pass.
3. Implement QoI extraction with the selected convention: stress integration on the exact circle, normalization using mean inlet speed 0.2 and cylinder diameter 0.1, and pressure samples at the declared front/rear points. Keep the no-extra-pressure-condition contract unless a human-approved contract change says otherwise.
4. Before any training, extend the trainer to (x,y) inputs, (u,v,p) outputs, all momentum/continuity derivatives, the five separate boundary components, and independent trusted evaluation. Existing `model.py`, `sampling.py`, and `train.py` are scalar (x,t) Heat/Burgers code. Smoke memory, precision, coordinate order, corner weighting, and outlet losses on CPU first.

Even after FEM convergence, DFG is not wired into the current thesis suite: its problem YAML declares tier 3, while `evals/suites/tier3_thesis.yaml` lists `ns_cavity_2d`. The residual-signal suite can host a non-evidence pilot only after the trusted evaluator/runner and preregistration exist.

## Shared gates and sequence

- `evals/suites/exploratory_rbf_signals.yaml` is explicitly `verifier_primitives_ready_runner_pending`, emits no scientific evidence, has no thresholds, and lists the trusted evaluator, trainer/instrumentation, preregistration/budget, and remote-auth validation as blockers. Its field-error locks include Allen-Cahn and DFG.
- `evals/graders/exploratory.py` returns `reference_not_verified` for either problem even if arrays are supplied. `evals/` is read-only under `AGENTS.md`; keep the lock until a human-reviewed reference audit and `eval_integrity` change.
- EXP-0003 and EXP-0004 folders contain draft `spec.md` and `parameter-decisions.md`, but no campaign manifest or human signature. The current tier-2 suite also remains declared-not-implemented and blocked on a baseline; DFG is absent from it.
- Recommended order: finish the EXP-0006 RAD proposer work and its authorized review; implement Allen-Cahn residual-only support and its CPU BDF reference in parallel; only then start DFG's FEM/reference and vector-trainer work. Keep both references independent of candidate training and final evaluation.

This audit only read repository artifacts and public reference pages. It ran no training, GPU job, test suite, authentication flow, or scientific acceptance step.
