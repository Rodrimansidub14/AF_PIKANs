# DFG Re=20 L-BFGS refinement

This new Kaggle T4x2 notebook continues the existing five paired RBF/MLP DFG checkpoints without repeating training. It uses the saved scrambled-Sobol pools, deterministic full-pool L-BFGS closures, up to 540 seconds per paired seed, and direct FEATFLOW stress traction for before/after QoIs.

The notebook is private, internet-disabled, and references only the completed private source kernel. Its hard session budget is 7,200 seconds; the watchdog stops at 6,900 seconds and a soft cutoff prevents starting work too near the cap. It records per-arm step count, runtime, memory, checkpoint hash, quadrature sensitivity, FEATFLOW errors, and held-out diagnostics. `field_relative_l2` remains null because a converged FEM field has not been validated; `scientific_accepted` remains false.

This is a literature-supported optimizer refinement (Adam followed by L-BFGS) of the current DFG baselines. It cannot establish neural-network SOTA because no directly matching published PINN score for stationary DFG 2D-1 Re=20 was identified. The physical/QoI anchor is [FEATFLOW DFG 2D-1 Re=20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).

The bundle has not been run or locally validated. The notebook itself requires exactly two Tesla T4 devices and stops if they are not visible.
