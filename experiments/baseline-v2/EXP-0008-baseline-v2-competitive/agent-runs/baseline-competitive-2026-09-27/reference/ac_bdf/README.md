# Allen–Cahn BDF candidate

Requires Python, NumPy, and SciPy. From this directory, run:

```powershell
python solve_ac_bdf.py --output-root runs
```

The script solves the DeepXDE Allen–Cahn contract with centered second-order finite differences in space and SciPy's sparse-Jacobian BDF integrator in time. It compares nested AC-L0–L5 meshes and performs the tighter time-step check from the protocol. If L5 does not meet the predeclared rule, the run stays provisional; an L6 extension requires a separately versioned protocol and script.

Each invocation creates a unique timestamped run directory under `runs/`. It contains per-level compressed fields, `convergence.csv`, solver/version manifests, an `environment.lock`, and `SHA256SUMS.txt`. If the candidate passes the stated tolerances, it also writes `ac_reference.npz` and the `.01` DeepXDE-grid export; otherwise it retains the finest candidate and returns exit code 2. Solver errors return exit code 1.

`environment.lock` currently records the Python, NumPy and SciPy versions observed at execution; it is not yet a transitive dependency lock. The script has not been run or locally validated under the requested skip of local validations.

A numerically converged output remains a **candidate**. The manifest always sets `evaluation_ready` to `false`; promotion requires external review of the protocol and results.
