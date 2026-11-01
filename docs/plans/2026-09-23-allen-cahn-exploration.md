# Allen–Cahn Exploratory Runner Implementation Plan

> **For agentic workers:** Use the approved native execution method in this task and implement each task with test-first steps.

**Goal:** Extend the existing RBF-PIKAN pipeline to run an exploratory, fixed-Sobol Allen–Cahn pilot that reports physical and training signals without field-error claims.

**Architecture:** Keep Burgers RAD as its own paired experiment and preserve its existing fail-closed scientific and exact-bundle gates. Add Allen–Cahn support to model, sampling, and training; evaluate a saved checkpoint on a second Sobol set with recomputed derivatives, hard-trace checks, and a descriptive energy curve. Keep the field-reference lock and `evals/` unchanged. Package the three-seed exploratory pilot only after local smoke tests; do not submit until the exact generated bundle receives its required human approval.

**Tech Stack:** PyTorch, PINA/Lightning, SciPy scrambled Sobol, NumPy, Plotly, W&B, Kaggle notebook bundle tooling.

**Spec:** `experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/spec.md` and `parameter-decisions.md`.

## Global Constraints

- `evals/` and `registry/claims.jsonl` are read-only.
- Allen–Cahn field error remains locked while its BDF reference is `pending_convergence`.
- The pilot uses only fixed scrambled Sobol points; RAD/RAR-D is a later, separate comparison.
- The current proposal is 8,192 training points, 16,384 held-out points, 3 seeds, and 20,000 steps or 1,200 seconds per seed.
- Every run is exploratory, threshold-free, non-accepted, and must retain exact seeds, pool matrices, hashes, configuration, hardware, logs, and plots.
- Kaggle dispatch requires a human approval JSON bound to the final bundle ID; this plan does not create that approval.

## Review Focus

- Hard constraints are exact at `t=0` and `x=±1`; test both boundaries over multiple times.
- The Allen–Cahn reaction sign and diffusion coefficient match the declared PDE; verify using a direct residual test.
- Training, diagnostic, and held-out Sobol streams are reproducible and independent; verify bounds and unequal arrays.
- The held-out evaluator recomputes derivatives and emits no field-error metric; test output keys and lock behavior.
- Energy uses the Allen–Cahn functional and a symmetric spatial grid; verify shape, finiteness, and boundary inclusion.

---

### Task 1: Allen–Cahn model, sampling, and trainer support

**Files:**
- Modify: `src/sciml/model.py`
- Modify: `src/sciml/sampling.py`
- Modify: `src/sciml/train.py`
- Test: `tests/test_allen_cahn_model.py`
- Test: `tests/test_allen_cahn_sampling.py`
- Test: `tests/test_allen_cahn_train.py`

**Interfaces:**
- `build_model("allen_cahn", config)` returns the existing fixed RBF-PIKAN with the exact hard transform `x²cos(πx)+t(1−x²)N(x,t)`.
- `pools("allen_cahn", seed, n=8192, boundary=512, heldout=16384)` returns reproducible `interior`, `diagnostic`, `heldout`, `boundary_probe`, and `initial_probe` arrays; only `interior` enters PINA's physics condition.
- `physics("allen_cahn", problem_config)` computes the normalized residual `(u_t−D u_xx−5(u−u³))/6.001`; the problem has no soft IC/BC conditions.

- [ ] **Step 1: Write failing tests** for hard traces, PDE residual sign/scale, distinct Sobol streams, no soft trace conditions, and an eight-step CPU trainer smoke.
- [ ] **Step 2: Run those tests and confirm expected failures** against the current Heat/Burgers-only implementation.
- [ ] **Step 3: Add the minimal model, point-pool, and trainer branches** without changing Heat/Burgers behavior or permitting RAD on Allen–Cahn.
- [ ] **Step 4: Run the new tests and the existing RAD/PINA tests** to verify both the new path and Burgers non-regression.

### Task 2: Independent exploratory evaluation and artifacts

**Files:**
- Create: `src/sciml/allen_cahn_evaluation.py`
- Modify: `src/sciml/campaign.py`
- Test: `tests/test_allen_cahn_evaluation.py`

**Interfaces:**
- `evaluate_allen_cahn(model, problem_config, out_dir, pools)` recomputes `u_x`, `u_t`, and `u_xx` on the held-out Sobol matrix, calls the existing exploratory residual grader, checks hard traces, and writes metrics plus arrays.
- The evaluator reports residual mean/RMSE/p95/max, exact trace errors, energy at declared times, and runtime/coverage signals. It never reads or returns `field_relative_l2`, a reference field, or interface masks.
- A Plotly artifact contains the predicted field and residual signal; energy is a separate time-series trace. W&B receives quantitative scalar summaries and the artifact.

- [ ] **Step 1: Write failing tests** for fresh derivative computation, locked field metrics, correct energy integration, and required saved artifact keys.
- [ ] **Step 2: Verify those tests fail** because no Allen–Cahn evaluator exists.
- [ ] **Step 3: Implement the evaluator and campaign dispatch branch** while leaving the Heat/Burgers verifier path intact.
- [ ] **Step 4: Run evaluator tests and inspect the generated Plotly artifact** from a deterministic synthetic checkpoint.

### Task 3: Reproducible campaign manifest and guarded bundle

**Files:**
- Create: `experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/manifest.json`
- Modify: `experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/spec.md`
- Modify: `experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/parameter-decisions.md`
- Create or update: `notebooks/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/allen-cahn-rbf-sobol-exploratory.ipynb`
- Test: `tests/test_allen_cahn_campaign.py`

**Interfaces:**
- Manifest fixes seeds 101–103, fixed-Sobol sampling, 8,192/16,384 train/held-out counts, 1,024 batch, Adam 1e-3, and a 20,000-step/1,200-second cap; it labels the run exploratory and leaves all scientific acceptance fields false.
- Notebook uses one T4 per independent seed job, at most two jobs concurrently, W&B entity `man22611-uvg`, and the existing remote bundler.
- The notebook checks manifest/suite consistency before training; Kaggle dispatch remains blocked until an `approval.json` is created by the human for the exact bundle ID.

- [ ] **Step 1: Write failing manifest and notebook contract tests** for declared seeds/budgets, fixed sampling, absence of RAD and field scoring, and exact-bundle preflight.
- [ ] **Step 2: Run the tests and confirm the missing campaign contract** is the failure.
- [ ] **Step 3: Add the manifest, notebook, and parameter rationale** from EXP-0003; record that no numerical field reference or SOTA claim is available.
- [ ] **Step 4: Render, compile, and locally smoke-test the notebook**; generate a candidate bundle without submitting it.

