# Baseline delivery — 2026-09-27

Status: operational results delivered. Scientific acceptance and baseline freezing remain false.

The user's current instruction is to skip local validations, execute notebooks and deliver the baselines. No local test suite, preflight, integrity validator or reference-convergence audit was run in this delivery. Four result notebooks and a consolidated notebook were executed against the completed campaign outputs, without duplicate GPU training.

## Delivered

- Heat: existing EXP-0005 RBF, MLP and B-spline checkpoints (three seeds per architecture) and RBF extension 104–105 measured against the analytic solution on the protected 512×256 grid. All RBF Heat jobs used 5,000 steps and 4,096 interior points. The old 20k-versus-5k explanation was incorrect. Cohorts remain separate because recorded source versions differ and no equivalence audit was requested. B-spline reached the 300-second limit after 4,276 / 4,395 / 4,714 steps; these actual costs are preserved.
- Burgers: existing five Sobol/RAD pairs consolidated; no retraining. RAD remains worse in every pair for this configuration.
- Allen–Cahn: all five existing checkpoints measured on the published DeepXDE 201×101 reference grid. Field L2, maximum field error, interface position and width, existing energy and held-out residuals are delivered. The historical interpolated-reference metric is retained separately. Field scores remain provisional.
- DFG: five seeds 301–305 recovered across r3 and r4. Kaggle r4 is ERROR, not QUEUED, because PyVista imports IPython.core.guarded_eval during post-training visualization. The three r4 workers had already completed. Their previously archived evaluation records execution_valid=true. r3 retains its post-training harness-failure provenance. No resubmission was necessary.

## Review corrections and limits

The r4 budget metadata (two seeds / 3,000 GPU-s) conflicts with its training continuation (three seeds / 3,600 GPU-s). Original files are unchanged and the defect is documented. DFG used 6,000 nominal GPU-s total, with 6,000.158 seconds actually recorded across five jobs. No independent FEM field is available. Allen–Cahn reference convergence and solver provenance remain unresolved. These are exploratory baselines, not SOTA claims or scientific acceptance.

## Deliverables

- [Report](delivery/baselines-report.md)
- [Executed consolidated notebook](delivery/baselines-results.ipynb)
- [Execution record](delivery/notebook-execution.json)
- [CSV](delivery/baselines.csv)
- [Interactive report](delivery/index.html)
- [ZIP](baselines-delivery-2026-09-27.zip)

The earlier plan and report are preserved as `*.pre-delivery-20260927.md`. Future reference-convergence or mechanism work is a separate campaign; no such training was silently started.
