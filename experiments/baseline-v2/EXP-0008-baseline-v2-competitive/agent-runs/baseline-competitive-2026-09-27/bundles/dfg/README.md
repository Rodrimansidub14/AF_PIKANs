# DFG Re=20 competitive campaign draft

`run_campaign.py` is a self-contained, CUDA-only **DRAFT_UNVERIFIED** package intended for a Kaggle T4x2 session. For each of five paired seeds it starts the fixed RBF-PIKAN and five-hidden-layer tanh MLP in separate processes on the two T4s. Both arms use the same steady Navier–Stokes residual, geometry, boundary/outlet conditions, Sobol pools, loss, float32 scaling and Adam schedule; each gets 1,200 synchronized seconds. The MLP is a strong same-contract architecture comparator, not an externally validated DFG SOTA reproduction.

FEATFLOW remains the only score anchor: `Cd=5.57953523384`, `Cl=0.010618948146`, and `delta_p=0.11752016697`. The runner scores these QoIs, a disjoint held-out residual/BC diagnostic and 512-to-1024 surface-quadrature sensitivity. It deliberately writes `field_relative_l2: null`: a converged, independently reviewed FEM field is missing. There is no CPU solver or training path. No directly matching published PINN score for stationary DFG 2D-1 Re=20 was identified.

The self-contained Kaggle entry point is [`dfg-re20-t4x2.ipynb`](dfg-re20-t4x2.ipynb), with [`kernel-metadata.json`](kernel-metadata.json) configured for GPU T4x2 and internet disabled. The notebook embeds the exact runner bytes, performs no package installation, checks both T4 devices, runs the campaign, and writes `session_summary.json`, runner logs and a downloadable ZIP under `/kaggle/working/dfg-re20-t4x2/`. Its watchdog allows 6,900 seconds from the first notebook code cell; the budget retains 300 seconds for setup/closeout within the 7,200-second session cap.

The Kaggle T4x2 session-quota cap is two runtime hours (7,200 seconds), including an external 300-second setup allowance; the runner stops at 6,900 seconds after startup. Five pairs ×1,200 seconds run concurrently on the two T4s, for 6,000 seconds wall time and 12,000 allocated active-device seconds. Scoring is capped at 600 seconds, with 300 seconds of safety reserve. Total budget is at most two Kaggle quota-hours and 12,600 active-device seconds (3.5 T4-device-hours). Kaggle's product update says one hour of T4x2 notebook runtime consumes one GPU-quota hour even though two T4s are available: [Kaggle T4x2 quota note](https://www.kaggle.com/product-feedback/361104). The second T4 is idle during scoring. Five seeds fit arithmetically only if paired workers execute concurrently and each finishes by its cap.

Intended Kaggle invocation after separate review and packaging:

```bash
python run_campaign.py --output /kaggle/working/dfg_re20_t4
```

This command is **not authorized or run**. No tests, syntax checks, preflight, local solver, training, upload, Kaggle submission, or approval file was created for this task.

## Concrete blockers

- The T4x2 subprocess path has not been tested on Kaggle. CUDA autodiff through second derivatives, memory use, elapsed-time accounting, process isolation and completion of five paired seeds are unverified.
- The adapted model/loss/QoI formulas require a read-only parity review against `src/research_pipeline/sciml/dfg.py` and FEATFLOW's official stress sign convention.
- 512-point surface integration is only checked against a 1024-point sensitivity diagnostic; convergence, especially for the cancellation-sensitive lift, is unverified.
- No converged FEM field exists; do not report field error or field-based selection. FEATFLOW QoIs do not substitute for that field.
- There is no exact published PINN score for this case, so comparisons are internal architecture evidence against RBF plus FEATFLOW QoIs, not a published SOTA claim.
- This package is not wired to remote bundle preparation, hash-bound approval, or provenance review. Do not upload or launch it from this draft alone.

Physical conditions and QoIs follow [`reference-protocol-ac-dfg.md`](../../reference-protocol-ac-dfg.md) and [FEATFLOW/TU Dortmund](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html). Literature context is in [`literature-ac-dfg.md`](../../literature-ac-dfg.md).
