# Allen–Cahn competitive campaign draft

`run_campaign.py` is a self-contained, CUDA-only **DRAFT_UNVERIFIED** runner for a Kaggle T4x2 session. It compares the incumbent fixed-RBF PIKAN with the four-hidden-layer tanh MLP from the DeepXDE Allen–Cahn example. For each of five paired seeds, the runner starts both arms in separate processes on the two visible T4s. Both share the same PDE, exact hard transform, seed-specific scrambled-Sobol pools, float64 precision and Adam schedule; each arm is capped at 600 synchronized seconds. The MLP is an Adam-only matched-time adaptation, not a reproduction of DeepXDE's 40,000 Adam steps plus L-BFGS.

The runner first attempts to generate an independent float64 finite-difference reference on `cuda:0` using explicit RK4. It checks the spatial-refinement and halved-step temporal gates, but reference generation is capped at 2,700 seconds. Failure or timeout blocks all field scoring. Passing gates yields candidate scores only; independent comparison/provenance review is still required before acceptance. This is a new numerical reference candidate, not the archived BDF reference and not an `evals/` accepted result. The T4's FP64 throughput and the actual convergence time are unknown.

The self-contained Kaggle entry point is [`allen-cahn-t4x2.ipynb`](allen-cahn-t4x2.ipynb), with [`kernel-metadata.json`](kernel-metadata.json) configured for GPU T4x2 and internet disabled. The notebook embeds the exact runner bytes, performs no package installation, checks both T4 devices, runs the campaign, and writes `session_summary.json`, runner logs and a downloadable ZIP under `/kaggle/working/allen-cahn-t4x2/`. Its watchdog allows 6,900 seconds from the first notebook code cell; the budget retains 300 seconds for setup/closeout within the 7,200-second session cap.

The Kaggle T4x2 session-quota cap is two runtime hours (7,200 seconds), including an external 300-second setup allowance; the runner stops at 6,900 seconds after startup. The budget reserves up to 2,700 seconds for the reference, 3,000 seconds for five 600-second seed pairs running concurrently, 600 seconds for scoring, and 600 seconds of safety reserve. This is 9,300 allocated active-device seconds (2.5833 T4-device-hours) within at most two Kaggle quota-hours because the two seed arms run in parallel. Kaggle's product update says one hour of T4x2 notebook runtime consumes one GPU-quota hour even though two T4s are available: [Kaggle T4x2 quota note](https://www.kaggle.com/product-feedback/361104). The second T4 is idle during reference and scoring. Five seeds fit arithmetically only if the reference passes within its cap and paired execution works as intended.

Intended Kaggle invocation after separate review and packaging:

```bash
python run_campaign.py --output /kaggle/working/allen_cahn_t4
```

This command is **not authorized or run**. No tests, syntax checks, preflight, local solver, reference solve, training, upload, Kaggle submission, or approval file was created for this task.

## Concrete blockers

- The runner and T4x2 multiprocessing path have not been tested on Kaggle; CUDA float64 RK4 stability, runtime, convergence, process isolation, checkpoint timing, and memory behavior are unverified.
- The reference must be compared against the archived BDF AC-L4 field on a common grid, then reviewed for provenance before any field score is accepted.
- The archived BDF candidate remains `evaluation_ready=false`; the `evals/` Allen–Cahn grader remains locked pending review.
- The matched-time MLP omits DeepXDE's L-BFGS stage. Claims must identify this as an Adam-only adaptation.
- This standalone draft is not integrated with the repository's remote bundle preparation or hash-bound approval workflow.

The physical contract and numerical gates come from [`reference-protocol-ac-dfg.md`](../../reference-protocol-ac-dfg.md). The comparator is the [DeepXDE forward Allen–Cahn example](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html); the recent [LearnQuad paper](https://papers.neurips.cc/paper_files/paper/2025/file/382b95a6580cfb0b5ca33c74b4e0e770-Paper-Conference.pdf) is a separately budgeted follow-up.
