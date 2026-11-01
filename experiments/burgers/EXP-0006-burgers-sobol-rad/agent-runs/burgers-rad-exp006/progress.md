# Execution ledger — approved roadmap after EXP-0005

Authorization: user "Adelante ejecuta", followed by "continua, se pauso por un reset".
Subagents: user requires GPT-6 Luna, maximum reasoning effort.

1. Audit EXP-0005 saved validation/diagnostic signals; no training reference leakage.
2. Implement/test RAD on proposer surface, integrate with PINA without changing PDE/model/optimizer.
3. Prepare five paired seeds, cost estimate, preregistration and reproducible notebook.
4. Review, smoke and launch only if all scientific/execution prerequisites are actually met.
5. Prepare Allen–Cahn/DFG references and external comparator feasibility separately.

Ruling: preserve existing codex/prime-harness checkout and uncommitted work; no clean/reset or new checkout that loses untracked experiment evidence.
Ruling: evals, existing scientific thresholds and registry/claims remain read-only. Execution authorization does not authorize accepting a scientific claim.
Preliminary Burgers residual localization: Spearman 0.119/0.298/0.231; top-decile error enrichment 2.58/4.82/2.81 for seeds 101/102/103. Descriptive only; n=3 seeds.
Task 1 and task 2 in progress. Test-first coverage created for RAD and audit ranking.

Task 1 complete: signal_audit.py produced 18-source validation audit, CSVs and standalone HTML. Interior-only Burgers RBF rho0.1179–0.2959; error enrichment2.595–4.842; hotspot Jaccard0.179–0.260. Descriptive n=3 seeds.
Task 2 complete: RAD sampler and PINA callback; RED missing modules, GREEN CPU tests verify live condition updates, untouched boundaries, continued step budget and actual adaptive batch consumption.
Task 3 prepared: five paired seeds201–205, ten jobs, two GPU slots; no submit or human approval artifact. Budget and proposed baseline hashes written; final bundle regeneration pending review fix.
Verification:30 targeted SciML tests and34 remote/reporting tests passed (one overlapping test). Both CPU smoke arms produced checkpoints, independent validation, local log figures and W&B offline records.
Authentication: Kaggle read-only status succeeded on completed EXP-0005; only key-presence booleans displayed. No new remote training.
Ruling: no frozen baseline exists. General execution authorization is not a human scientific signature for newly proposed numeric gates. Prepare exact snapshot/protocol for review; no eval edits or fabricated signature.
Final review: GPT-6 Luna(max) found important missing enforcement of unsigned gate. Fixed with assert_launch_ready before notebook dispatch and before adapter imports/training/output creation. RED failing tests, GREEN5/5 targeted gate/notebook tests. No repeat reviewer dispatch.
Final: minor deferred — operational1200s training limit is checked between operations and may overshoot during a refresh; actual time is measured, external2100s timeout stays authoritative. Parameter documentation clarifies this.
Final: minor deferred — legacy W&B artifact type baseline-evidence remains; arm/seed names are distinct, protocol/scientific_accepted metadata disambiguates experiment evidence.
Final: reviewer scope rulings — unrelated preexisting edits left alone; proposed scientific thresholds remain human-owned; live T4 feasibility not claimed without execution. Cost if ignored: false acceptance or incorrect timing expectation.

Final package: bundle-v2, SHA256 ID33dfe79bc128759c6670e308fc6eb8b740bc52a92be3cdbcb618ffeccc883f34. Notebook schema valid;12 code cells compile;3 preparation cells executed with10 paired sampling rows. Unsigned launch raises before seed-job directories exist. Superseded bundle-v1 notebook renamed experiment.superseded.json; transport verification rejects it.
Final verification:32 SciML tests passed. Remote/reporting suite38 tests had one preexisting2-worker timing test timeout under local load; isolated retry passed in2.145s. Earlier full remote/reporting run passed34/34. No production timeout threshold changed to hide the failure. Logs preserved.
CPU pair smoke:8 steps each, identical initial pools/predictions, independent validation, explicit log figures and two actual .wandb offline records verified. These are technical smoke artifacts, not scientific results. Corrected postprocessing counter to find W&B's nested directory; training not repeated for a counting issue.
Protected verifier Python sources match EXP-0005 bundle hashes. No claims accepted, no baseline frozen, no new remote campaign submitted. User-facing signature package: approval-package.md with exact proposed values; execution authorization already retained.

Allen–Cahn CPU preparation complete: candidate FD/BDF fields on513/1025/2049 nodes and tighter time tolerance saved under allen-cahn-reference/study/. Last relative spatial field change6.27386e-5, energy change3.25664e-4, time-tolerance field change1.26497e-8. IC/BC value checks and energy monotonicity pass. Full reference claim remains false: effect-relative convergence criterion undefined; first-order parabolic corner compatibility at t=0 is an explicit regularity caveat. No eval unlock.
Allen–Cahn Plotly candidate diagnostics generated at allen-cahn-reference/study/allen_cahn_candidate_diagnostics.html from saved arrays, without rerunning solvers. DFG remains at reference/source reconciliation and vector-trainer preparation. RGA-KAN feasibility is documented, not executed.
