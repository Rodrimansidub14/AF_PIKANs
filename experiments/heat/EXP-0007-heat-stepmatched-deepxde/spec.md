# EXP-0007 — Heat step-matched comparison (v5)

Status: PENDING_HUMAN_SIGNATURE. This contract is unsigned and unaccepted.

## Question and scope

Under a fixed Heat PDE and saved per-seed collocation pools, what field error do the AF RBF and cubic B-spline arms reach after the DeepXDE tutorial optimizer budget? This is an exploratory comparison, not a superiority or SOTA claim.

## Problem

Solve u_t = 0.1 u_xx on [0,1] x [0,1], with u(x,0)=sin(pi*x), homogeneous Dirichlet boundaries, and exact solution sin(pi*x) exp(-0.1*pi^2*t). Residual is scaled by 1.1. Traces use sin(pi*x)+t*x*(1-x)*raw(x,t).

## Arms and seeds

RBF (rbf_fixed) and cubic B-spline (bspline_cubic), five paired seeds 901-905. The FNN arm is omitted before launch under the plan runtime rule because the 15-arm estimate exceeds 65 minutes. DeepXDE alpha=0.1 uses seeds 11,22,33,44,55, so comparisons to it are unpaired.

## Frozen pools and training

Reuse the exact saved v4 Torch scrambled Sobol NPZ pools for seeds 901-905; expected hashes are recorded in the run manifest and verified by the runner. Each arm uses the same pool for its seed, float64, full batch of 4,096 interior points, 1,024 initial points, and 512 boundary points per side. Adam lr 1e-3 for exactly 20,000 updates, then PyTorch L-BFGS with DeepXDE v1.12.2 defaults: maxiter=15000, maxfun=18750, history size 100, gradient tolerance 1e-8, function-change tolerance 0, and strong-Wolfe line search. Calls are chunked at 25 optimizer iterations to persist checkpoints and respect time caps. Per-run cap 680 seconds.

## Evaluation and resources

Train only on fixed pools. After all runs finish, score final checkpoints with the common CUDA evaluator on heat-confirmatory-512x256-v1. Session: private Kaggle T4x2, hard cap 4,500 seconds (1.25 quota-hours). Atomic progress and checkpoints are retained. Completion is summarized by Adam steps, L-BFGS steps/stopping reason, and confirmation score count.

## Pre-registration and decision boundary

This contract remains PENDING_HUMAN_SIGNATURE; no agent signs it. scientific_accepted=false and reference_accepted=false in every artifact. Results are evidence for the human scientific_claim gate only. DeepXDE comparisons are descriptive and unpaired; collocation, network, trace, and implementation differ.
