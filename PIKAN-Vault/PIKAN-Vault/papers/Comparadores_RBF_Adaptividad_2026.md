---
title: "Frontier comparators for fixed RBF-PIKAN and adaptive sampling"
authors: Juan Diego Toscano et al. · Weihang Ouyang et al. · Shao-Ting Chiu et al.
year: 2026
venue: "npj Artificial Intelligence; Advanced Intelligent Discovery; arXiv preprint"
layer: METHODS
status: source-reviewed
source_role: literature_scout
tags: #pinn #pikan #rbf #adaptive-sampling #benchmarks
---

# Frontier comparators for AF_PIKANs

This note records three primary sources relevant to interpreting the current fixed-basis RBF-PIKAN baseline and choosing later, controlled comparisons. Published results are contextual references unless the equation, domain, boundary/initial conditions, enforcement, metric, and training budget are made comparable.

## vRBA: variational residual-based adaptivity

Toscano, Chen, Oommen, Darbon, and Karniadakis introduce a variational framework that connects convex transformations of PDE residuals to objective functionals and adaptive sampling distributions. The paper studies PINN and operator-learning cases, including Burgers and Allen–Cahn, and reports comparisons among several potentials and training configurations. It is a peer-reviewed 2026 article in *npj Artificial Intelligence*.

**Contract fit.** Their standard Burgers case uses the canonical 1D setting with ν = 0.01/π and initial data −sin(πx) on x ∈ [−1,1], t ∈ [0,1]. The project uses the same PDE parameters and data, but vRBA uses Fourier/periodic encodings while the project’s EXP-0005 imposes zero Dirichlet and initial traces through sampled constraints. Therefore the equation and data are close, while the admissible function space, constraint enforcement, architecture, optimizer, and budget differ; do not treat the published error as a direct score comparison.

Their Allen–Cahn test is not the project contract: it uses diffusion coefficient 10⁻⁴ and periodic spatial conditions. EXP-0003 uses diffusion 10⁻³ and Dirichlet traces u(±1,t) = −1. Use vRBA as a method-design reference, not as a numeric Allen–Cahn comparator for EXP-0003.

## RAMS: adversarial-gradient moving samples

Ouyang, Zhu, Xiong, Liu, and Lu propose RAMS, which treats sample coordinates as trainable variables and moves them along an adversarial-gradient direction to increase the physics residual. The journal paper demonstrates the method with PINNs, operator-learning methods, and existing nonadaptive/adaptive samplers. For its 1D Burgers PINN benchmark it reports ten independent runs.

**Contract fit.** The RAMS 1D Burgers case matches the project’s canonical PDE, viscosity ν = 0.01/π, domain, sine initial condition, and homogeneous Dirichlet boundaries. Its numeric errors are still not directly comparable: the paper uses a 3-layer, width-100 MLP, 5,000 residual points, staged sample movement/resampling, and an Adam/L-BFGS schedule; the project’s RBF campaign uses a different architecture, pool, enforcement/training protocol, and budget. RAMS is therefore a strong candidate to reimplement on the fixed RBF model under a common budget, retaining Sobol as the control.

The paper’s other PINN examples do not establish direct Heat or Allen–Cahn comparisons for the contracts currently recorded in AF_PIKANs.

## Free-RBF-KAN: adaptive centers and kernel smoothness

Chiu et al. propose Free-RBF-KAN with learnable RBF centers and shape/smoothness parameters. The work is an **arXiv preprint**, not a peer-reviewed version-of-record. Its physics-informed section compares fixed RBF-KAN, Free-RBF-KAN, spline KAN, and MLP on a high-frequency heat-conduction problem, plus a Helmholtz problem. The authors report that adaptive centers and smoothness improve the heat result relative to their fixed RBF-KAN and spline KAN configurations.

**Contract fit.** Its heat case uses u(x,0) = sin(50πx), zero boundaries, and α = (50π)⁻², with exact solution e⁻ᵗsin(50πx). This is not the project’s smooth fundamental-mode Heat case u(x,0) = sin(πx), α = 0.1. The paper does not provide a matching Burgers or Allen–Cahn PINN benchmark. It supports considering a separate high-frequency Heat stress contract and a Free-RBF-KAN architecture experiment; its reported numbers should not be attached to current EXP-0005 scores.

The existing source note [[Chiu_2026_Free_RBF_KAN]] contains the paper’s basic architecture summary. The current AF_PIKANs model holds RBF centers and widths fixed, so it is not a Free-RBF-KAN reproduction.

## Relevance to signal interpretability

The fixed RBF model offers a stable coordinate grid for measuring per-basis activations, coefficient contributions, effective basis occupancy, sensitivity of the field/residual to basis parameters, and correspondence between basis responses and held-out error regions. These are candidate observables, not proof that RBF is inherently more interpretable than other architectures. Spline bases also expose explicit basis functions, while nonlinear hidden-layer composition means single-layer coefficients alone do not explain the final PDE field. Test interpretation claims with cross-seed stability and ablations, and keep the MLP as a contextual architecture reference if desired.

## Primary sources

- Toscano, J. D., Chen, D. T., Oommen, V., Darbon, J., & Karniadakis, G. E. (2026). [A variational framework for residual-based adaptivity in neural PDE solvers and operator learning](https://doi.org/10.1038/s44387-026-00084-4). *npj Artificial Intelligence*, 2(1), 32.
- Ouyang, W., Zhu, M., Xiong, W., Liu, S.-W., & Lu, L. (2026). [RAMS: Residual-Based Adversarial-Gradient Moving Sample Method for Scientific Machine Learning in Solving Partial Differential Equations](https://doi.org/10.1002/aidi.202500214). *Advanced Intelligent Discovery*, 2(3), e202500214.
- Chiu, S.-T., Cheung, S. W., Braga-Neto, U., Lee, C. S., & Li, R. P. (2026). [Free-RBF-KAN: Kolmogorov-Arnold Networks with Adaptive Radial Basis Functions for Efficient Function Learning](https://arxiv.org/abs/2601.07760). arXiv preprint 2601.07760, v3.

## BibTeX

The complete records are in `sources/AF_PIKANs_Research_Brief_References_2026-09-23.bib` under keys `toscano2026variationalrba`, `ouyang2026rams`, and `chiu2026freerbfkan`.
