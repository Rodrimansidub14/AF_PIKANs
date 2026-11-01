---
title: "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks"
authors: Chenxi Wu · Min Zhu · Qinyang Tan · Yadhu Kartha · Lu Lu
year: 2023
venue: "Computer Methods in Applied Mechanics and Engineering 403, article 115671"
layer: ARCH
doi: "10.1016/j.cma.2022.115671"
url: "https://doi.org/10.1016/j.cma.2022.115671"
preprint: "arXiv:2207.10289 (2022)"
status: reading
source_role: literature_scout
tags: #arch #sampling #rad #rar-d #pinn
---

# A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks

**Wu et al. · 2023 · CMAME 403, article 115671**

> Estudio comparativo de muestreo uniforme, cuasi Monte Carlo y estrategias adaptativas RAD y RAR-D para PINNs. La versión publicada apareció en 2023; arXiv conserva la versión inicial de 2022.

## Links

- Versión publicada: https://doi.org/10.1016/j.cma.2022.115671
- Preprint: https://arxiv.org/abs/2207.10289

### Papers relacionados
- [[Adaptive_Collocation]]
- [[Takamoto_2022_PDEBench]]
- [[Raissi_2019_PINNs]]

### Dominios
- [[Fluidos_y_Mecánica_Estructural]]

### Conceptos
- [[Adaptive_Collocation]]
- [[Spectral Bias]]

## Notas

- Compara secuencias uniformes, entre ellas Sobol, con estrategias de remuestreo adaptativo basadas en residual.
- El artículo informa experimentos extensos y mejoras para los problemas estudiados. Es una motivación para probar RAD frente a Sobol con presupuesto y evaluación propios, no una garantía de mejora en esta campaña.
- EXP-0006 de AF_PIKANs usa una adaptación finita y explícitamente documentada; sus parámetros y mecanismo no se deben atribuir como reproducción exacta del artículo.

## BibTeX verificado

    @article{wu2023adaptivesamplingpinns,
      title = {A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks},
      author = {Wu, Chenxi and Zhu, Min and Tan, Qinyang and Kartha, Yadhu and Lu, Lu},
      journal = {Computer Methods in Applied Mechanics and Engineering},
      volume = {403},
      pages = {115671},
      year = {2023},
      doi = {10.1016/j.cma.2022.115671},
      eprint = {2207.10289},
      archivePrefix = {arXiv},
      url = {https://doi.org/10.1016/j.cma.2022.115671}
    }
