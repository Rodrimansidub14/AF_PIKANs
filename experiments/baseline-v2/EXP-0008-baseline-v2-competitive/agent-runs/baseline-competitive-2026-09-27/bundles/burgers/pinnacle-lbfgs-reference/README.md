# PINNacle Burgers 1D — Vanilla + Adam_LBFGS

Esta variante reproduce el modo publicado --method lbfgs del commit PINNacle fijado 6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf: Adam con lr=1e-3 hasta el paso 4,999; L-BFGS comienza en el paso 5,000. Usa PyTorch LBFGS con lr=1, max_iter=20 y total de 20,000 pasos, el modelo 5x100 tanh, las mismas pérdidas, puntos, cinco seeds y puntuación Cole–Hopf/FEM que el bundle Vanilla.

El benchmark.py del commit pasa explícitamente switch_epoch=5000, aunque Adam_LBFGS.py declara 10000 como valor predeterminado. Se reproduce el override del benchmark, no el default.

Kernel Kaggle: rodrimandub/pinnacle-reference-burgers-lbfgs; GPU NvidiaTeslaT4 (dos T4). Hard cap 4 h, watchdog reserva 5 min. No se ejecutan entrenamientos, pruebas ni validaciones locales.
Fuentes primarias: benchmark.py y src/optimizer/adam_lbfgs.py del repositorio PINNacle fijado; paper NeurIPS 2024 indicado en el notebook.
