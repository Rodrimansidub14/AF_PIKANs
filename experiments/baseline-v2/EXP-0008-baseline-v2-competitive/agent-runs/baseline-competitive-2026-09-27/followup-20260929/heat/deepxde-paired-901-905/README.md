# Heat 1D α=0.1 · receta DeepXDE v1.12.2 (Kaggle)

Reproducción remota del demo oficial con cinco seeds y referencia analítica. Requiere Kaggle GPU T4×2. DeepXDE 1.12.2 se instala solo si falta el pin exacto.

## Protocolo

- Heat 1D en [0,1] × [0,1]: u_t = 0.1 u_xx, Dirichlet homogénea y u(x,0)=sin(πx).
- Red [2,20,20,20,1], tanh, Glorot normal; 2540 puntos interiores, 80 frontera, 160 iniciales, muestreo Hammersley.
- Adam lr=1e-3, 20 000 iteraciones, seguido de L-BFGS-B.
- Seeds [901,902,903,904,905]; MAE, RMSE, L2 relativo y residual medio absoluto en 512×256 y 256×201.
- Referencia exacta: exp(-0.1 π² t) sin(πx).

Este brazo conserva la receta DeepXDE 1.12.2 (red, puntos y optimizador) pero fija α=0.1 para igualar el contrato Heat de AF_PIKANs. Reporta la grilla 256×201 del demo como evaluación secundaria; el demo oficial usa α=0.4, por lo que esas métricas no se mezclan.

## Presupuesto, hardware y ejecución

El watchdog corta a 950 s desde el inicio de la primera celda, incluidos instalación condicional, imports, entrenamiento, evaluación y escritura. Techo CLI: 1000 s. Deja status truncated, seed/fase, costo parcial y provenance en /kaggle/working/heat_deepxde_reference/run_summary.json, más CSV parcial seed_metrics.csv.

La metadata solicita NvidiaTeslaT4 (Kaggle T4×2) y valida dos T4 visibles. DeepXDE/PyTorch corre cada modelo en cuda:0; la segunda T4 queda sin uso por corrida. El JSON estima GPU-horas de pared y dispositivo-horas si se cuentan ambas tarjetas.

Para reservar al menos 12 h semanales, usar el techo conservador de 4 h de dispositivo por corrida de hasta 2 h: ejecutar solo con saldo previo de al menos 16 h. Este bundle no consulta ni consume cuota.

Reemplaza el username en kernel-metadata.json y ejecuta:

    kaggle kernels push -p var/agent_runs/baseline-competitive-2026-09-27/followup-20260929/heat/deepxde-paired-901-905 --accelerator NvidiaTeslaT4 --timeout 1000

Sin datasets ni kernels fuente. Internet se usa solo si hay que descargar el pin DeepXDE. Referencias: [demo oficial](https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/heat.html), [metadata Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md), [CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
