# PINNacle Burgers 1D — Vanilla PINN

Notebook Kaggle autocontenida para el caso forward Burgers1D-C del benchmark PINNacle. Implementa el baseline Vanilla del paper con PyTorch: cinco capas ocultas de 100 neuronas, `tanh`, inicialización Glorot normal, pérdidas MSE sin ponderar, 8192 puntos interiores, Adam con tasa `1e-3` y 20 000 épocas. Ejecuta cinco seeds fijas (`0`–`4`) y evalúa cada modelo en una grilla Cole–Hopf común de 1025 posiciones por 129 tiempos, con cuadratura Gauss–Hermite de orden 128.

El notebook requiere el acelerador Kaggle **GPU T4 x2**. Usa una GPU por seed y corre hasta dos seeds simultáneas. El watchdog establece un máximo de 7200 segundos para el trabajo completo y reserva 120 segundos para escribir los resultados. Si no terminan los cinco entrenamientos, los estados incompletos quedan marcados como `truncated`; no se reporta una reproducción completa.

## Condiciones y muestreo

Se resuelve `u_t + u u_x = nu u_xx` en `x ∈ [-1, 1]`, `t ∈ [0, 1]`, con `nu = 0.01/pi`, `u(x,0) = -sin(pi x)` y `u(-1,t) = u(1,t) = 0`. La pérdida es la suma uniforme de los MSE del residual PDE, de la condición inicial y de las condiciones de frontera.

El código oficial fija `DEFAULT_NUM_DOMAIN_POINTS = 8192`, `DEFAULT_NUM_BOUNDARY_POINTS = 2048` y `DEFAULT_NUM_INITIAL_POINTS = 2048` en `src/pde/baseclass.py`. Por tanto, para este problema el pool de condiciones se desglosa en **2048 puntos iniciales + 2048 puntos de frontera** (2048 entre ambos extremos espaciales); son conjuntos distintos. El baseline oficial emplea el muestreo Hammersley por defecto de DeepXDE. La notebook genera esa secuencia de forma nativa para evitar instalar DeepXDE/scikit-optimize en Kaggle. Las fronteras usan selección pseudoaleatoria reproducible de los extremos, como en `GeometryXTime.random_boundary_points`.

El paper reporta tres ejecuciones por método; este artefacto fija cinco seeds por instrucción del proyecto. Las medias y desviaciones guardadas describen esas cinco ejecuciones y no son una réplica numérica de la tabla publicada.

## Evaluación y resultados

La métrica principal es el error relativo L2 sobre la misma grilla Cole–Hopf 1025×129 en todos los seeds. Se guardan también MSE, MAE y error absoluto máximo. Como comparación opcional, el notebook descarga el archivo FEM/COMSOL `ref/burgers1d.dat` del commit oficial fijado y calcula L2RE en sus 101 posiciones y 11 tiempos con la misma fórmula de `TesterCallback` del repositorio. Si la descarga no está disponible, se omite solo esa puntuación; la evaluación Cole–Hopf continúa.

Los resultados se escriben en `/kaggle/working/pinnacle-reference-burgers/`:

- `seed_0.json` … `seed_4.json`: estado, seed, épocas, tiempos, pérdidas, métricas y hashes por ejecución.
- `seed_0.csv` … `seed_4.csv`: pérdidas y evaluaciones por checkpoint.
- `summary.csv`: una fila por seed, incluyendo estados `complete`, `truncated` o `failed`.
- `manifest.json`: configuración, versiones, hardware, hashes de la grilla y de los archivos oficiales, y estado global.

El brazo L-BFGS del repositorio oficial es un método separado (`--method lbfgs`, que cambia a L-BFGS en la época 5000 dentro del presupuesto de iteraciones); no forma parte del Vanilla Adam reproducido aquí.

## Fuentes primarias

- Paper PINNacle, NeurIPS 2024: [PDF oficial de NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/8c63299fb2820ef41cb05e2ff11836f5-Paper-Datasets_and_Benchmarks_Track.pdf), [arXiv 2306.08827](https://arxiv.org/abs/2306.08827).
- Repositorio oficial: [i207M/PINNacle](https://github.com/i207M/PINNacle), commit `6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf`.
- Implementación Burgers: [src/pde/burgers.py](https://github.com/i207M/PINNacle/blob/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/src/pde/burgers.py).
- Conteos y condiciones: [src/pde/baseclass.py](https://github.com/i207M/PINNacle/blob/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/src/pde/baseclass.py).
- Arquitectura, inicialización, optimizador y épocas: [benchmark.py](https://github.com/i207M/PINNacle/blob/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/benchmark.py) y [src/optimizer/adam_lbfgs.py](https://github.com/i207M/PINNacle/blob/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/src/optimizer/adam_lbfgs.py).
- Muestreador DeepXDE fijado por el repositorio: [deepxde/geometry/sampler.py](https://github.com/i207M/PINNacle/blob/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/deepxde/geometry/sampler.py) y [deepxde/geometry/timedomain.py](https://github.com/i207M/PINNacle/blob/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/deepxde/geometry/timedomain.py).
- Referencia FEM/COMSOL opcional: [ref/burgers1d.dat](https://raw.githubusercontent.com/i207M/PINNacle/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/ref/burgers1d.dat).
- Evaluador oficial de referencia: [src/utils/callbacks.py](https://github.com/i207M/PINNacle/blob/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/src/utils/callbacks.py).

## Uso en Kaggle

Abre `burgers-pinnacle-reference.ipynb`, selecciona **GPU T4 x2** y ejecuta todas las celdas. El `kernel-metadata.json` solicita `NvidiaTeslaT4` y habilita Internet únicamente para el archivo FEM opcional. Antes de publicar mediante Kaggle CLI, sustituye `replace-with-your-kaggle-username` por tu slug de Kaggle. Esta entrega no publica ni ejecuta el notebook.
