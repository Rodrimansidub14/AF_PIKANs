# Allen–Cahn DeepXDE v1.12.2 reference

Notebook Kaggle reproducible para el ejemplo oficial Allen–Cahn de DeepXDE, con PyTorch sobre CUDA y cinco seeds. Mantiene el PDE, la transformación dura, el FNN `[2, 20, 20, 20, 1]`, los 8.000 puntos interiores, 400 de frontera, 800 iniciales, Adam por 40.000 pasos y después L-BFGS.

## Ejecución en Kaggle

Abre `allen-cahn-deepxde-reference.ipynb`, habilita Internet y selecciona el acelerador **GPU T4x2**. La notebook comprueba dos T4 visibles, fija DeepXDE `1.12.2`, descarga la referencia por URL del tag y empieza el límite de 7.200 s antes de instalar o preparar datos. DeepXDE entrena con `cuda:0`; la receta estándar de este FNN no reparte el trabajo entre las dos GPU.

El tiempo de espera/asignación de Kaggle ocurre antes de que inicie Python y no se puede medir desde la notebook. Los 7.200 s cubren la instalación, descarga, preparación, entrenamiento y evaluación desde la primera celda. Un watchdog deja tres minutos para consolidar resultados.

## Referencia oficial y eje espacial

- Código upstream: [Allen_Cahn.py en DeepXDE v1.12.2](https://github.com/lululxvi/deepxde/blob/v1.12.2/examples/pinn_forward/Allen_Cahn.py).
- Campo: [Allen_Cahn.mat en DeepXDE v1.12.2](https://raw.githubusercontent.com/lululxvi/deepxde/v1.12.2/examples/dataset/Allen_Cahn.mat).
- Tag: `v1.12.2` (release commit `414e457`).
- SHA-256 registrado: `f9175685675685ab8d2a850f0d212d16c709f55ed89b8fe078493ddcd3724aa0`.
- Blob SHA-1 de Git: `2617d4be2746ea1689e04b9d29f0217ac1899a5f`.
- Formas observadas: `t: (1, 101)`, `u: (101, 201)`.

El archivo del tag contiene `t` y `u`, pero no la variable `x` que el código del ejemplo intenta leer. La notebook conserva `u` tal cual y reconstruye `x=linspace(-1,1,201)`: los extremos salen del `Interval(-1,1)` del código del mismo tag y el número de nodos del ancho de `u`. `meshgrid(x,t)` produce exactamente el orden de raster `(101,201)` requerido por el campo. No se interpola ni altera la solución de referencia. La irregularidad y la conversión también quedan registradas en `metadata.json` y en la provenance de ejecución.

## Métricas y estado

Por seed se guardan el error relativo L2 sobre todo el campo, el residual absoluto medio, errores máximos en IC y BC, y energía inicial/final/cambio/media. La energía es `integral_x [0.5*d*u_x^2 + 1.25*(u^2-1)^2] dx`, integrada por trapecios sobre la grilla de referencia. El tiempo y GPU-h asignadas se guardan por seed. El coste monetario queda nulo por defecto porque Kaggle no expone facturación; opcionalmente se puede definir `KAGGLE_T4X2_USD_PER_HOUR` si existe un precio externo aplicable.

Los archivos de resultados (`results.json`, `results.csv`, `provenance.json`) aparecen en `/kaggle/working/allen_cahn_deepxde_reference`. La notebook solo establece `score_valid=true` cuando las cinco seeds completan Adam, L-BFGS y la evaluación. Si el límite o el entorno impiden completar la receta, registra `truncated`, `blocked` o `worker_failed` con el motivo; esa corrida no cuenta como completada.

## Dependencias fijadas y alcance

Se instala DeepXDE `1.12.2`; PyTorch, CUDA y SciPy se toman del runtime Kaggle y sus versiones reales se registran en `provenance.json`. Si falta SciPy, se instala junto con DeepXDE. La notebook verifica que el backend sea PyTorch y que haya dos GPU T4 disponibles. No incluye una subida ni una ejecución en Kaggle; es el artefacto de referencia listo para abrir en Kaggle.
