# Baselines RBF-PIKAN — entrega de resultados

Se recuperaron las corridas GPU y se ejecutaron cuatro notebooks de resultados. No se ejecutaron tests, smoke tests, auditorías de convergencia ni validadores de integridad locales en esta entrega. Las mediciones de campo son posprocesado de checkpoints existentes; no consumen entrenamiento GPU.

| baseline | n | mean | sample_sd | reference |
| --- | --- | --- | --- | --- |
| Heat / RBF · EXP-0005 | 3 | 8.872368e-05 | 4.530022e-05 | Analítica · 512×256 |
| Heat / RBF · extensión | 2 | 0.0003626496 | 0.0003644286 | Analítica · 512×256 |
| Burgers / Sobol | 5 | 0.0395498 | 0.02564915 | Cole–Hopf · 1025×129 |
| Allen–Cahn | 5 | 0.01843233 | 0.003375772 | DeepXDE · 201×101 · provisional |
| DFG / Cd | 5 | 1.075206 | 0.09458527 | FEATFLOW QoI |
| DFG / Cl | 5 | -0.04652075 | 0.06925719 | FEATFLOW QoI |
| DFG / Δp | 5 | 0.07768156 | 0.0055397 | FEATFLOW QoI |

Media ± desviación estándar muestral entre semillas. Los IC t del CSV son descriptivos, no gates de aceptación.

## Qué cambió respecto al informe anterior

1. **DFG ya no está en cola:** r4 produjo las tres semillas faltantes. El estado ERROR corresponde a la importación de PyVista/IPython en visualización, posterior al entrenamiento. Se recuperaron los resultados sin otro envío GPU.
2. **Heat tenía un error documental:** ambas cohortes usaron 5,000 pasos y 4,096 puntos. La razón anterior para separarlas (20k frente a 5k) era incorrecta. Se mantienen separadas en esta entrega porque cambiaron versiones de los módulos de entrenamiento y no se auditó su equivalencia. Todos los checkpoints se puntuaron en 512×256.
3. **Allen–Cahn ahora tiene campo e interfaz para cinco semillas:** el score usa directamente la malla publicada 201×101. El cálculo histórico interpolado se conserva como otra columna. La referencia sigue provisional.
4. **Presupuesto r4 inconsistente:** budget declara dos semillas y 3,000 GPU-s; training y la autorización de continuación describen tres semillas y 3,600 GPU-s. Los archivos originales se preservan.

## heat

Error relativo L2 con cuadratura trapezoidal en la malla protegida 512×256, solución analítica, inferencia float64. Corrección del informe anterior: las dos cohortes Heat usaron 5,000 pasos y 4,096 puntos interiores; los 20,000 pasos eran el override de Burgers. Se conservan separadas: los hashes archivados de model.py, train.py y sampling.py difieren y esta entrega no ejecuta una auditoría de equivalencia ni validaciones locales. Los tres comparadores de EXP-0005 comparten los topes del protocolo, pero B-spline alcanza el tope temporal antes del de pasos; la tabla registra los pasos y segundos realmente realizados. Ningún entrenamiento nuevo.

[Notebook ejecutado](delivery/heat-results.ipynb) · [Métricas por semilla](delivery/heat-by-seed.csv) · [Resumen](delivery/heat-summary.csv) · [Gráfica interactiva](delivery/heat.html)

## burgers

Cinco pares existentes, 20,000 pasos por corrida; baseline Sobol fijo y RAD como contraste adverso. Se reutiliza la evaluación canónica Cole–Hopf de 1025×129 del experimento, sin reentrenamiento. RAD empeora en los cinco pares; esto solo descarta esta configuración, no toda focalización.

[Notebook ejecutado](delivery/burgers-results.ipynb) · [Métricas por semilla](delivery/burgers-by-seed.csv) · [Resumen](delivery/burgers-summary.csv) · [Gráfica interactiva](delivery/burgers.html)

## allen-cahn

Cinco semillas. Error de campo provisional, calculado directamente con checkpoints en los 201×101 puntos publicados de DeepXDE. Se entregan L2 trapezoidal y L2 discreto sin pesos; el primero es el score mostrado. También se conserva el cálculo histórico en 257×129 con referencia interpolada, para no confundir mallas. Posición de interfaz: cruce u=0 por semiplano. Anchura: distancia entre cruces u=−0.5 y +0.5; un nivel no alcanzado o cruces múltiples producen null, no cero. Referencia con dx=0.01. La convergencia BDF y la procedencia del solver publicado siguen pendientes; no se desbloquea el evaluador protegido.

[Notebook ejecutado](delivery/allen-cahn-results.ipynb) · [Métricas por semilla](delivery/allen-cahn-by-seed.csv) · [Resumen](delivery/allen-cahn-summary.csv) · [Gráfica interactiva](delivery/allen-cahn.html)

## dfg

Semillas 301–302 recuperadas de r3 y 303–305 completadas en r4, 1,200 s nominales por semilla. Kaggle r4 terminó en ERROR al importar PyVista en la celda de visualización: ModuleNotFoundError: No module named 'IPython.core.guarded_eval'. El campaign.json registra tres semillas completed y la evaluación previamente archivada marca execution_valid=true. r3 no tiene cierre válido del harness; se preserva su condición de evidencia recuperada. No se repiten entrenamientos. El budget de r4 declara 2 semillas/3,000 GPU-s, contradictorio con training de 3 semillas/3,600 GPU-s; esta entrega conserva y señala ese defecto. El total nominal de la campaña es 6,000 GPU-s. No hay campo FEM independiente; el error de campo queda sin medir. Umbrales ausentes, scientific_accepted=false.

[Notebook ejecutado](delivery/dfg-results.ipynb) · [Métricas por semilla](delivery/dfg-by-seed.csv) · [Resumen](delivery/dfg-summary.csv) · [Gráfica interactiva](delivery/dfg.html)

## DFG frente a la referencia

El error relativo de Cd es 80.73% y el de Δp 33.90%. Para Cl se reporta el error absoluto medio 0.069974, porque su referencia está cerca de cero. Las referencias Cd=5.57953523384, Cl=0.010618948146 y Δp=0.11752016697 proceden de [FEATFLOW](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html). No es una reproducción precisa del benchmark.

El tiempo de entrenamiento DFG observado suma 6000.158 GPU-s (nominal: 6,000). Los 0.158 s aproximados adicionales corresponden al cierre de pasos; no se informa el nominal como tiempo medido.

Las tablas conservan W&B y rutas de origen por semilla. Los hashes y veredictos históricos no se regeneraron. [Log de Kaggle](delivery/dfg-kaggle-logs.txt).

## Alcance científico

Entrega operativa de baselines exploratorios. `scientific_accepted=false`, `baseline_frozen=false`. Sigue pendiente una referencia convergida para Allen–Cahn y un campo FEM independiente para DFG; Heat conserva cohortes separadas y los comparadores de arquitectura tienen n=3. La evidencia no sostiene SOTA.

La referencia de Allen–Cahn es el archivo publicado por [DeepXDE](https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html), archivado por la campaña con commit y hash; su documentación no especifica solver ni tolerancias de generación. No se modifica el evaluador para desbloquear métricas.
