# Informe de avance — baselines RBF-PIKAN para cuatro problemas

**Fecha:** 27 de septiembre de 2026  
**Estado:** exploratorio; DFG Re=20 r3 falló después de entrenar dos semillas y r4 continúa solo con las tres faltantes.  
**Aceptación científica:** `scientific_accepted=false`.

## Propósito y alcance

Esta campaña prepara baselines reproducibles e instrumentados para (1) calor 1D en una varilla, (2) Burgers viscoso 1D, (3) Allen–Cahn 1D y (4) el benchmark DFG 2D-1 de Navier–Stokes a Re=20. Poisson se mantiene diferido. La pregunta experimental no es solo cuál red obtiene menor error: es qué señales medibles —cobertura de puntos, autodiferenciación, residuo, ocupación RBF, sensibilidad, trazas físicas y costo— pueden justificar una focalización posterior.

Se aplica un ciclo inspirado en [autoresearch de Karpathy](https://github.com/karpathy/autoresearch/blob/master/program.md?plain=1): preservar el evaluador y el presupuesto, guardar cada configuración por hash, cambiar una hipótesis por iteración y clasificar el intento como válido, descartado o fallido. La corrida de ahora es el registro de control con RBF fija; no es todavía una búsqueda de hiperparámetros ni un resultado SOTA.

## Protocolo que se ejecutó

La generación de puntos usa **Sobol scrambled fijo**, con streams independientes y semillas archivadas. Es muestreo cuasi-Monte Carlo de baja discrepancia, reproducible; no es muestreo uniforme aleatorio IID y no se adapta durante el entrenamiento. Se separan los puntos de entrenamiento de los pools held-out. Cada informe conserva los tamaños, discrepancia, distancias a vecino más cercano y coordenadas observadas.

Heat y Allen–Cahn r3 entrenaron dos semillas en paralelo, una por cada Tesla T4. Burgers reutiliza el contraste previo de cinco pares Sobol/RAD porque repetirlo no añadiría evidencia. DFG r3 entrenó las semillas 301–302 durante 1,200 s cada una en dos Tesla T4, pero el runtime falló al indexar un symlink de W&B antes de cerrar el manifiesto de corrida. Los artefactos de esas dos semillas se preservaron y sincronizaron. El bundle r4, con el arreglo de inventario de archivos y sin cambios en PDE/modelo/puntos/optimizador, ejecuta solo 303–305.

El diseño de autoresearch aún necesita congelar umbrales confirmatorios antes de afirmar aceptación. En r3 el gate solo declara Allen–Cahn `normalized_residual_mean`, sin umbral; Heat no tiene métrica primaria registrada en el gate. Por eso la validación técnica de artefactos no equivale a pasar una puerta científica.

## Resultados cuantitativos disponibles

Las desviaciones son desviaciones estándar muestrales entre semillas. Las métricas con solo dos semillas se consideran screening; no tienen precisión confirmatoria.

| Problema / evidencia | Métrica primaria observada | Resultado | Estado e interpretación |
|---|---|---:|---|
| Heat 1D, r3, semillas 104–105 | Error relativo L2 contra solución analítica, cuadrícula 129×65 | `3.626e-4 ± 3.645e-4` (semillas: `6.203e-4`, `1.049e-4`) | `execution_valid=true`; solo n=2. No se mezcla con EXP-0005: r3 entrenó 5,000 pasos y 4,096 puntos interiores, frente al presupuesto mayor de la pantalla de arquitectura. Falta evaluar los mismos checkpoints en la cuadrícula protegida 512×256. |
| Heat EXP-0005, 3 semillas por arquitectura | Error relativo L2 | RBF `8.875e-5 ± 4.543e-5`; MLP tanh `2.520e-4 ± 8.872e-5`; B-spline cúbica `2.300e-5 ± 9.487e-6` | En esta pantalla, B-spline es más precisa que RBF y MLP. RBF supera MLP, pero se retiene por la trazabilidad de sus bases, no por superioridad universal. |
| Burgers canónico, EXP-0006, cinco pares | Error relativo L2, Sobol vs RAD | Sobol `0.03955 ± 0.02565`; RAD `0.21269 ± 0.07526`; diferencia pareada RAD−Sobol `0.17314`, IC 95% t pareado `[0.07859, 0.26768]` | RAD empeoró en los cinco pares; media geométrica del cociente RAD/Sobol `5.865`. En este protocolo se descarta RAD residual-global como siguiente configuración por defecto. |
| Allen–Cahn 1D, cinco semillas protocol-matched | Media del residuo normalizado | `1.220e-3 ± 8.412e-4`; intervalo t 95% aproximado `[1.76e-4, 2.26e-3]` | Seeds 101–103 de EXP-0003 se pueden combinar con 104–105: coinciden exactamente PDE, arquitectura, pools, Adam, 20k pasos, Sobol y hashes de `model.py`, `train.py`, `sampling.py` y evaluador. El error de campo sigue provisional por la referencia publicada. |
| DFG 2D-1 Re=20, r3 parcial (semillas 301–302) | `Cd`, `Cl`, `Δp` contra FEATFLOW | `Cd=1.0154 ± 0.1061` (error relativo `81.8% ± 1.9%`); `Cl=−0.03966 ± 0.11649`; `Δp=0.07527 ± 0.00709` (error relativo `36.0% ± 6.0%`) | Dos procesos completaron su presupuesto de 1,200 s en T4, pero el harness abortó al hashear un symlink externo de W&B; n=2 es solo diagnóstico y no constituye baseline cerrado. 303–305 no se ejecutaron en r3. El r4 continúa esas tres semillas con el arreglo de runtime. |

En DFG r3, los residuos held-out globales fueron continuidad `0.04100 ± 0.00579`, momentum-x `0.02815 ± 0.00007` y momentum-y `0.03211 ± 0.00590`; cerca del cilindro, continuidad `0.12437 ± 0.01914`, momentum-x `0.09228 ± 0.00399` y momentum-y `0.11238 ± 0.02529`. El error RMS de velocidad de no-deslizamiento en el cilindro fue `0.02780 ± 0.00325`; el error RMS de tracción normal en salida, `0.00320 ± 0.00044`. Las dos GPU telemetrías confirman Tesla T4, ranuras 0 y 1, 122 muestras cada una; utilización media ~28.3% y ~28.1%, memoria media ~250 MiB y pico 323 MiB. Las corridas rindieron 15,838 y 15,671 pasos en 1,200 s. Son estimaciones exploratorias con solo dos semillas, y el coeficiente de sustentación está cerca de cero, por lo que su error relativo no es una métrica estable.

En Allen–Cahn, el residuo normalizado RMSE held-out de cinco semillas es `1.722e-3 ± 1.255e-3` (IC 95% t aproximado `[1.63e-4, 3.28e-3]`). La energía decrece en promedio `87.82% ± 0.30%`; las trazas impuestas de condición inicial y frontera quedan cerca de precisión numérica. Estos checks físicos no sustituyen el error de campo.

### Contraste con literatura y límites de comparabilidad

- **Burgers:** la formulación coincide con el caso viscoso canónico de [Raissi et al. (2019)](https://www.aer.com/siteassets/21data/raissietal2019.pdf): `ν=0.01/π`, `u(x,0)=−sin(πx)`, dominio `[-1,1]×[0,1]`, bordes de Dirichlet y referencia Cole–Hopf. El artículo reporta error relativo L2 de `6.7e-4`; el control RBF Sobol actual es aproximadamente 59 veces mayor. Es un ancla de literatura, no una comparación perfectamente pareada de optimizer, presupuesto ni malla. Los resultados no son SOTA.
- **Allen–Cahn:** se usa el problema de [DeepXDE 1.12.2](https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html): difusión `0.001`, reacción `5(u−u³)`, condición inicial `x²cos(πx)` y Dirichlet `−1`. La guía muestra 8k puntos interiores, 400 frontera, 800 inicial y 40k Adam + L-BFGS. Este baseline usa 20k Adam y trazas duras, así que el residuo no es comparable directamente con su error de campo. La referencia `.mat` de DeepXDE está fijada por commit y hash, pero el origen numérico/tolerancia de esa solución no está documentado.
- El estudio reciente [vRBA](https://pmc.ncbi.nlm.nih.gov/articles/PMC12967272/) reporta errores bajos, pero para Allen–Cahn cambia a `k=1e-4` y condiciones periódicas/Fourier, frente a `k=1e-3` y Dirichlet duros aquí. Sus números sirven como contexto metodológico, no como ranking directo.
- **DFG:** [FEATFLOW DFG 2D-1 Re20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html) fija geometría, condiciones y QoIs de arrastre, sustentación y caída de presión; referencias `Cd=5.57953523384`, `Cl=0.010618948146`, `Δp=0.11752016697`. Las dos semillas parciales ya muestran errores de 81.8% en arrastre y 36.0% en `Δp`; son evidencia de diagnóstico, no una reproducción competitiva del benchmark. No hay campo FE independiente para error L2.
- **Heat:** la solución es analítica y el criterio principal es error relativo L2 en una malla fija held-out. No se ha encontrado un score de artículo con entrenamiento, malla y presupuesto suficientemente pareados. El resultado se reporta contra la solución exacta, no se etiqueta como récord de literatura.
- El benchmark [PINNacle](https://arxiv.org/abs/2306.08827) y el método [EI-RAR/EI-Grad (2024)](https://doi.org/10.1109/ACCESS.2024.3422224) son referencias de evaluación y muestreo adaptativo sobre varios PDEs. No se transplantan sus scores: los problemas, redes, referencia, optimización y presupuesto no coinciden.

## Señales observadas que pueden guiar mecanismos

Las señales justifican experimentos distintos por PDE; no existe todavía un escalar común ni un umbral universal.

| Familia de señal | Medición | Lectura limitada a este baseline |
|---|---|---|
| Muestreo | Sobol centrada: Heat aproximadamente `5.30e-8`; Allen–Cahn `1.39e-8`. Mediana de vecino más cercano: Heat `0.00976–0.01103`; Allen–Cahn `0.00655–0.00753`. | Cuantifica cobertura del pool fijo. Las distancias no se comparan sin normalizar por geometría y dimensión efectiva. |
| RBF | Heat: capas 1–2 usan efectivamente ~`58–59` y `52–53` bases; hay `93–94` y `100–106` bases con ocupación `<1%`. Allen–Cahn (n=5): capa 2 usa ~`115.3 ± 3.9` bases efectivas y capa 1 ~`77.7 ± 1.8`; ocupación `<1%` en promedio 74.6 y 32 bases. | La ocupación y el conteo efectivo hacen observable uso/desuso por capa. Son candidatos a explorar con localización del error, no prueba de que bases poco ocupadas causen error. |
| Derivadas | Allen–Cahn `RMS(u_xx)=108.1 ± 29.6`; Heat r3 `RMS(u_xx)≈4.61` en ambas semillas. | Señala escalas de diferenciación muy distintas por PDE; comparar valores crudos o focalizarlos con el mismo corte sería injustificado. |
| Error localizado | Burgers Sobol: RMSE del frente media `0.2161`, fuera del frente `0.00741`; ratio de medias ~`29.2`. | El score global oculta un frente pronunciado. La siguiente hipótesis debería comprobar si señales locales de frente predicen error, no aumentar puntos por residuo global sin control. |
| Estabilidad física | AC energía y trazas; Heat/Burgers trazas de borde/inicial; DFG held-out residuals de continuidad/momentum por región, cuatro fronteras/no-slip y QoIs. | En las dos semillas DFG parciales, residual y error de velocidad son mayores cerca del cilindro que en campo lejano. Es una señal medible para localizar el siguiente error; no prueba causalidad. |
| Costo | 5k Heat + 20k AC; r3 telemetría válida, 100/101 muestras por T4. Utilización media observada `35.46%` y `34.32%`, memoria pico `327 MiB`. | Confirma actividad GPU; los casos son pequeños y limitados por el trabajo de segundo orden/overhead. La columna `device` en `training_summary.json` dice `cpu`, contradicción con hardware CUDA y telemetría; corregir la instrumentación antes de tomar ese campo como evidencia. |

## Lectura respecto del objetivo de mecanismo

El baseline ya aporta señales observables para empezar a diseñar hipótesis; todavía no demuestra una regla que generalice. La justificación de RBF es interpretabilidad estructural: los centros y anchos fijos permiten cuantificar cobertura y ocupación de bases y relacionarlas con error, derivadas y sensibilidad. La comparación de arquitectura evita una conclusión exagerada: RBF aventaja al MLP en la pantalla previa, B-spline es mejor en Heat y el error Burgers sigue muy lejos de la referencia canónica.

El primer mecanismo no debería reutilizar el resultado RAD adverso tal como está. Para cada PDE debe haber una señal local normalizada y una única intervención: (a) Burgers, error/residuo y ocupación cerca del frente; (b) Allen–Cahn, interfaz y evolución temporal, condicionadas por residuo y `u_xx`; (c) Heat, gradiente temporal/espacial y vecindad de frontera/tiempo inicial; (d) DFG, gradientes y residuo alrededor del cilindro, estela y paredes, contrastados con fuerzas y presión. Cada candidato se compara contra el Sobol fijo con semillas pareadas, mismo evaluador, mismo presupuesto T4 y un criterio de parada antes de leer resultados.

Orden propuesto al salir DFG de la cola:

1. Completar las semillas DFG 303–305 en r4, combinar con trazabilidad las semillas 301–302 de r3, y contrastar las cinco contra `Cd`, `Cl` y `Δp`; si falla física o numéricamente, conservar el fallo como descarte y arreglar el runner antes de explorar.
2. Puntuar Heat en la cuadrícula held-out protegida 512×256 usando los checkpoints existentes; actualizar el baseline de dos semillas y medir su variación.
3. Congelar métrica primaria, gates físicos y costo por PDE antes de la siguiente campaña; no afinar esos gates mirando los mismos scores.
4. Elegir una intervención de señal por problema y correrla como experimento separado estilo autoresearch (`keep`, `discard`, `crash`), conservando Sobol como control.
5. Si el método se ve prometedor, confirmar con al menos cinco semillas y un comparador SciML de literatura bajo presupuesto y métrica realmente pareados.

## Artefactos reproducibles

- Heat/Allen–Cahn r3: [evaluación y hashes](heat-ac-extension-r3/download/evaluation.json), [notebook](heat-ac-extension-r3/download/pikan-bcc43eb34dfe1060/analysis/allen_cahn-heat-results.ipynb), [señales cuantitativas con unidades](heat-ac-extension-r3/download/pikan-bcc43eb34dfe1060/analysis/quantitative-signals-with-units.csv), [curvas de pérdida](heat-ac-extension-r3/download/pikan-bcc43eb34dfe1060/analysis/training-loss-curves.html), [señales internas](heat-ac-extension-r3/download/pikan-bcc43eb34dfe1060/analysis/training-internal-signals.html), [muestreo Heat](heat-ac-extension-r3/download/pikan-bcc43eb34dfe1060/analysis/heat-sampling.html), [muestreo Allen–Cahn](heat-ac-extension-r3/download/pikan-bcc43eb34dfe1060/analysis/allen_cahn-sampling.html).
- Burgers EXP-0006: [notebook](../burgers-rad-exp006-v4/download-attempts/attempt-01/pikan-4393e173686298ee/analysis/burgers-results.ipynb), [comparación pareada](../burgers-rad-exp006-v4/download-attempts/attempt-01/pikan-4393e173686298ee/analysis/paired-runs.csv), [resumen primario](../burgers-rad-exp006-v4/download-attempts/attempt-01/pikan-4393e173686298ee/analysis/paired-primary-summary.csv), [señales cuantitativas](../burgers-rad-exp006-v4/download-attempts/attempt-01/pikan-4393e173686298ee/analysis/quantitative-signals-with-units.csv), [puntos RAD por semilla](../burgers-rad-exp006-v4/download-attempts/attempt-01/pikan-4393e173686298ee/analysis/rad-points-201.html).
- Allen–Cahn EXP-0003: [resumen de campaña](../allen-cahn-exp003/analysis/campaign-summary.md) y métricas por corrida bajo `../allen-cahn-exp003/download/`.
- W&B: [proyecto AF_PIKANS](https://wandb.ai/man22611-uvg/AF_PIKANS). Cuatro runs r3 sincronizados: Heat seeds 104/105 `koh98uwg`, `2syfumxn`; Allen–Cahn seeds 104/105 `9s0fzu99`, `wf1fx5ui`. El resto de los 26 runs válidos anteriores ya estaban en W&B; se sincronizaron cinco runs que aún estaban offline.
- DFG r2 falló antes del entrenamiento; los [logs completos](dfg-re20-gpu-r2/kaggle-logs-r2.json) se conservan. DFG r3 ejecutó semillas 301–302 en dos T4 y guardó los artefactos, pero el harness falló al calcular hashes al encontrar el symlink `wandb/.../logs/debug-core.log` que resolvía fuera de la carpeta de corrida. Ambas corridas alcanzaron 1,200 s y W&B se sincronizó para `cy8be123` y `05m949wk`; `evaluation.json` del harness no existe porque no se escribió `run_summary.json`, así que el resultado se conserva como parcial y `scientific_accepted=false`. Se agregó la regresión `test_artifact_hashing_ignores_wandb_symlink_outside_run_directory`, observada fallar antes del arreglo y pasar después; el runtime ahora excluye symlinks del inventario de archivos regulares. Suite total: 166 pruebas pasadas, 5 advertencias de dependencias.
- DFG r4 continúa solo las semillas 303–305 con el mismo presupuesto de 1,200 s por semilla y 3,600 GPU-s restantes; los 2,400 GPU-s ya usados por 301–302 mantienen el presupuesto total del piloto en 6,000 GPU-s. Bundle `c349281ede3181b8984ee0de6bf95db8b7406a77e40daa07a4bcdc197ab93a2b`; [kernel privado de Kaggle r4](https://www.kaggle.com/code/rodrimandub/af-pikans-dfg-re20-baseline-20260927-r4). La única diferencia de código es el manejo seguro del symlink durante el hash; PDE, modelo, Sobol, pools, entrenamiento y métricas se conservan.

## Referencias

Las referencias consultadas están además en `references.bib` de esta campaña y en la bibliografía de PIKAN-Vault.

- Raissi, Perdikaris & Karniadakis, 2019, *Physics-informed neural networks*, [JCP](https://www.aer.com/siteassets/21data/raissietal2019.pdf).
- DeepXDE, [ejemplo canónico Allen–Cahn v1.12.2](https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html).
- FEATFLOW, [benchmark DFG 2D-1 Re=20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).
- Wang et al., 2023, [PINNacle benchmark](https://arxiv.org/abs/2306.08827).
- Liu et al., 2024, [EI-RAR/EI-Grad](https://doi.org/10.1109/ACCESS.2024.3422224).
- vRBA, 2026, [estudio reciente de muestreo residual adaptativo](https://pmc.ncbi.nlm.nih.gov/articles/PMC12967272/); se usa con la salvedad de que sus problemas no emparejan exactamente estas condiciones.
- Karpathy, [autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md?plain=1).
