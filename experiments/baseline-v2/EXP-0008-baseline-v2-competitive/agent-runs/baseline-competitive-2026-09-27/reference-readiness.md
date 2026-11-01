# Auditoría documental de referencias reutilizables

Esta auditoría combina resultados archivados y una referencia BDF candidata para Allen–Cahn; no corrió tests ni validadores locales. Las cuatro PDE vigentes son Heat, Burgers canónico, Allen–Cahn y DFG Re=20. Estado DFG actualizado el 2026-09-28: los scores antiguos de fuerzas se supersedieron con un re-score de tracción FEATFLOW directo en Kaggle GPU.

| Problema | Referencia disponible | Lectura para la siguiente fase |
|---|---|---|
| Heat | Solución analítica en `evals/problems/heat_1d/`; score 512×256. | La referencia es fuerte, pero la cuadratura y equivalencia de cohortes deben describirse antes de agregar cinco semillas. |
| Burgers canónico | Cole–Hopf 1025×129 en `evals/problems/burgers_pinnacle_1d/`. | Usar esta instancia `ν=0.01/π`; el benchmark Burgers estacionario histórico no es el mismo caso. |
| Allen–Cahn | Campo DeepXDE publicado 201×101 y referencia independiente nueva BDF AC-L4, 4096 intervalos espaciales. | La candidata cumple dos pares espaciales seguidos y chequeo temporal; sigue `evaluation_ready=false` y requiere revisión externa. El score de las semillas existentes sigue provisional. |
| DFG | QoIs oficiales FEATFLOW; el re-score CUDA aplica σn y completó 10/10 checkpoints; sin campo FEM independiente. | Cd medio RBF 3.894 (error 30.2%); Δp error 35.4%; falta referencia de campo y comparador neural publicado. |

## Allen–Cahn: qué muestra realmente el estudio BDF archivado

El estudio usó discretización espacial de diferencias finitas de segundo orden con 513, 1025 y 2049 nodos; integró en tiempo con BDF y Jacobiano analítico disperso. Reporta para el cambio relativo ponderado de campo `1.170267e-4` (513→1025) y `6.273861e-5` (1025→2049), así como `1.264968e-8` al apretar tolerancias temporales en 2049 nodos. El cambio de energía máximo del último refinamiento es `3.256636e-4`. Estos valores pasan los límites absolutos archivados de campo (<0.005) y energía (<0.001), pero el cociente de diferencias espaciales es `0.5361`, correspondiente a un orden aparente de apenas `~0.90` si se lo interpretara asintóticamente. Dos diferencias no establecen ese régimen; por ello el criterio de cambio subsiguiente/efecto sigue sin definición operativa y el artefacto marca `canonical_reference=false`.

La nueva referencia candidata finalizó en 12.3 s de CPU (solo referencia, no entrenamiento GPU), con niveles AC-L0…AC-L4 y BDF de orden variable. El último par (2048→4096 intervalos) dio cambio relativo espacio-tiempo `4.6086e-6` y peor snapshot `1.1053e-5`; el anterior (1024→2048) dio `1.8448e-5` y `4.4261e-5`. El chequeo temporal en 4096 intervalos dio `2.4211e-10` espacio-tiempo y `5.5596e-10` máximo por snapshot, frente a tolerancias `5e-5` y `1e-7`. El artefacto conserva campos, versiones y hashes en [esta corrida](reference/ac_bdf/runs/20260927T175610Z-8f26e215/manifest.json). El manifiesto mantiene `evaluation_ready=false`; antes de usarlo para reemplazar el score provisional hay que revisar procedencia y comparar sobre malla común.

El investigador pidió que las nuevas corridas de baseline se hagan en GPU. Esta referencia corta fue ejecutada antes de esa aclaración y no cuenta como entrenamiento ni como evidencia de rendimiento.

Fuentes internas: [informe BDF archivado](../burgers-rad-exp006/allen-cahn-reference/study/study_report.md), [métricas BDF](../burgers-rad-exp006/allen-cahn-reference/study/metrics.json), [resultados entregados](../baseline-close-2026-09-27/delivery/baselines-report.md).

## DFG: resultado actualizado y límites

El primer score-only usó una reducción tangencial cuya orientación no concordaba con la tracción de FEATFLOW. El benchmark define las fuerzas por la integral de σn, con σ=ν∇u−pI. Se re-evaluaron en Kaggle los diez checkpoints guardados, integrando directamente esa tracción y usando la normal radial saliente; no se entrenó de nuevo. La implementación corregida cambia Cd/Cl, pero deja intactos Δp y los residuos held-out.

El RBF da Cd=3.89365 ± 0.12040 frente a 5.579535 (error relativo medio por semilla 30.22%); Cl=0.02590 ± 0.07843 (error absoluto medio 0.05598, referencia 0.01062); y Δp=0.07594 ± 0.00336 frente a 0.117520 (35.38%). El campo FEM sigue faltando. El exceso de residuos cerca del cilindro es descriptivo y no identifica causalidad.

El cambio canónico está en src/research_pipeline/sciml/dfg.py y tiene una prueba analítica de regresión agregada que no se ejecutó localmente, de acuerdo con la instrucción de saltar validaciones locales. El detalle, notebook y telemetría están en competitive-baseline-execution.md y bundles/dfg/force-score-only/.
