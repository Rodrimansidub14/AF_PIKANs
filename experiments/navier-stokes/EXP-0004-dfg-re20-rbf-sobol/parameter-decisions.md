# Decisiones de parámetros — EXP-0004 DFG Re=20

Estado: propuesta previa al run. Los valores de cómputo son provisionales y
no constituyen aprobación de ejecución remota.

| Decisión | Propuesta | Motivo / limitación |
|---|---|---|
| PDE | steady incompressible Navier–Stokes, `ν=.001`, `ρ=1` | Caso DFG 2D-1 Re=20 existente; p cinemática. |
| Geometría | canal 2.2×.41; cilindro radio .05 centrado (.2,.2) | Conservar benchmark y máscara fluida del verifier. |
| Entrada | perfil parabólico máximo .3; `v=0` | Contrato FEATFLOW; velocidad media .2. Evita confundir U media y máxima. |
| Red | candidata [2,16,16,3] con 11 RBF por arista | Extiende el control de 1D a dos coordenadas y tres campos; falta implementar/autodiferenciar. |
| Pool interior | 16.384 puntos Sobol 2D con rechazo del cilindro | Potencia de dos antes del filtro; pool filtrado exacto guardado. Rechazo altera regularidad QMC cerca de la geometría. |
| Fronteras | 512 puntos por componente; círculo uniforme en ángulo | Arclength uniforme en cada superficie; esquinas deben deduplicarse o asignarse a una sola frontera. |
| Muestreo adaptativo | No en el control | RAD/RAR-D queda como campaña comparativa luego; separar adaptación del modelo. |
| Batch | 2.048 propuesto | Mayor throughput potencial en T4; confirmar memoria y cómo PINA agrega cada condición. |
| Optimización | Adam `lr=0.001`, 20k pasos o 1.200 s | Presupuesto piloto; registrar pasos efectivos y pérdida por condición. |
| Semillas | 101,102,103 | Dispersión exploratoria; no claim estadístico. |
| Precisión | Float32 preferida para probar en T4; comparar gradientes/residuo contra float64 en smoke | T4 tiene bajo throughput FP64; cambio requiere decisión antes del run. No mezclar precisiones dentro de una comparación. |
| Evaluación | Pools QMC independientes por residual y frontera | Sin fuga; regiones near-cylinder y wake son diagnósticos, no umbrales. |
| Calidad de campo | Bloqueada; QoI pendiente | La referencia FEM, fuerza por tracción e interpolación de presión requieren convergencia auditada. |

Antes de ejecutar faltan: cerrar precisión, balance de pérdidas, forma exacta
de la condición natural de salida, validar el runner multi-salida y resolver
cómo se distribuirán las semillas si Kaggle asigna T4×2. No abrir claim SOTA.

Revisar señales del run anterior antes de fijar estos valores. Solo throughput,
telemetría e integración RBF son directamente portables. Allen–Cahn puede
informar derivadas de segundo orden, pero no determina el muestreo alrededor
del cilindro. Las proporciones de puntos cerca del cilindro/wake se definirán
como hipótesis propia y compararán contra el pool Sobol base; no se retocarán
por el error de evaluación.
