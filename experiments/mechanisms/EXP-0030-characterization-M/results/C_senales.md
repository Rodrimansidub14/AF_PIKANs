# EXP-0030, bloque C: señales por recálculo (exploratorio, post hoc)

Cuotas = traza del bloque GN por grupo físico sumada sobre todas las aristas y capas, entre la traza total. `cond p90` = percentil 90 de la condición del bloque normalizado (B/m + 1e-3 I) sobre aristas vivas, mediana de las capas. Solo M registra cuotas (`gtrace_*`); E y B antiguos no.

## Cuotas por grupo y condición en recálculos escogidos (M)

| problema | semilla | paso del recálculo | mecánica | eléctrica | equilibrio | borde | cond p90 |
|---|---|---|---|---|---|---|---|
| conv | 28 | 0 | 0.095 | 0.000 | 0.905 | 0.000 | 1.04e+04 |
| conv | 28 | 500 | 0.418 | 0.001 | 0.580 | 0.001 | 9.35e+03 |
| conv | 28 | 3000 | 0.615 | 0.001 | 0.383 | 0.001 | 9.97e+03 |
| conv | 28 | 8000 | 0.621 | 0.000 | 0.378 | 0.001 | 1.04e+04 |
| conv | 28 | 18000 | 0.810 | 0.000 | 0.190 | 0.000 | 1.06e+04 |
| conv | 29 | 0 | 0.088 | 0.000 | 0.911 | 0.000 | 1.03e+04 |
| conv | 29 | 500 | 0.421 | 0.001 | 0.578 | 0.001 | 1.01e+04 |
| conv | 29 | 3000 | 0.462 | 0.001 | 0.537 | 0.001 | 1.02e+04 |
| conv | 29 | 8000 | 0.791 | 0.000 | 0.208 | 0.000 | 1.05e+04 |
| conv | 29 | 18000 | 0.783 | 0.000 | 0.216 | 0.000 | 1.05e+04 |
| conv | 30 | 0 | 0.056 | 0.000 | 0.944 | 0.000 | 1.02e+04 |
| conv | 30 | 500 | 0.284 | 0.000 | 0.715 | 0.001 | 9.95e+03 |
| conv | 30 | 3000 | 0.385 | 0.001 | 0.614 | 0.001 | 9.9e+03 |
| conv | 30 | 8000 | 0.784 | 0.000 | 0.216 | 0.000 | 1.01e+04 |
| conv | 30 | 18000 | 0.732 | 0.000 | 0.268 | 0.000 | 1.02e+04 |
| dire | 28 | 0 | 0.003 | 0.001 | 0.994 | 0.002 | 1.07e+04 |
| dire | 28 | 500 | 0.024 | 0.002 | 0.971 | 0.002 | 9.81e+03 |
| dire | 28 | 3000 | 0.062 | 0.010 | 0.924 | 0.004 | 9.72e+03 |
| dire | 28 | 8000 | 0.050 | 0.009 | 0.938 | 0.004 | 9.62e+03 |
| dire | 28 | 18000 | 0.042 | 0.008 | 0.946 | 0.004 | 9.53e+03 |
| dire | 29 | 0 | 0.003 | 0.001 | 0.993 | 0.002 | 1.07e+04 |
| dire | 29 | 500 | 0.048 | 0.006 | 0.943 | 0.003 | 9.87e+03 |
| dire | 29 | 3000 | 0.062 | 0.008 | 0.925 | 0.004 | 9.83e+03 |
| dire | 29 | 8000 | 0.076 | 0.011 | 0.908 | 0.005 | 9.93e+03 |
| dire | 29 | 18000 | 0.048 | 0.008 | 0.940 | 0.004 | 1.01e+04 |
| dire | 30 | 0 | 0.004 | 0.001 | 0.992 | 0.002 | 1.05e+04 |
| dire | 30 | 500 | 0.063 | 0.007 | 0.927 | 0.002 | 1.01e+04 |
| dire | 30 | 3000 | 0.084 | 0.012 | 0.901 | 0.003 | 9.86e+03 |
| dire | 30 | 8000 | 0.040 | 0.007 | 0.948 | 0.004 | 9.79e+03 |
| dire | 30 | 18000 | 0.080 | 0.013 | 0.904 | 0.003 | 1e+04 |

## b elegido por P2, eventos de P4 en sombra y costo del recálculo (M)

`b:conteo` = veces que se eligió cada b en los recálculos. Eventos de P4 = pérdida de un paso > 5× el mínimo de los 200 previos (solo observa). Primer / mediano = pasos de esos eventos.

| problema | semilla | b:conteo | recálculos | eventos P4 | primer / mediano paso | s por recálculo |
|---|---|---|---|---|---|---|
| conv | 28 | 0.1:10 0.2:12 | 22 | 39 | 502 / 8915 | 10.7 |
| conv | 29 | 0.1:11 0.2:9 0.3:1 0.5:1 | 22 | 38 | 252 / 9646 | 10.6 |
| conv | 30 | 0.1:7 0.2:15 | 22 | 38 | 253 / 10231 | 10.7 |
| dire | 28 | 0.5:22 | 22 | 68 | 253 / 10301 | 11.7 |
| dire | 29 | 0.5:22 | 22 | 64 | 502 / 10701 | 10.5 |
| dire | 30 | 0.5:22 | 22 | 66 | 252 / 10201 | 10.4 |

## Condición del bloque (p90, mediana de capas) a lo largo del entrenamiento: E y M, converso

| brazo | semilla | hardware | paso 0 | paso 1000 | paso 5000 | paso 10000 | paso 19000 |
|---|---|---|---|---|---|---|---|
| E | 28 | T4 | 1.04e+04 | 9.12e+03 | 1.01e+04 | 1.08e+04 | 1.08e+04 |
| E | 29 | H100 | 1.03e+04 | 9.54e+03 | 1.04e+04 | 1.05e+04 | 1.07e+04 |
| E | 30 | H100 | 1.02e+04 | 9.4e+03 | 1.06e+04 | 1.09e+04 | 1.09e+04 |
| E | 31 | A100-80 | 1.08e+04 | 1e+04 | 1.07e+04 | 1.08e+04 | 1.08e+04 |
| E | 32 | A100-80 | 1e+04 | 1.02e+04 | 1.03e+04 | 1.05e+04 | 1.06e+04 |
| M | 28 | A100-80 | 1.04e+04 | 9.39e+03 | 1.03e+04 | 1.05e+04 | 1.05e+04 |
| M | 29 | A100-80 | 1.03e+04 | 1.01e+04 | 1.04e+04 | 1.05e+04 | 1.06e+04 |
| M | 30 | A100-80 | 1.02e+04 | 9.83e+03 | 9.67e+03 | 9.92e+03 | 9.99e+03 |

## Por arista, dentro de una corrida (descriptivo; n = aristas, no semillas)

| problema | semilla (M) | capa | aristas vivas | Spearman traza-cond | Spearman cuota eléctrica-cond |
|---|---|---|---|---|---|
| conv | 28 | capa 0 | 64 | -0.23 | -0.07 |
| conv | 28 | capa 1 | 256 | -0.09 | +0.12 |
| dire | 29 | capa 0 | 64 | -0.71 | -0.54 |
| dire | 29 | capa 1 | 256 | -0.28 | -0.63 |