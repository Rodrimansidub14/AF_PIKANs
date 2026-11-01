# K. Uso de las aristas (post hoc, exploratorio)

Estados finales. Importancia de arista = sqrt(Σ_g ocupación_g · c_g²) con coeficientes efectivos. `f90` = fracción de aristas que reúne el 90 % de la energía (menor = más concentrada); `gini` = desigualdad. Mediana entre semillas por brazo y capa.

| problema | brazo | n | capa | f90 | gini |
|---|---|---|---|---|---|
| converse | B | 5 | 0 | 0.70 | 0.22 |
| converse | B | 5 | 1 | 0.72 | 0.20 |
| converse | B | 5 | 2 | 0.15 | 0.50 |
| converse | E | 5 | 0 | 0.47 | 0.40 |
| converse | E | 5 | 1 | 0.64 | 0.26 |
| converse | E | 5 | 2 | 0.21 | 0.55 |
| converse | M | 3 | 0 | 0.56 | 0.32 |
| converse | M | 3 | 1 | 0.67 | 0.24 |
| converse | M | 3 | 2 | 0.26 | 0.50 |
| direct | B | 3 | 0 | 0.34 | 0.45 |
| direct | B | 3 | 1 | 0.52 | 0.33 |
| direct | B | 3 | 2 | 0.15 | 0.71 |
| direct | E | 3 | 0 | 0.23 | 0.57 |
| direct | E | 3 | 1 | 0.55 | 0.32 |
| direct | E | 3 | 2 | 0.13 | 0.71 |
| direct | M | 3 | 0 | 0.45 | 0.38 |
| direct | M | 3 | 1 | 0.58 | 0.30 |
| direct | M | 3 | 2 | 0.18 | 0.67 |

## Similitud de las funciones de la capa 0 (correlación mediana entre aristas; misma semilla = mismo modelo inicial)

| problema | par | semillas | correlación mediana |
|---|---|---|---|
| converse | B vs E | 5 | 0.54 (0.47–0.59) |
| converse | B vs M | 3 | 0.62 (0.55–0.62) |
| converse | E vs M | 3 | 0.79 (0.55–0.80) |
| converse | mismo brazo, semillas distintas (referencia) | 20 | -0.04 (-0.29–0.20) |
| direct | B vs E | 3 | 0.44 (0.17–0.46) |
| direct | B vs M | 3 | 0.47 (0.31–0.65) |
| direct | E vs M | 3 | 0.45 (0.25–0.46) |
| direct | mismo brazo, semillas distintas (referencia) | 6 | -0.01 (-0.23–0.24) |