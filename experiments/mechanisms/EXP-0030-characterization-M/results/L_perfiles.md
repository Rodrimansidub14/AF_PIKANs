# L. Probabilidad de mejora y perfiles de desempeño (post hoc, exploratorio; sin intervalos)

`P(X<Y)` = fracción de pares (una corrida de X, una de Y) donde X tiene menor l2re_u (empates ½); 0.5 = sin diferencia. Con n ≤ 5 solo toma valores en pasos de 1/(n·m): se lee como descripción. Perfiles: fracción de corridas con error ≤ tau.

| problema | métrica | comparación | P(X<Y) | n |
|---|---|---|---|---|
| converse | último paso | E frente a B | 0.52 | 5×5 |
| converse | último paso | M frente a B | 0.33 | 3×5 |
| converse | último paso | M frente a E | 0.20 | 3×5 |
| converse | robusta (18k-20k) | E frente a B | 0.68 | 5×5 |
| converse | robusta (18k-20k) | M frente a B | 0.20 | 3×5 |
| converse | robusta (18k-20k) | M frente a E | 0.00 | 3×5 |
| direct | último paso | E frente a B | 0.56 | 3×3 |
| direct | último paso | M frente a B | 0.78 | 3×3 |
| direct | último paso | M frente a E | 0.56 | 3×3 |
| direct | robusta (18k-20k) | E frente a B | 0.44 | 3×3 |
| direct | robusta (18k-20k) | M frente a B | 0.67 | 3×3 |
| direct | robusta (18k-20k) | M frente a E | 0.56 | 3×3 |

| problema | métrica | brazo | n | ≤0.05 | ≤0.10 | ≤0.20 | ≤0.30 | ≤0.50 |
|---|---|---|---|---|---|---|---|---|
| converse | último paso | B | 5 | 0.40 | 0.40 | 0.80 | 0.80 | 1.00 |
| converse | último paso | E | 5 | 0.40 | 0.60 | 0.80 | 0.80 | 0.80 |
| converse | último paso | M | 3 | 0.00 | 0.00 | 0.67 | 0.67 | 1.00 |
| converse | robusta (18k-20k) | B | 5 | 0.20 | 0.60 | 0.80 | 0.80 | 1.00 |
| converse | robusta (18k-20k) | E | 5 | 0.20 | 0.80 | 1.00 | 1.00 | 1.00 |
| converse | robusta (18k-20k) | M | 3 | 0.00 | 0.00 | 0.33 | 0.67 | 1.00 |
| direct | último paso | B | 3 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 |
| direct | último paso | E | 3 | 0.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| direct | último paso | M | 3 | 0.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| direct | robusta (18k-20k) | B | 3 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 |
| direct | robusta (18k-20k) | E | 3 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 |
| direct | robusta (18k-20k) | M | 3 | 0.00 | 0.00 | 0.67 | 1.00 | 1.00 |