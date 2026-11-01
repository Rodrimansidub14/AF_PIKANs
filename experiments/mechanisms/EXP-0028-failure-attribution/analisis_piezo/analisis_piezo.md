## 0. Error absoluto l2re_u al paso 20 000 por semilla (la razón con B engaña cuando B varía mucho)

| problema | brazo | semillas 28 / 29 / 30 | media | CV (desv/media) | referencia del paper (MLP) |
|---|---|---|---|---|---|
| converse | B | 0.367 / 0.048 / 0.180 | 0.198 | 0.81 | 0.107 |
| converse | F | 0.234 / 0.232 / 0.241 | 0.236 | 0.02 | 0.107 |
| converse | E | 0.050 / 0.077 / 0.033 | 0.053 | 0.42 | 0.107 |
| converse | J | 0.107 / 0.038 / 0.121 | 0.089 | 0.50 | 0.107 |
| direct | B | 0.180 / 0.240 / 0.204 | 0.208 | 0.14 | 0.117 |
| direct | F | 0.973 / 0.634 / 0.801 | 0.803 | 0.21 | 0.117 |
| direct | E | 0.129 / 0.198 / 0.253 | 0.193 | 0.32 | 0.117 |
| direct | J | 0.170 / 0.180 / 0.193 | 0.181 | 0.06 | 0.117 |

## 1. Error por campo (razón brazo/B, mediana de 3 semillas; por semilla entre paréntesis)

**converse**

| campo | F | E | J |
|---|---|---|---|
| l2re_u | **1.34** (0.64, 4.80, 1.34) | **0.18** (0.13, 1.59, 0.18) | **0.67** (0.29, 0.79, 0.67) |
| l2re_v | **1.40** (0.70, 8.00, 1.40) | **0.07** (0.07, 1.83, 0.06) | **0.59** (0.27, 0.59, 0.72) |
| l2re_phi | **0.97** (0.98, 0.97, 0.94) | **1.18** (1.25, 1.18, 1.11) | **1.00** (1.10, 0.88, 1.00) |
| sigma_xx nrms % | **1.04** (0.95, 1.04, 1.15) | **1.04** (1.03, 1.06, 1.04) | **1.01** (0.99, 1.07, 1.01) |
| D_y nrms % | **1.11** (1.11, 1.26, 1.07) | **0.93** (0.91, 0.93, 1.06) | **0.92** (0.92, 0.84, 0.94) |
| pérdida final | **1.25** (1.25, 1.06, 1.92) | **0.78** (0.78, 0.63, 0.90) | **0.87** (0.87, 0.82, 1.66) |

**direct**

| campo | F | E | J |
|---|---|---|---|
| l2re_u | **3.93** (5.39, 2.64, 3.93) | **0.82** (0.71, 0.82, 1.24) | **0.94** (0.94, 0.75, 0.95) |
| l2re_v | **3.51** (4.87, 2.47, 3.51) | **0.84** (0.65, 0.84, 1.21) | **0.89** (0.99, 0.77, 0.89) |
| l2re_phi | **3.26** (5.27, 2.43, 3.26) | **0.82** (0.82, 0.82, 1.12) | **0.98** (0.98, 0.77, 1.00) |
| sigma_xx nrms % | **0.70** (0.75, 0.52, 0.70) | **0.67** (0.64, 0.71, 0.67) | **0.97** (0.97, 0.51, 1.04) |
| D_y nrms % | **2.45** (2.68, 2.12, 2.45) | **1.08** (0.48, 1.08, 1.22) | **1.08** (1.30, 1.08, 0.77) |
| pérdida final | **4.08** (4.08, 0.62, 5.57) | **0.68** (0.68, 0.23, 0.92) | **0.86** (1.11, 0.21, 0.86) |

## 2. Dónde está la pérdida (parte de la pérdida de B al final de Adam, y razón brazo/B por grupo)

**converse**

| grupo | parte de la pérdida de B | F/B | E/B | J/B |
|---|---|---|---|---|
| constitutivas σ | 76 % | 1.44 | 0.73 | 0.93 |
| constitutivas D | 14 % | 0.90 | 1.33 | 1.01 |
| equilibrio y Gauss | 10 % | 0.13 | 0.06 | 0.08 |
| bordes mecánicos | 0 % | 0.43 | 0.49 | 1.50 |
| bordes eléctricos | 0 % | 0.51 | 0.12 | 0.55 |

**direct**

| grupo | parte de la pérdida de B | F/B | E/B | J/B |
|---|---|---|---|---|
| constitutivas σ | 57 % | 2.45 | 0.39 | 0.85 |
| constitutivas D | 2 % | 0.71 | 0.62 | 0.71 |
| equilibrio y Gauss | 41 % | 7.17 | 0.89 | 0.82 |
| bordes mecánicos | 0 % | – | – | – |
| bordes eléctricos | 0 % | 6.23 | 122.85 | 3.07 |

## 3. Estabilidad: saltos de pérdida (> 5× el mínimo previo, tras el paso 2 000), una cifra por semilla

| problema | B | F | E | J |
|---|---|---|---|---|
| converse | 0, 0, 2 | 0, 1, 0 | 1, 1, 2 | 2, 2, 1 |
| direct | 30, 38, 25 | 113, 99, 122 | 26, 36, 32 | 13, 18, 29 |

## 4. Curvatura por arista y lo que F hace sobre ella

Método: E guarda, en cada recálculo, la traza del bloque Gauss-Newton por arista y P = (B/m + 1e-3·I)^(-1/2), con m = traza/11. Con eso, B_e = m·(P⁻² − 1e-3·I). Con esa B_e se calcula la rigidez que produce F (λmax de P_F·B_e·P_F), frente a la de B (λmax de B_e) y a la de E (λmax de P_E·B_e·P_E). q = λmax(precondicionada)/λmax(B_e): q > 1 significa que el precondicionador *agrava* la dirección más rígida. Estados a los pasos 2 000, 10 000 y 20 000 (P de E del último recálculo anterior).

| problema | paso | dispersión de trazas (log10 p95/p5) | λmax(B_e), log10 [p5, mediana, p95] | aristas con q_F > 1 / > 10 / > 100 | q_F mediana [p5, p95] | q_E mediana [p5, p95] |
|---|---|---|---|---|---|---|
| converse | 2000 | 3.28 | 0.3, 2.4, 3.5 | 65 % / 1 % / 0 % | 1.49 [0.58, 3.88] | 0.122 [0.092, 0.246] |
| converse | 10000 | 3.93 | 0.3, 2.3, 4.1 | 55 % / 1 % / 0 % | 1.28 [0.51, 4.34] | 0.110 [0.092, 0.225] |
| converse | 20000 | 3.95 | 0.3, 2.0, 4.1 | 47 % / 1 % / 0 % | 0.86 [0.51, 4.55] | 0.106 [0.091, 0.211] |
| direct | 2000 | 3.75 | -1.1, 1.5, 2.6 | 28 % / 0 % / 0 % | 0.65 [0.48, 3.37] | 0.110 [0.093, 0.262] |
| direct | 10000 | 4.37 | -1.1, 2.0, 2.8 | 26 % / 0 % / 0 % | 0.65 [0.54, 3.60] | 0.096 [0.092, 0.275] |
| direct | 20000 | 4.40 | -1.1, 1.9, 2.8 | 25 % / 0 % / 0 % | 0.65 [0.59, 3.87] | 0.093 [0.091, 0.303] |

## 5. Condicionamiento dentro de cada arista, regularizado igual que en E (cond de B/m + 1e-3·I; sin la regularización los bloques casi singulares dan cifras sin sentido)

E hace cond = 1 por construcción (P = (B/m + 1e-3·I)^(-1/2)); lo informativo es cuánto reduce F, con su P fija, el condicionamiento real de cada bloque. Mediana [p5, p95] sobre aristas y semillas.

| problema | paso | cond de B/m + 1e-3·I | cond tras F | reducción mediana (cond sin / cond con F) | aristas donde F empeora el cond |
|---|---|---|---|---|---|
| converse | 2000 | 8.2e+03 [3.3e+03, 1.1e+04] | 1.7e+04 [1.3e+02, 9.7e+04] | 0.5x | 82 % |
| converse | 10000 | 9.1e+03 [3.8e+03, 1.1e+04] | 1.8e+04 [3.5e+02, 9.4e+04] | 0.5x | 84 % |
| converse | 20000 | 9.5e+03 [4.0e+03, 1.1e+04] | 1.7e+04 [3.3e+02, 1.0e+05] | 0.6x | 84 % |
| direct | 2000 | 9.1e+03 [3.2e+03, 1.1e+04] | 1.1e+04 [4.2e+01, 6.1e+04] | 0.8x | 63 % |
| direct | 10000 | 1.0e+04 [3.0e+03, 1.1e+04] | 1.7e+04 [4.4e+01, 6.6e+04] | 0.6x | 86 % |
| direct | 20000 | 1.1e+04 [2.5e+03, 1.1e+04] | 1.9e+04 [4.7e+01, 8.9e+04] | 0.6x | 88 % |
