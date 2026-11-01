# EXP-0028 Fase 1 — analisis (criterio uniforme, por problema)

## Allen-Cahn d=1e-3 (70 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.352 (10/10, max 0.51) | sirve | 0.292 (10/10, max 0.46) | sirve | Adam y L-BFGS (se mantiene) |
| J | 0.937 (6/10, max 1.33) | indiferente | 0.995 (5/10, max 1.37) | indiferente | ninguna |
| E | 0.162 (10/10, max 0.60) | sirve | 0.283 (10/10, max 0.47) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.144 (6/10, max 11.62) | dependiente de semilla | 0.221 (6/10, max 5.73) | dependiente de semilla | ninguna |
| S | 1.392 (1/5, max 1.69) | daña | 1.741 (0/5, max 3.02) | daña | Adam y L-BFGS (se mantiene) |
| R | 0.793 (5/5, max 0.89) | sirve | 0.922 (5/5, max 0.98) | indiferente | aparece en L-BFGS (sirve); Adam: indiferente |
| C | 0.783 (5/5, max 0.85) | sirve | 0.786 (4/5, max 1.02) | sirve | Adam y L-BFGS (se mantiene) |
| W (oraculo) | 0.937 (4/5, max 1.01) | indiferente | 1.000 (0/5, max 1.00) | indiferente | ninguna |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.425 (9/10) | True | False | 0.919 |
| J frente a B (¿basta reescalar?) | 0.937 (6/10) | False | False | 0.995 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.169 (10/10) | True | False | 0.300 |
| I frente a E (¿importa acoplar aristas?) | 1.465 (4/10) | False | False | 1.167 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -1.045539380285441, "E_logB": -1.8192900623470853, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 95.58019585200009}, "F": {"n": 10, "mediana_s": 98.61766072249975}, "J": {"n": 10, "mediana_s": 102.78933068749984}, "E": {"n": 10, "mediana_s": 102.58019753899993}, "I": {"n": 10, "mediana_s": 103.54561403750006}, "S": {"n": 5, "mediana_s": 105.42045525499998}, "R": {"n": 5, "mediana_s": 104.03212121399974}, "C": {"n": 5, "mediana_s": 96.28581149699949}, "W": {"n": 5, "mediana_s": 95.7329505309998}}

## Allen-Cahn d=2e-4 (70 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.405 (10/10, max 0.72) | sirve | 0.467 (10/10, max 0.76) | sirve | Adam y L-BFGS (se mantiene) |
| J | 0.839 (8/10, max 1.17) | sirve | 0.959 (6/10, max 1.33) | indiferente | aparece en L-BFGS (sirve); Adam: indiferente |
| E | 0.243 (10/10, max 0.35) | sirve | 0.404 (10/10, max 0.64) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.142 (9/10, max 10.66) | dependiente de semilla | 0.183 (9/10, max 5.84) | dependiente de semilla | ninguna |
| S | 1.049 (1/5, max 1.26) | indiferente | 1.351 (0/5, max 2.43) | daña | Adam (daña); se pierde o cambia en L-BFGS (indiferente) |
| R | 0.802 (5/5, max 0.88) | sirve | 1.046 (0/5, max 1.15) | indiferente | aparece en L-BFGS (sirve); Adam: indiferente |
| C | 0.671 (5/5, max 0.82) | sirve | 0.726 (5/5, max 0.94) | sirve | Adam y L-BFGS (se mantiene) |
| W (oraculo) | 0.949 (5/5, max 0.99) | indiferente | 1.000 (0/5, max 1.00) | indiferente | ninguna |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.501 (10/10) | True | False | 0.852 |
| J frente a B (¿basta reescalar?) | 0.839 (8/10) | True | False | 0.959 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.295 (10/10) | True | False | 0.428 |
| I frente a E (¿importa acoplar aristas?) | 0.814 (5/10) | False | False | 0.380 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.9053357408064855, "E_logB": -1.4150967001131938, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 95.94705430250008}, "F": {"n": 10, "mediana_s": 98.5745514170003}, "J": {"n": 10, "mediana_s": 103.17979881500014}, "E": {"n": 10, "mediana_s": 103.21711182799982}, "I": {"n": 10, "mediana_s": 102.35577722499988}, "S": {"n": 5, "mediana_s": 104.75843565800005}, "R": {"n": 5, "mediana_s": 103.27008469899988}, "C": {"n": 5, "mediana_s": 96.22897088700006}, "W": {"n": 5, "mediana_s": 95.70697520800013}}

## Burgers RBF48 (65 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.285 (10/10, max 0.80) | sirve | 0.625 (8/10, max 2.82) | dependiente de semilla | aparece en L-BFGS (sirve); Adam: dependiente de semilla |
| J | 2.937 (3/10, max 10.80) | daña | 1.929 (4/10, max 8.21) | dependiente de semilla | aparece en L-BFGS (daña); Adam: dependiente de semilla |
| E | 0.240 (9/10, max 2.37) | dependiente de semilla | 1.144 (4/10, max 5.22) | dependiente de semilla | ninguna |
| I | 47.861 (0/9, max 193.55) | daña | 6.237 (1/9, max 21.22) | daña | Adam y L-BFGS (se mantiene) |
| S | 12.885 (0/4, max 29.37) | daña | 5.829 (0/4, max 10.26) | daña | Adam y L-BFGS (se mantiene) |
| R | 76.994 (0/4, max 157.71) | daña | 7.570 (0/4, max 12.71) | daña | Adam y L-BFGS (se mantiene) |
| C | 4.432 (0/4, max 13.89) | daña | 0.943 (2/4, max 1.93) | dependiente de semilla | aparece en L-BFGS (daña); Adam: dependiente de semilla |
| W (oraculo) | 1.057 (2/4, max 1.55) | indiferente | 1.000 (0/4, max 1.00) | indiferente | ninguna |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.715 (7/10) | True | False | 1.560 |
| J frente a B (¿basta reescalar?) | 2.937 (3/10) | False | True | 1.929 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.080 (9/10) | True | False | 0.866 |
| I frente a E (¿importa acoplar aristas?) | 214.463 (0/9) | False | True | 5.677 |

Atribucion: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": true, "F_logB": -1.257058606910546, "E_logB": -1.4644113168417525, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 106.18914034449983}, "F": {"n": 10, "mediana_s": 106.22132926350002}, "J": {"n": 10, "mediana_s": 116.06737287099989}, "E": {"n": 10, "mediana_s": 116.92440140749989}, "I": {"n": 9, "mediana_s": 121.80328408900004}, "S": {"n": 4, "mediana_s": 109.61352506900016}, "R": {"n": 4, "mediana_s": 107.59858618249973}, "C": {"n": 4, "mediana_s": 106.58718285249984}, "W": {"n": 4, "mediana_s": 106.2343543595}}

## Fisher-KPP (70 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 1.835 (4/10, max 10.47) | dependiente de semilla | 1.094 (5/10, max 6.86) | dependiente de semilla | ninguna |
| J | 2.379 (3/10, max 23.63) | daña | 2.716 (2/10, max 23.32) | daña | Adam y L-BFGS (se mantiene) |
| E | 0.767 (6/10, max 21.79) | dependiente de semilla | 0.762 (7/10, max 16.47) | dependiente de semilla | ninguna |
| I | 1.390 (5/10, max 13.65) | dependiente de semilla | 1.325 (5/10, max 16.53) | dependiente de semilla | ninguna |
| S | 3.188 (0/5, max 38.54) | daña | 2.961 (0/5, max 39.01) | daña | Adam y L-BFGS (se mantiene) |
| R | 2.561 (1/5, max 33.29) | daña | 1.165 (2/5, max 32.76) | dependiente de semilla | aparece en L-BFGS (daña); Adam: dependiente de semilla |
| C | 1.324 (1/5, max 2.78) | daña | 1.148 (1/5, max 3.14) | daña | Adam y L-BFGS (se mantiene) |
| W (oraculo) | 0.987 (3/5, max 1.21) | indiferente | 1.000 (0/5, max 1.00) | indiferente | ninguna |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.772 (6/10) | False | False | 0.819 |
| J frente a B (¿basta reescalar?) | 2.379 (3/10) | False | True | 2.716 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.507 (10/10) | True | False | 0.321 |
| I frente a E (¿importa acoplar aristas?) | 1.101 (5/10) | False | False | 1.371 |

Atribucion: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": false, "motivo": "E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 99.30841344600026}, "F": {"n": 10, "mediana_s": 103.06261658750009}, "J": {"n": 10, "mediana_s": 107.59441504200015}, "E": {"n": 10, "mediana_s": 91.51794814050004}, "I": {"n": 10, "mediana_s": 105.65465207850025}, "S": {"n": 5, "mediana_s": 111.18076648700003}, "R": {"n": 5, "mediana_s": 107.24564208699985}, "C": {"n": 5, "mediana_s": 102.12871797499974}, "W": {"n": 5, "mediana_s": 100.78192960099986}}
