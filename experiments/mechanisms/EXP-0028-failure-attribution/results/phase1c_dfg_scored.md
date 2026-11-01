# EXP-0028 Fase 1c — DFG 2D-1 Re20 (50 corridas)

## Metrica: PRIMARIA: L2RE aplanado (u, v, p) (`l2re_flat`)


## DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.567 (10/10, max 0.80) | sirve | 0.800 (10/10, max 0.84) | sirve | Adam y L-BFGS (se mantiene) |
| J | 2.068 (0/10, max 2.81) | daña | 1.019 (2/10, max 1.07) | indiferente | aparece en L-BFGS (daña); Adam: indiferente |
| E | 0.815 (10/10, max 0.95) | sirve | 0.834 (10/10, max 0.91) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.387 (10/10, max 0.73) | sirve | 0.756 (10/10, max 0.85) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 1.191 (1/10) | False | True | 1.037 |
| J frente a B (¿basta reescalar?) | 2.068 (0/10) | False | True | 1.019 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.394 (10/10) | True | False | 0.814 |
| I frente a E (¿importa acoplar aristas?) | 0.483 (9/10) | True | False | 0.929 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.5677302128825801, "E_logB": -0.20462422842808906, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 215.75938269250037}, "F": {"n": 10, "mediana_s": 223.28302774250005}, "J": {"n": 10, "mediana_s": 248.07403752699997}, "E": {"n": 10, "mediana_s": 242.52391772900023}, "I": {"n": 10, "mediana_s": 250.36322547200007}}


## Metrica: QoI: error relativo de Cd (`Cd_relerr`)


## DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.467 (10/10, max 0.63) | sirve | 0.689 (10/10, max 0.75) | sirve | Adam y L-BFGS (se mantiene) |
| J | 3.111 (0/10, max 4.08) | daña | 1.048 (0/10, max 1.06) | indiferente | aparece en L-BFGS (daña); Adam: indiferente |
| E | 0.499 (10/10, max 0.70) | sirve | 0.762 (10/10, max 0.89) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.332 (10/10, max 0.42) | sirve | 0.671 (10/10, max 0.80) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 1.024 (5/10) | False | False | 1.122 |
| J frente a B (¿basta reescalar?) | 3.111 (0/10) | False | True | 1.048 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.174 (10/10) | True | False | 0.739 |
| I frente a E (¿importa acoplar aristas?) | 0.581 (9/10) | True | False | 0.892 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.7615748447296458, "E_logB": -0.6964443774075881, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 215.75938269250037}, "F": {"n": 10, "mediana_s": 223.28302774250005}, "J": {"n": 10, "mediana_s": 248.07403752699997}, "E": {"n": 10, "mediana_s": 242.52391772900023}, "I": {"n": 10, "mediana_s": 250.36322547200007}}


## Metrica: QoI: error relativo de dp (`dp_relerr`)


## DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.512 (10/10, max 0.81) | sirve | 0.736 (10/10, max 0.80) | sirve | Adam y L-BFGS (se mantiene) |
| J | 2.926 (0/10, max 5.03) | daña | 1.031 (0/10, max 1.05) | indiferente | aparece en L-BFGS (daña); Adam: indiferente |
| E | 0.572 (10/10, max 0.67) | sirve | 0.806 (10/10, max 0.91) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.316 (10/10, max 0.60) | sirve | 0.724 (10/10, max 0.84) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 1.053 (5/10) | False | False | 1.101 |
| J frente a B (¿basta reescalar?) | 2.926 (0/10) | False | True | 1.031 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.202 (10/10) | True | False | 0.787 |
| I frente a E (¿importa acoplar aristas?) | 0.579 (10/10) | True | False | 0.890 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.6690793496231245, "E_logB": -0.5623337171105722, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 215.75938269250037}, "F": {"n": 10, "mediana_s": 223.28302774250005}, "J": {"n": 10, "mediana_s": 248.07403752699997}, "E": {"n": 10, "mediana_s": 242.52391772900023}, "I": {"n": 10, "mediana_s": 250.36322547200007}}


## Metrica: QoI: error absoluto de Cl (`Cl_abserr`)


## DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.731 (8/10, max 110.68) | dependiente de semilla | 1.621 (2/10, max 22.63) | daña | Adam (daña); se pierde o cambia en L-BFGS (dependiente de semilla) |
| J | 1.503 (3/10, max 85.47) | daña | 0.771 (6/10, max 9.13) | dependiente de semilla | aparece en L-BFGS (daña); Adam: dependiente de semilla |
| E | 0.353 (7/10, max 12.47) | dependiente de semilla | 2.720 (3/10, max 49.75) | daña | Adam (daña); se pierde o cambia en L-BFGS (dependiente de semilla) |
| I | 0.405 (8/10, max 69.26) | dependiente de semilla | 1.483 (1/10, max 56.07) | daña | Adam (daña); se pierde o cambia en L-BFGS (dependiente de semilla) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.387 (7/10) | True | False | 1.541 |
| J frente a B (¿basta reescalar?) | 1.503 (3/10) | False | True | 0.771 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.166 (9/10) | True | False | 2.856 |
| I frente a E (¿importa acoplar aristas?) | 2.363 (4/10) | False | False | 1.203 |

Atribucion: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": true, "F_logB": -0.3150214322266887, "E_logB": -1.0866643888803846, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 215.75938269250037}, "F": {"n": 10, "mediana_s": 223.28302774250005}, "J": {"n": 10, "mediana_s": 248.07403752699997}, "E": {"n": 10, "mediana_s": 242.52391772900023}, "I": {"n": 10, "mediana_s": 250.36322547200007}}


## Metrica: media de componentes (`l2re_mean_components`)


## DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.594 (10/10, max 0.97) | sirve | 0.817 (10/10, max 0.87) | sirve | Adam y L-BFGS (se mantiene) |
| J | 1.658 (0/10, max 2.09) | daña | 1.032 (1/10, max 1.06) | indiferente | aparece en L-BFGS (daña); Adam: indiferente |
| E | 0.868 (9/10, max 1.10) | sirve | 0.858 (10/10, max 0.93) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.428 (10/10, max 0.94) | sirve | 0.787 (10/10, max 0.89) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 1.319 (2/10) | False | True | 1.044 |
| J frente a B (¿basta reescalar?) | 1.658 (0/10) | False | True | 1.032 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.544 (10/10) | True | False | 0.832 |
| I frente a E (¿importa acoplar aristas?) | 0.485 (8/10) | True | False | 0.932 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.5206856287412197, "E_logB": -0.14242446591242078, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 215.75938269250037}, "F": {"n": 10, "mediana_s": 223.28302774250005}, "J": {"n": 10, "mediana_s": 248.07403752699997}, "E": {"n": 10, "mediana_s": 242.52391772900023}, "I": {"n": 10, "mediana_s": 250.36322547200007}}


## Metrica: componente u (`l2re_u`)


## DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.555 (10/10, max 0.79) | sirve | 0.797 (10/10, max 0.83) | sirve | Adam y L-BFGS (se mantiene) |
| J | 2.137 (0/10, max 2.95) | daña | 1.019 (2/10, max 1.08) | indiferente | aparece en L-BFGS (daña); Adam: indiferente |
| E | 0.795 (10/10, max 0.90) | sirve | 0.831 (10/10, max 0.91) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.374 (10/10, max 0.69) | sirve | 0.751 (10/10, max 0.84) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 1.157 (1/10) | False | True | 1.038 |
| J frente a B (¿basta reescalar?) | 2.137 (0/10) | False | True | 1.019 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.370 (10/10) | True | False | 0.811 |
| I frente a E (¿importa acoplar aristas?) | 0.485 (9/10) | True | False | 0.928 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.5896225809774438, "E_logB": -0.22969199529368178, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 215.75938269250037}, "F": {"n": 10, "mediana_s": 223.28302774250005}, "J": {"n": 10, "mediana_s": 248.07403752699997}, "E": {"n": 10, "mediana_s": 242.52391772900023}, "I": {"n": 10, "mediana_s": 250.36322547200007}}


## Metrica: componente v (`l2re_v`)


## DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.528 (10/10, max 0.70) | sirve | 0.660 (10/10, max 0.79) | sirve | Adam y L-BFGS (se mantiene) |
| J | 2.087 (0/10, max 2.88) | daña | 1.030 (2/10, max 1.07) | indiferente | aparece en L-BFGS (daña); Adam: indiferente |
| E | 0.563 (10/10, max 0.78) | sirve | 0.755 (10/10, max 0.84) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.369 (10/10, max 0.59) | sirve | 0.632 (10/10, max 0.77) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 1.030 (4/10) | False | False | 1.081 |
| J frente a B (¿basta reescalar?) | 2.087 (0/10) | False | True | 1.030 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.282 (10/10) | True | False | 0.733 |
| I frente a E (¿importa acoplar aristas?) | 0.614 (9/10) | True | False | 0.860 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.638587087835936, "E_logB": -0.5753200512635638, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 215.75938269250037}, "F": {"n": 10, "mediana_s": 223.28302774250005}, "J": {"n": 10, "mediana_s": 248.07403752699997}, "E": {"n": 10, "mediana_s": 242.52391772900023}, "I": {"n": 10, "mediana_s": 250.36322547200007}}


## Metrica: componente p (`l2re_p`)


## DFG 2D-1 Re20 (RBF16, Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.627 (9/10, max 1.14) | sirve | 0.917 (10/10, max 0.98) | indiferente | aparece en L-BFGS (sirve); Adam: indiferente |
| J | 1.321 (0/10, max 1.71) | daña | 1.032 (2/10, max 1.07) | indiferente | aparece en L-BFGS (daña); Adam: indiferente |
| E | 0.976 (6/10, max 1.36) | indiferente | 0.945 (9/10, max 1.00) | indiferente | ninguna |
| I | 0.459 (9/10, max 1.13) | sirve | 0.906 (10/10, max 1.00) | indiferente | aparece en L-BFGS (sirve); Adam: indiferente |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 1.462 (2/10) | False | True | 1.031 |
| J frente a B (¿basta reescalar?) | 1.321 (0/10) | False | True | 1.032 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.732 (10/10) | True | False | 0.914 |
| I frente a E (¿importa acoplar aristas?) | 0.468 (8/10) | True | False | 0.956 |

Atribucion: {"E": "indiferente", "H-ATTR-A": {"aplica": false, "motivo": "E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 215.75938269250037}, "F": {"n": 10, "mediana_s": 223.28302774250005}, "J": {"n": 10, "mediana_s": 248.07403752699997}, "E": {"n": 10, "mediana_s": 242.52391772900023}, "I": {"n": 10, "mediana_s": 250.36322547200007}}

