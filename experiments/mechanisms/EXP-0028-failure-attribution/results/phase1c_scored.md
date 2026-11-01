# EXP-0028 Fase 1c — NS2d-CG (50 corridas)

## Metrica: PRIMARIA: L2RE aplanado (`l2re_flat`)


## NS2d-CG (RBF16, receta indicada en el JSON de cada corrida; este bloque: t1 = Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.830 (10/10, max 0.93) | sirve | 0.676 (10/10, max 0.78) | sirve | Adam y L-BFGS (se mantiene) |
| J | 0.974 (6/10, max 1.19) | indiferente | 1.002 (5/10, max 1.18) | indiferente | ninguna |
| E | 0.792 (10/10, max 0.94) | sirve | 0.641 (10/10, max 0.82) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.736 (10/10, max 0.81) | sirve | 0.561 (10/10, max 0.68) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.939 (8/10) | False | False | 0.997 |
| J frente a B (¿basta reescalar?) | 0.974 (6/10) | False | False | 1.002 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.805 (10/10) | True | False | 0.654 |
| I frente a E (¿importa acoplar aristas?) | 0.923 (9/10) | False | False | 0.844 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.1861379029977619, "E_logB": -0.23400738630499918, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 144.71963270900005}, "F": {"n": 10, "mediana_s": 149.87522558249987}, "J": {"n": 10, "mediana_s": 166.90608442899992}, "E": {"n": 10, "mediana_s": 167.21433733750007}, "I": {"n": 10, "mediana_s": 169.62396735100003}}


## Metrica: media de componentes (`l2re_mean_components`)


## NS2d-CG (RBF16, receta indicada en el JSON de cada corrida; este bloque: t1 = Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.887 (10/10, max 0.97) | sirve | 0.738 (10/10, max 0.86) | sirve | Adam y L-BFGS (se mantiene) |
| J | 0.986 (6/10, max 1.11) | indiferente | 0.986 (6/10, max 1.12) | indiferente | ninguna |
| E | 0.840 (10/10, max 0.97) | sirve | 0.696 (10/10, max 0.88) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.804 (10/10, max 0.85) | sirve | 0.655 (10/10, max 0.78) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.943 (8/10) | False | False | 1.005 |
| J frente a B (¿basta reescalar?) | 0.986 (6/10) | False | False | 0.986 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.868 (10/10) | True | False | 0.728 |
| I frente a E (¿importa acoplar aristas?) | 0.939 (9/10) | False | False | 0.893 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.11985609328872632, "E_logB": -0.17468046850507932, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 144.71963270900005}, "F": {"n": 10, "mediana_s": 149.87522558249987}, "J": {"n": 10, "mediana_s": 166.90608442899992}, "E": {"n": 10, "mediana_s": 167.21433733750007}, "I": {"n": 10, "mediana_s": 169.62396735100003}}


## Metrica: componente u (`l2re_u`)


## NS2d-CG (RBF16, receta indicada en el JSON de cada corrida; este bloque: t1 = Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.749 (10/10, max 0.89) | sirve | 0.618 (10/10, max 0.72) | sirve | Adam y L-BFGS (se mantiene) |
| J | 0.970 (6/10, max 1.30) | indiferente | 0.995 (6/10, max 1.23) | indiferente | ninguna |
| E | 0.689 (10/10, max 0.88) | sirve | 0.580 (10/10, max 0.76) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.578 (10/10, max 0.68) | sirve | 0.466 (10/10, max 0.56) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.923 (10/10) | False | False | 1.001 |
| J frente a B (¿basta reescalar?) | 0.970 (6/10) | False | False | 0.995 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.710 (10/10) | True | False | 0.587 |
| I frente a E (¿importa acoplar aristas?) | 0.841 (10/10) | True | False | 0.777 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.28886131603096266, "E_logB": -0.372066347090195, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 144.71963270900005}, "F": {"n": 10, "mediana_s": 149.87522558249987}, "J": {"n": 10, "mediana_s": 166.90608442899992}, "E": {"n": 10, "mediana_s": 167.21433733750007}, "I": {"n": 10, "mediana_s": 169.62396735100003}}


## Metrica: componente v (`l2re_v`)


## NS2d-CG (RBF16, receta indicada en el JSON de cada corrida; este bloque: t1 = Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.871 (10/10, max 0.95) | sirve | 0.721 (10/10, max 0.88) | sirve | Adam y L-BFGS (se mantiene) |
| J | 0.980 (7/10, max 1.11) | indiferente | 0.985 (6/10, max 1.10) | indiferente | ninguna |
| E | 0.814 (10/10, max 0.96) | sirve | 0.701 (10/10, max 0.89) | sirve | Adam y L-BFGS (se mantiene) |
| I | 0.749 (10/10, max 0.80) | sirve | 0.643 (10/10, max 0.75) | sirve | Adam y L-BFGS (se mantiene) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.938 (9/10) | False | False | 1.010 |
| J frente a B (¿basta reescalar?) | 0.980 (7/10) | False | False | 0.985 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.843 (10/10) | True | False | 0.729 |
| I frente a E (¿importa acoplar aristas?) | 0.899 (10/10) | True | False | 0.886 |

Atribucion: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.13829876861527557, "E_logB": -0.2055863409800807, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 144.71963270900005}, "F": {"n": 10, "mediana_s": 149.87522558249987}, "J": {"n": 10, "mediana_s": 166.90608442899992}, "E": {"n": 10, "mediana_s": 167.21433733750007}, "I": {"n": 10, "mediana_s": 169.62396735100003}}


## Metrica: componente p (`l2re_p`)


## NS2d-CG (RBF16, receta indicada en el JSON de cada corrida; este bloque: t1 = Adam 2000 + L-BFGS 500 it.) (50 corridas)

| brazo | mediana e_X/e_B final (mejora k/n, max) | clase (final) | mediana en Adam | clase (Adam) | fase del efecto |
|---|---|---|---|---|---|
| F | 0.929 (9/10, max 1.00) | indiferente | 0.766 (10/10, max 0.87) | sirve | Adam (sirve); se pierde o cambia en L-BFGS (indiferente) |
| J | 1.002 (5/10, max 1.08) | indiferente | 0.994 (5/10, max 1.12) | indiferente | ninguna |
| E | 0.902 (9/10, max 1.00) | indiferente | 0.727 (10/10, max 0.90) | sirve | Adam (sirve); se pierde o cambia en L-BFGS (indiferente) |
| I | 0.911 (10/10, max 0.94) | indiferente | 0.687 (10/10, max 0.87) | sirve | Adam (sirve); se pierde o cambia en L-BFGS (indiferente) |

| contraste | mediana final (A mejor k/n) | respaldado | contrario | mediana Adam |
|---|---|---|---|---|
| D5: E frente a F (¿residuo o solo la base?) | 0.965 (8/10) | False | False | 0.998 |
| J frente a B (¿basta reescalar?) | 1.002 (5/10) | False | False | 0.994 |
| E frente a J (¿importan los terminos fuera de la diagonal?) | 0.923 (10/10) | False | False | 0.768 |
| I frente a E (¿importa acoplar aristas?) | 0.989 (7/10) | False | False | 0.933 |

Atribucion: {"E": "indiferente", "H-ATTR-A": {"aplica": false, "motivo": "E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

Reloj (mediana s por corrida): {"B": {"n": 10, "mediana_s": 144.71963270900005}, "F": {"n": 10, "mediana_s": 149.87522558249987}, "J": {"n": 10, "mediana_s": 166.90608442899992}, "E": {"n": 10, "mediana_s": 167.21433733750007}, "I": {"n": 10, "mediana_s": 169.62396735100003}}

