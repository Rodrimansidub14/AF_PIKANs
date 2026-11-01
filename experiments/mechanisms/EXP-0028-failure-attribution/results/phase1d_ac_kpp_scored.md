# EXP-0028 Fase 1d — confirmacion con receta completa (criterio uniforme, n = 5, por problema)

Etiqueta de la fase 1 = `results/phase1_scored.md`; veredicto = sostenido / se pierde / cambia de signo (aparece / sin efecto cuando la fase 1 no era sirve/daña). Fase 1: n = 10 (F, E) o 5 (C) a receta corta; 1d: n = 5 a receta completa.

## Allen-Cahn d=1e-3 (20 corridas) — receta: Adam 40 000 f32 + L-BFGS f32 (tope 300 s / 15 000 it.)

B a receta completa: mediana fin de Adam 2.802e-02, tras L-BFGS 1.526e-03 (n = 5); linea base congelada 2.838e-02 / 2.686e-03 -> razon final 0.57x (informativo).

| brazo | fase 1 (tras L-BFGS) | 1d (tras L-BFGS) | **veredicto** | fase 1 (Adam) | 1d (Adam) | veredicto Adam | L-BFGS 1d | fase del efecto 1d |
|---|---|---|---|---|---|---|---|---|
| C | 0.783 (5/5, max 0.85) sirve | 1.007 (2/5, max 1.73) dependiente de semilla | **se pierde — neutro: la mediana cae en la banda [0.90, 1.10] (dependiente de semilla)** | 0.786 (4/5, max 1.02) sirve | 0.794 (4/5, max 1.09) sirve | sostenido | {"lbfgs_plateau": 5} | Adam (sirve); se pierde o cambia en L-BFGS (dependiente de semilla) |
| E | 0.162 (10/10, max 0.60) sirve | 0.389 (4/5, max 3.14) dependiente de semilla | **se pierde — debilitado: la mediana conserva la direccion (dependiente de semilla)** | 0.283 (10/10, max 0.47) sirve | 0.212 (5/5, max 0.36) sirve | sostenido | {"lbfgs_plateau": 5} | Adam (sirve); se pierde o cambia en L-BFGS (dependiente de semilla) |
| F | 0.352 (10/10, max 0.51) sirve | 0.812 (3/5, max 3.95) dependiente de semilla | **se pierde — debilitado: la mediana conserva la direccion (dependiente de semilla)** | 0.292 (10/10, max 0.46) sirve | 0.281 (5/5, max 0.50) sirve | sostenido | {"lbfgs_plateau": 5} | Adam (sirve); se pierde o cambia en L-BFGS (dependiente de semilla) |

D5: E frente a F (¿residuo o solo la base?): 1d mediana 0.906 (E mejor 3/5; respaldado=False, contrario=False); fase 1: 0.425 (9/10).

Atribucion 1d: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": true, "F_logB": -0.20797617171380225, "E_logB": -0.9436513568029327, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}
Atribucion fase 1: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -1.045539380285441, "E_logB": -1.8192900623470853, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Burgers RBF48 (0 corridas) — receta: 16 384 pts, Adam 24 000 f32 + L-BFGS f64 (tope 1 800 s)

| brazo | fase 1 (tras L-BFGS) | 1d (tras L-BFGS) | **veredicto** | fase 1 (Adam) | 1d (Adam) | veredicto Adam | L-BFGS 1d | fase del efecto 1d |
|---|---|---|---|---|---|---|---|---|

Atribucion 1d: {"E": "sin datos"}
Atribucion fase 1: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": true, "F_logB": -1.257058606910546, "E_logB": -1.4644113168417525, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Fisher-KPP (15 corridas) — receta: receta de Allen-Cahn (heredada; sin linea base propia)

B a receta completa: mediana fin de Adam 1.325e-01, tras L-BFGS 8.630e-02 (n = 5); sin linea base congelada para este problema.

| brazo | fase 1 (tras L-BFGS) | 1d (tras L-BFGS) | **veredicto** | fase 1 (Adam) | 1d (Adam) | veredicto Adam | L-BFGS 1d | fase del efecto 1d |
|---|---|---|---|---|---|---|---|---|
| E | 0.767 (6/10, max 21.79) dependiente de semilla | 0.119 (5/5, max 0.57) sirve | **aparece — fase 1: dependiente de semilla** | 0.762 (7/10, max 16.47) dependiente de semilla | 0.077 (5/5, max 0.75) sirve | aparece | {"lbfgs_plateau": 5} | Adam y L-BFGS (se mantiene) |
| F | 1.835 (4/10, max 10.47) dependiente de semilla | 0.640 (3/5, max 2.37) dependiente de semilla | **cambia de signo — dependiente de semilla -> dependiente de semilla, la mediana pasa de empeora a mejora (ninguna fue sirve/daña)** | 1.094 (5/10, max 6.86) dependiente de semilla | 0.429 (4/5, max 3.15) dependiente de semilla | sin efecto | {"lbfgs_plateau": 5} | ninguna |

D5: E frente a F (¿residuo o solo la base?): 1d mediana 0.242 (E mejor 5/5; respaldado=True, contrario=False); fase 1: 0.772 (6/10).

Atribucion 1d: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.4462501079021234, "E_logB": -2.1309966938612916, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}
Atribucion fase 1: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": false, "motivo": "E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Resumen para predictions.md (lo confirmado pesa mas que lo corto)

| problema | brazo | fase 1 | 1d | veredicto |
|---|---|---|---|---|
| Allen-Cahn d=1e-3 | C | sirve | dependiente de semilla | se pierde |
| Allen-Cahn d=1e-3 | E | sirve | dependiente de semilla | se pierde |
| Allen-Cahn d=1e-3 | F | sirve | dependiente de semilla | se pierde |
| Fisher-KPP | E | dependiente de semilla | sirve | aparece |
| Fisher-KPP | F | dependiente de semilla | dependiente de semilla | cambia de signo |