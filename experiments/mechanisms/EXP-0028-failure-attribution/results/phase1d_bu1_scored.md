# EXP-0028 Fase 1d — confirmacion con receta completa (criterio uniforme, n = 5, por problema)

Etiqueta de la fase 1 = `results/phase1_scored.md`; veredicto = sostenido / se pierde / cambia de signo (aparece / sin efecto cuando la fase 1 no era sirve/daña). Fase 1: n = 10 (F, E) o 5 (C) a receta corta; 1d: n = 5 a receta completa.

## Allen-Cahn d=1e-3 (0 corridas) — receta: Adam 40 000 f32 + L-BFGS f32 (tope 300 s / 15 000 it.)

| brazo | fase 1 (tras L-BFGS) | 1d (tras L-BFGS) | **veredicto** | fase 1 (Adam) | 1d (Adam) | veredicto Adam | L-BFGS 1d | fase del efecto 1d |
|---|---|---|---|---|---|---|---|---|

Atribucion 1d: {"E": "sin datos"}
Atribucion fase 1: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -1.045539380285441, "E_logB": -1.8192900623470853, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Burgers RBF48 (6 corridas) — receta: 16 384 pts, Adam 24 000 f32 + L-BFGS f64 (tope 1 800 s)

B a receta completa: mediana fin de Adam 5.601e-03, tras L-BFGS 1.103e-04 (n = 2); linea base congelada 5.415e-03 / 7.165e-05 -> razon final 1.54x (informativo).

| brazo | fase 1 (tras L-BFGS) | 1d (tras L-BFGS) | **veredicto** | fase 1 (Adam) | 1d (Adam) | veredicto Adam | L-BFGS 1d | fase del efecto 1d |
|---|---|---|---|---|---|---|---|---|
| E | 0.240 (9/10, max 2.37) dependiente de semilla | 1.369 (1/2, max 2.16) dependiente de semilla | **sin veredicto — n = 2 < 4** | 1.144 (4/10, max 5.22) dependiente de semilla | 0.435 (2/2, max 0.47) sirve | sin veredicto | {"lbfgs_time_cap": 2} | Adam (sirve); se pierde o cambia en L-BFGS (dependiente de semilla) |
| F | 0.285 (10/10, max 0.80) sirve | 2.045 (0/2, max 2.47) daña | **sin veredicto — n = 2 < 4** | 0.625 (8/10, max 2.82) dependiente de semilla | 0.258 (2/2, max 0.32) sirve | sin veredicto | {"lbfgs_time_cap": 2} | Adam (sirve); L-BFGS invierte (daña) |

D5: E frente a F (¿residuo o solo la base?): 1d mediana 0.616 (E mejor 2/2; respaldado=True, contrario=False); fase 1: 0.715 (7/10).

Atribucion 1d: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": false, "motivo": "E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)"}, "H-ATTR-B": {"aplica": true, "J_empeora": false, "F_empeora": true, "lectura": "el dano esta en la reparametrizacion misma"}}
Atribucion fase 1: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": true, "F_logB": -1.257058606910546, "E_logB": -1.4644113168417525, "respaldada": true, "lectura": "la ventaja es de correccion de la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Fisher-KPP (0 corridas) — receta: receta de Allen-Cahn (heredada; sin linea base propia)

| brazo | fase 1 (tras L-BFGS) | 1d (tras L-BFGS) | **veredicto** | fase 1 (Adam) | 1d (Adam) | veredicto Adam | L-BFGS 1d | fase del efecto 1d |
|---|---|---|---|---|---|---|---|---|

Atribucion 1d: {"E": "sin datos"}
Atribucion fase 1: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": false, "motivo": "E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Resumen para predictions.md (lo confirmado pesa mas que lo corto)

| problema | brazo | fase 1 | 1d | veredicto |
|---|---|---|---|---|
| Burgers RBF48 | E | dependiente de semilla | dependiente de semilla | sin veredicto |
| Burgers RBF48 | F | sirve | daña | sin veredicto |