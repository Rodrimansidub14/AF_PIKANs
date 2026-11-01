# EXP-0028 Fase 1e — confirmacion con receta completa (criterio uniforme, por problema; fin de Adam y fin de L-BFGS por separado)

Metrica de NS/DFG: `l2re_flat`. Veredicto = sostenido / se pierde / cambia de signo / aparece / sin efecto frente a la etiqueta de la fase 1 corta y, si el brazo existe en 1d, frente a la de 1d. n < 5 -> PROVISIONAL; n < 4 -> sin veredicto. J e I de ac/kpp/burgers se parean con el B de 1d.

## NS2d-CG (RBF16) — 1e: 25 corridas (+ 0 de 1d para el pareo)

Receta completa: Adam 20 000 f32 + L-BFGS 500 it. Fase 1 corta: Adam 2 000 + L-BFGS 500 it. (1c, t1).

B (fuente 1e, n = 5): mediana fin de Adam 6.853e-02, tras L-BFGS 6.613e-02.

### FIN DE L-BFGS (primario)

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F | 0.830 (10/10, max 0.93) sirve | - | **0.980 (4/5, max 1.02) indiferente** | **se pierde — neutro: la mediana cae en la banda [0.90, 1.10] (indiferente)** | sin etiqueta 1d — este problema no tiene 1d |
| J | 0.974 (6/10, max 1.19) indiferente | - | **1.056 (0/5, max 1.08) indiferente** | **sin efecto — fase 1: indiferente; 1d: indiferente** | sin etiqueta 1d — este problema no tiene 1d |
| E | 0.792 (10/10, max 0.94) sirve | - | **0.971 (5/5, max 0.98) indiferente** | **se pierde — neutro: la mediana cae en la banda [0.90, 1.10] (indiferente)** | sin etiqueta 1d — este problema no tiene 1d |
| I | 0.736 (10/10, max 0.81) sirve | - | **0.952 (5/5, max 0.99) indiferente** | **se pierde — neutro: la mediana cae en la banda [0.90, 1.10] (indiferente)** | sin etiqueta 1d — este problema no tiene 1d |

### FIN DE ADAM

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F | 0.676 (10/10, max 0.78) sirve | - | **0.970 (5/5, max 0.99) indiferente** | **se pierde — neutro: la mediana cae en la banda [0.90, 1.10] (indiferente)** | sin etiqueta 1d — este problema no tiene 1d |
| J | 1.002 (5/10, max 1.18) indiferente | - | **1.150 (0/5, max 1.16) daña** | **aparece — fase 1: indiferente** | sin etiqueta 1d — este problema no tiene 1d |
| E | 0.641 (10/10, max 0.82) sirve | - | **0.956 (5/5, max 0.96) indiferente** | **se pierde — neutro: la mediana cae en la banda [0.90, 1.10] (indiferente)** | sin etiqueta 1d — este problema no tiene 1d |
| I | 0.561 (10/10, max 0.68) sirve | - | **0.945 (4/5, max 1.01) indiferente** | **se pierde — neutro: la mediana cae en la banda [0.90, 1.10] (indiferente)** | sin etiqueta 1d — este problema no tiene 1d |

Fase del efecto y L-BFGS (1e):

- F: ninguna; L-BFGS {"lbfgs_iter_cap": 5}, iteraciones (mediana) 500.0
- J: Adam (daña); se pierde o cambia en L-BFGS (indiferente); L-BFGS {"lbfgs_iter_cap": 5}, iteraciones (mediana) 500.0
- E: ninguna; L-BFGS {"lbfgs_iter_cap": 4, "lbfgs_plateau": 1}, iteraciones (mediana) 500.0
- I: ninguna; L-BFGS {"lbfgs_iter_cap": 5}, iteraciones (mediana) 500.0

- D5: E frente a F (¿residuo o solo la base?): L-BFGS mediana 0.981 (A mejor 4/5; respaldado=False, contrario=False); Adam mediana 0.975 (4/5)
- J frente a B (¿basta reescalar?): L-BFGS mediana 1.056 (A mejor 0/5; respaldado=False, contrario=False); Adam mediana 1.150 (0/5)
- E frente a J (¿importan los terminos fuera de la diagonal?): L-BFGS mediana 0.910 (A mejor 5/5; respaldado=False, contrario=False); Adam mediana 0.823 (5/5)
- I frente a E (¿importa acoplar aristas?): L-BFGS mediana 1.006 (A mejor 1/5; respaldado=False, contrario=False); Adam mediana 1.011 (1/5)

Atribucion 1e: {"E": "indiferente", "H-ATTR-A": {"aplica": false, "motivo": "E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Allen-Cahn d=1e-3 — 1e: 10 corridas (+ 20 de 1d para el pareo)

Receta completa: Adam 40 000 f32 + L-BFGS f32 (tope 300 s / 15 000 it.). Fase 1 corta: Adam 4 000 + L-BFGS 500 it. (fase 1).

B (fuente 1d, n = 5): mediana fin de Adam 2.802e-02, tras L-BFGS 1.526e-03. Linea base congelada 2.838e-02 / 2.686e-03 (razon final 0.57x, informativo).

### FIN DE L-BFGS (primario)

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F (solo referencia) | 0.352 (10/10, max 0.51) sirve | 0.812 (3/5, max 3.95) dependiente de semilla | **-** | **-** | - |
| J | 0.937 (6/10, max 1.33) indiferente | - | **1.144 (2/5, max 26.65) dependiente de semilla** | **sin efecto — fase 1: indiferente; 1d: dependiente de semilla; direccion de la mediana: neutro -> empeora** | sin etiqueta 1d — 1d no tiene este brazo |
| E (solo referencia) | 0.162 (10/10, max 0.60) sirve | 0.389 (4/5, max 3.14) dependiente de semilla | **-** | **-** | - |
| I | 0.144 (6/10, max 11.62) dependiente de semilla | - | **3.534 (1/5, max 31.44) daña** | **aparece — fase 1: dependiente de semilla** | sin etiqueta 1d — 1d no tiene este brazo |
| C (solo referencia) | 0.783 (5/5, max 0.85) sirve | 1.007 (2/5, max 1.73) dependiente de semilla | **-** | **-** | - |

### FIN DE ADAM

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F (solo referencia) | 0.292 (10/10, max 0.46) sirve | 0.281 (5/5, max 0.50) sirve | **-** | **-** | - |
| J | 0.995 (5/10, max 1.37) indiferente | - | **0.896 (3/5, max 2.08) dependiente de semilla** | **sin efecto — fase 1: indiferente; 1d: dependiente de semilla; direccion de la mediana: neutro -> mejora** | sin etiqueta 1d — 1d no tiene este brazo |
| E (solo referencia) | 0.283 (10/10, max 0.47) sirve | 0.212 (5/5, max 0.36) sirve | **-** | **-** | - |
| I | 0.221 (6/10, max 5.73) dependiente de semilla | - | **0.194 (4/5, max 1.35) sirve** | **aparece — fase 1: dependiente de semilla** | sin etiqueta 1d — 1d no tiene este brazo |
| C (solo referencia) | 0.786 (4/5, max 1.02) sirve | 0.794 (4/5, max 1.09) sirve | **-** | **-** | - |

Fase del efecto y L-BFGS (1e):

- J: ninguna; L-BFGS {"lbfgs_plateau": 5}, iteraciones (mediana) 2127.0
- I: Adam (sirve); L-BFGS invierte (daña); L-BFGS {"lbfgs_plateau": 5}, iteraciones (mediana) 12.0

- D5: E frente a F (¿residuo o solo la base?): L-BFGS mediana 0.906 (A mejor 3/5; respaldado=False, contrario=False); Adam mediana 0.724 (4/5)
- J frente a B (¿basta reescalar?): L-BFGS mediana 1.144 (A mejor 2/5; respaldado=False, contrario=False); Adam mediana 0.896 (3/5)
- E frente a J (¿importan los terminos fuera de la diagonal?): L-BFGS mediana 0.471 (A mejor 4/5; respaldado=True, contrario=False); Adam mediana 0.175 (5/5)
- I frente a E (¿importa acoplar aristas?): L-BFGS mediana 3.529 (A mejor 1/5; respaldado=False, contrario=True); Adam mediana 1.360 (2/5)

Atribucion 1e: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": true, "F_logB": -0.20797617171380225, "E_logB": -0.9436513568029327, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Fisher-KPP — 1e: 10 corridas (+ 15 de 1d para el pareo)

Receta completa: receta de Allen-Cahn (heredada; sin linea base propia). Fase 1 corta: Adam 4 000 + L-BFGS 500 it. (fase 1).

B (fuente 1d, n = 5): mediana fin de Adam 1.325e-01, tras L-BFGS 8.630e-02.

### FIN DE L-BFGS (primario)

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F (solo referencia) | 1.835 (4/10, max 10.47) dependiente de semilla | - | **-** | **-** | - |
| J | 2.379 (3/10, max 23.63) daÃ±a | - | **0.235 (5/5, max 0.38) sirve** | **aparece — fase 1: daÃ±a** | sin etiqueta 1d — 1d no tiene este brazo |
| E (solo referencia) | 0.767 (6/10, max 21.79) dependiente de semilla | - | **-** | **-** | - |
| I | 1.390 (5/10, max 13.65) dependiente de semilla | - | **0.160 (4/5, max 22.28) dependiente de semilla** | **cambia de signo — dependiente de semilla -> dependiente de semilla, la mediana pasa de empeora a mejora (ninguna fue sirve/daña)** | sin etiqueta 1d — 1d no tiene este brazo |

### FIN DE ADAM

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F (solo referencia) | 1.094 (5/10, max 6.86) dependiente de semilla | - | **-** | **-** | - |
| J | 2.716 (2/10, max 23.32) daÃ±a | - | **0.202 (5/5, max 0.37) sirve** | **aparece — fase 1: daÃ±a** | sin etiqueta 1d — 1d no tiene este brazo |
| E (solo referencia) | 0.762 (7/10, max 16.47) dependiente de semilla | - | **-** | **-** | - |
| I | 1.325 (5/10, max 16.53) dependiente de semilla | - | **0.160 (4/5, max 21.90) dependiente de semilla** | **cambia de signo — dependiente de semilla -> dependiente de semilla, la mediana pasa de empeora a mejora (ninguna fue sirve/daña)** | sin etiqueta 1d — 1d no tiene este brazo |

Fase del efecto y L-BFGS (1e):

- J: Adam y L-BFGS (se mantiene); L-BFGS {"lbfgs_plateau": 5}, iteraciones (mediana) 9.0
- I: ninguna; L-BFGS {"lbfgs_plateau": 5}, iteraciones (mediana) 17.0

- D5: E frente a F (¿residuo o solo la base?): L-BFGS mediana 0.242 (A mejor 5/5; respaldado=True, contrario=False); Adam mediana 0.237 (5/5)
- J frente a B (¿basta reescalar?): L-BFGS mediana 0.235 (A mejor 5/5; respaldado=True, contrario=False); Adam mediana 0.202 (5/5)
- E frente a J (¿importan los terminos fuera de la diagonal?): L-BFGS mediana 0.568 (A mejor 3/5; respaldado=False, contrario=False); Adam mediana 0.568 (3/5)
- I frente a E (¿importa acoplar aristas?): L-BFGS mediana 3.028 (A mejor 0/5; respaldado=False, contrario=True); Adam mediana 3.026 (0/5)

Atribucion 1e: {"E": "sirve", "H-ATTR-A": {"aplica": true, "F_logB": -0.4462501079021234, "E_logB": -2.1309966938612916, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Resumen para predictions.md (lo confirmado a receta completa pesa mas que lo corto)

| problema | brazo | fase 1 (Adam / L-BFGS) | 1d (Adam / L-BFGS) | 1e (Adam / L-BFGS) | veredicto Adam | veredicto L-BFGS | vs 1d (L-BFGS) |
|---|---|---|---|---|---|---|---|
| NS2d-CG (RBF16) | F | sirve / sirve | sin datos / sin datos | indiferente / indiferente | se pierde | se pierde | sin etiqueta 1d |
| NS2d-CG (RBF16) | J | indiferente / indiferente | sin datos / sin datos | daña / indiferente | aparece | sin efecto | sin etiqueta 1d |
| NS2d-CG (RBF16) | E | sirve / sirve | sin datos / sin datos | indiferente / indiferente | se pierde | se pierde | sin etiqueta 1d |
| NS2d-CG (RBF16) | I | sirve / sirve | sin datos / sin datos | indiferente / indiferente | se pierde | se pierde | sin etiqueta 1d |
| Allen-Cahn d=1e-3 | J | indiferente / indiferente | sin datos / sin datos | dependiente de semilla / dependiente de semilla | sin efecto | sin efecto | sin etiqueta 1d |
| Allen-Cahn d=1e-3 | I | dependiente de semilla / dependiente de semilla | sin datos / sin datos | sirve / daña | aparece | aparece | sin etiqueta 1d |
| Fisher-KPP | J | daÃ±a / daÃ±a | sin datos / sin datos | sirve / sirve | aparece | aparece | sin etiqueta 1d |
| Fisher-KPP | I | dependiente de semilla / dependiente de semilla | sin datos / sin datos | dependiente de semilla / dependiente de semilla | cambia de signo | cambia de signo | sin etiqueta 1d |