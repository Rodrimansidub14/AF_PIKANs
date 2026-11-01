# EXP-0028 Fase 1e — confirmacion con receta completa (criterio uniforme, por problema; fin de Adam y fin de L-BFGS por separado)

Metrica de NS/DFG: `l2re_flat`. Veredicto = sostenido / se pierde / cambia de signo / aparece / sin efecto frente a la etiqueta de la fase 1 corta y, si el brazo existe en 1d, frente a la de 1d. n < 5 -> PROVISIONAL; n < 4 -> sin veredicto. J e I de ac/kpp/burgers se parean con el B de 1d.

## NS2d-CG (RBF16) — 1e: 15 corridas (+ 0 de 1d para el pareo)

Receta completa: Adam 20 000 f32 + L-BFGS 500 it. Fase 1 corta: Adam 2 000 + L-BFGS 500 it. (1c, t1).

B (fuente 1e, n = 3): mediana fin de Adam 6.853e-02, tras L-BFGS 6.464e-02.

### FIN DE L-BFGS (primario)

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F | 0.830 (10/10, max 0.93) sirve | - | **0.984 (2/3, max 1.02) indiferente** | **sin veredicto — n = 3 < 4** | sin etiqueta 1d — este problema no tiene 1d |
| J | 0.974 (6/10, max 1.19) indiferente | - | **1.077 (0/3, max 1.08) indiferente** | **sin veredicto — n = 3 < 4** | sin etiqueta 1d — este problema no tiene 1d |
| E | 0.792 (10/10, max 0.94) sirve | - | **0.981 (3/3, max 0.98) indiferente** | **sin veredicto — n = 3 < 4** | sin etiqueta 1d — este problema no tiene 1d |
| I | 0.736 (10/10, max 0.81) sirve | - | **0.981 (3/3, max 0.99) indiferente** | **sin veredicto — n = 3 < 4** | sin etiqueta 1d — este problema no tiene 1d |

### FIN DE ADAM

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F | 0.676 (10/10, max 0.78) sirve | - | **0.970 (3/3, max 0.98) indiferente** | **sin veredicto — n = 3 < 4** | sin etiqueta 1d — este problema no tiene 1d |
| J | 1.002 (5/10, max 1.18) indiferente | - | **1.150 (0/3, max 1.16) daña** | **sin veredicto — n = 3 < 4** | sin etiqueta 1d — este problema no tiene 1d |
| E | 0.641 (10/10, max 0.82) sirve | - | **0.960 (3/3, max 0.96) indiferente** | **sin veredicto — n = 3 < 4** | sin etiqueta 1d — este problema no tiene 1d |
| I | 0.561 (10/10, max 0.68) sirve | - | **0.966 (2/3, max 1.01) indiferente** | **sin veredicto — n = 3 < 4** | sin etiqueta 1d — este problema no tiene 1d |

Fase del efecto y L-BFGS (1e):

- F: ninguna; L-BFGS {"lbfgs_iter_cap": 3}, iteraciones (mediana) 500.0
- J: Adam (daña); se pierde o cambia en L-BFGS (indiferente); L-BFGS {"lbfgs_iter_cap": 3}, iteraciones (mediana) 500.0
- E: ninguna; L-BFGS {"lbfgs_iter_cap": 3}, iteraciones (mediana) 500.0
- I: ninguna; L-BFGS {"lbfgs_iter_cap": 3}, iteraciones (mediana) 500.0

- D5: E frente a F (¿residuo o solo la base?): L-BFGS mediana 0.999 (A mejor 2/3; respaldado=False, contrario=False); Adam mediana 0.994 (2/3)
- J frente a B (¿basta reescalar?): L-BFGS mediana 1.077 (A mejor 0/3; respaldado=False, contrario=False); Adam mediana 1.150 (0/3)
- E frente a J (¿importan los terminos fuera de la diagonal?): L-BFGS mediana 0.910 (A mejor 3/3; respaldado=False, contrario=False); Adam mediana 0.836 (3/3)
- I frente a E (¿importa acoplar aristas?): L-BFGS mediana 1.001 (A mejor 1/3; respaldado=False, contrario=False); Adam mediana 1.011 (1/3)

Atribucion 1e: {"E": "indiferente", "H-ATTR-A": {"aplica": false, "motivo": "E no mejora a B en este problema (mediana <= 0.90 y >= 7/10)"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Allen-Cahn d=1e-3 — 1e: 0 corridas (+ 20 de 1d para el pareo)

Receta completa: Adam 40 000 f32 + L-BFGS f32 (tope 300 s / 15 000 it.). Fase 1 corta: Adam 4 000 + L-BFGS 500 it. (fase 1).

B (fuente 1d, n = 5): mediana fin de Adam 2.802e-02, tras L-BFGS 1.526e-03. Linea base congelada 2.838e-02 / 2.686e-03 (razon final 0.57x, informativo).

### FIN DE L-BFGS (primario)

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F (solo referencia) | 0.352 (10/10, max 0.51) sirve | 0.812 (3/5, max 3.95) dependiente de semilla | **-** | **-** | - |
| J (solo referencia) | 0.937 (6/10, max 1.33) indiferente | - | **-** | **-** | - |
| E (solo referencia) | 0.162 (10/10, max 0.60) sirve | 0.389 (4/5, max 3.14) dependiente de semilla | **-** | **-** | - |
| I (solo referencia) | 0.144 (6/10, max 11.62) dependiente de semilla | - | **-** | **-** | - |
| C (solo referencia) | 0.783 (5/5, max 0.85) sirve | 1.007 (2/5, max 1.73) dependiente de semilla | **-** | **-** | - |

### FIN DE ADAM

| brazo | fase 1 corta | 1d | **1e** | veredicto vs fase 1 | veredicto vs 1d |
|---|---|---|---|---|---|
| F (solo referencia) | 0.292 (10/10, max 0.46) sirve | 0.281 (5/5, max 0.50) sirve | **-** | **-** | - |
| J (solo referencia) | 0.995 (5/10, max 1.37) indiferente | - | **-** | **-** | - |
| E (solo referencia) | 0.283 (10/10, max 0.47) sirve | 0.212 (5/5, max 0.36) sirve | **-** | **-** | - |
| I (solo referencia) | 0.221 (6/10, max 5.73) dependiente de semilla | - | **-** | **-** | - |
| C (solo referencia) | 0.786 (4/5, max 1.02) sirve | 0.794 (4/5, max 1.09) sirve | **-** | **-** | - |

Fase del efecto y L-BFGS (1e):


- D5: E frente a F (¿residuo o solo la base?): L-BFGS mediana 0.906 (A mejor 3/5; respaldado=False, contrario=False); Adam mediana 0.724 (4/5)

Atribucion 1e: {"E": "dependiente de semilla", "H-ATTR-A": {"aplica": true, "F_logB": -0.20797617171380225, "E_logB": -0.9436513568029327, "respaldada": false, "lectura": "la ventaja no se explica solo por la base"}, "H-ATTR-B": {"aplica": false, "motivo": "E no empeora (mediana <= 1.10)"}}

## Resumen para predictions.md (lo confirmado a receta completa pesa mas que lo corto)

| problema | brazo | fase 1 (Adam / L-BFGS) | 1d (Adam / L-BFGS) | 1e (Adam / L-BFGS) | veredicto Adam | veredicto L-BFGS | vs 1d (L-BFGS) |
|---|---|---|---|---|---|---|---|
| NS2d-CG (RBF16) | F | sirve / sirve | sin datos / sin datos | indiferente / indiferente | sin veredicto | sin veredicto | sin etiqueta 1d |
| NS2d-CG (RBF16) | J | indiferente / indiferente | sin datos / sin datos | daña / indiferente | sin veredicto | sin veredicto | sin etiqueta 1d |
| NS2d-CG (RBF16) | E | sirve / sirve | sin datos / sin datos | indiferente / indiferente | sin veredicto | sin veredicto | sin etiqueta 1d |
| NS2d-CG (RBF16) | I | sirve / sirve | sin datos / sin datos | indiferente / indiferente | sin veredicto | sin veredicto | sin etiqueta 1d |