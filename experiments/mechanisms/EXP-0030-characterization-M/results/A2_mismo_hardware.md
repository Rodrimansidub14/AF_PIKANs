# EXP-0030, bloque A2: misma GPU (A100-80), semillas distintas (exploratorio, post hoc)

u robusto = mediana de las evaluaciones de los pasos 18 000-20 000. Converso. Las semillas de M (28-30) y de B/E (31-32) no coinciden: no es un pareo, es un contraste de bandas dentro del mismo hardware.

| brazo | hardware | semillas | u robusto por semilla | mín - máx |
|---|---|---|---|---|
| B | T4 | [28] | [0.391] | 0.391 - 0.391 |
| B | H100 | [29, 30] | [0.071, 0.164] | 0.071 - 0.164 |
| B | A100-80 | [31, 32] | [0.039, 0.094] | 0.039 - 0.094 |
| E | T4 | [28] | [0.074] | 0.074 - 0.074 |
| E | H100 | [29, 30] | [0.077, 0.033] | 0.033 - 0.077 |
| E | A100-80 | [31, 32] | [0.07, 0.102] | 0.070 - 0.102 |
| M | A100-80 | [28, 29, 30] | [0.369, 0.168, 0.207] | 0.168 - 0.369 |

- E en todo el hardware (n = 5): 0.033 - 0.102. E solo en A100-80: 0.070 - 0.102.
- B en todo el hardware (n = 5): 0.039 - 0.391. B solo en A100-80: 0.039 - 0.094.
- M en A100-80 (n = 3): 0.168 - 0.369. El mínimo de M (0.168) supera el máximo de E en cualquier hardware (0.102).

Lectura: la banda de E es angosta y parecida en T4, H100 y A100-80; la «pérdida de la ventaja de E sobre B» en las semillas 31-32 viene de que B salió bien en la A100 (0.039, 0.094), no de que E saliera peor. Las tres semillas de M quedan por encima de la banda de E, también en la misma GPU. El hardware no explica esa diferencia, pero tampoco se puede descartar que el efecto de las semillas 28-30 sobre M difiera del de 31-32 sobre E (semillas distintas). Con n = 2 de E en A100-80, es una observación, no una prueba.