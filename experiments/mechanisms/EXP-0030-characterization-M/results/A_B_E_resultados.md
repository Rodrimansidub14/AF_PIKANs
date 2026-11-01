# EXP-0030, bloques A, B y E (exploratorio, post hoc; sin p-valores)

## A. Efecto por corrida

`u_rob` y `phi_rob` = mediana de las evaluaciones de los pasos 18 000-20 000 (post hoc). `salida` = primer paso evaluado con l2re_u < 0.5. Hardware siempre visible.

| problema | brazo | semilla | hardware | u final | u rob. | φ final | φ rob. | salida del llano | s/paso |
|---|---|---|---|---|---|---|---|---|---|
| conv | B | 28 | T4 | 0.367 | 0.391 | 0.041 | 0.039 | 7000 | 0.432 |
| conv | B | 29 | H100 | 0.048 | 0.071 | 0.044 | 0.044 | 6000 | 0.213 |
| conv | B | 30 | H100 | 0.180 | 0.164 | 0.046 | 0.046 | 8000 | 0.207 |
| conv | B | 31 | A100-80 | 0.031 | 0.039 | 0.047 | 0.047 | 7500 | 0.311 |
| conv | B | 32 | A100-80 | 0.135 | 0.094 | 0.047 | 0.047 | 6000 | 0.311 |
| conv | E | 28 | T4 | 0.050 | 0.074 | 0.051 | 0.051 | 3500 | 0.456 |
| conv | E | 29 | H100 | 0.077 | 0.077 | 0.053 | 0.051 | 5500 | 0.222 |
| conv | E | 30 | H100 | 0.033 | 0.033 | 0.051 | 0.051 | 7000 | 0.231 |
| conv | E | 31 | A100-80 | 0.108 | 0.070 | 0.055 | 0.053 | 6500 | 0.321 |
| conv | E | 32 | A100-80 | 0.521 | 0.102 | 0.040 | 0.049 | 4500 | 0.321 |
| conv | M | 28 | A100-80 | 0.336 | 0.369 | 0.040 | 0.039 | 12500 | 0.351 |
| conv | M | 29 | A100-80 | 0.166 | 0.168 | 0.044 | 0.043 | 6000 | 0.350 |
| conv | M | 30 | A100-80 | 0.165 | 0.207 | 0.044 | 0.043 | 5000 | 0.347 |
| dire | B | 28 | T4 | 0.180 | 0.195 | 0.184 | 0.245 | 1000 | 0.326 |
| dire | B | 29 | H100 | 0.240 | 0.226 | 0.264 | 0.246 | 1500 | 0.171 |
| dire | B | 30 | H100 | 0.204 | 0.204 | 0.247 | 0.247 | 1500 | 0.163 |
| dire | E | 28 | T4 | 0.129 | 0.121 | 0.151 | 0.168 | 1500 | 0.349 |
| dire | E | 29 | H100 | 0.198 | 0.204 | 0.216 | 0.231 | 1500 | 0.187 |
| dire | E | 30 | H100 | 0.253 | 0.296 | 0.277 | 0.795 | 1500 | 0.172 |
| dire | M | 28 | A100-80 | 0.153 | 0.162 | 0.142 | 0.196 | 1000 | 0.297 |
| dire | M | 29 | A100-40 | 0.178 | 0.181 | 0.215 | 0.228 | 500 | 0.246 |
| dire | M | 30 | A100-40 | 0.208 | 0.227 | 0.254 | 0.250 | 1000 | 0.249 |

## A. Cola por brazo (mínimo / mediana / máximo)

| problema | brazo | n | métrica | mín. | mediana | máx. |
|---|---|---|---|---|---|---|
| conv | B | 5 | u final | 0.0306 | 0.135 | 0.367 |
| conv | B | 5 | u robusto | 0.0389 | 0.0943 | 0.391 |
| conv | B | 5 | φ final | 0.0408 | 0.046 | 0.0468 |
| conv | B | 5 | salida del llano | 6e+03 | 7e+03 | 8e+03 |
| conv | E | 5 | u final | 0.0326 | 0.0767 | 0.521 |
| conv | E | 5 | u robusto | 0.0326 | 0.0742 | 0.102 |
| conv | E | 5 | φ final | 0.0403 | 0.0513 | 0.0552 |
| conv | E | 5 | salida del llano | 3.5e+03 | 5.5e+03 | 7e+03 |
| conv | M | 3 | u final | 0.165 | 0.166 | 0.336 |
| conv | M | 3 | u robusto | 0.168 | 0.207 | 0.369 |
| conv | M | 3 | φ final | 0.0404 | 0.0437 | 0.0444 |
| conv | M | 3 | salida del llano | 5e+03 | 6e+03 | 1.25e+04 |
| dire | B | 3 | u final | 0.18 | 0.204 | 0.24 |
| dire | B | 3 | u robusto | 0.195 | 0.204 | 0.226 |
| dire | B | 3 | φ final | 0.184 | 0.247 | 0.264 |
| dire | B | 3 | salida del llano | 1e+03 | 1.5e+03 | 1.5e+03 |
| dire | E | 3 | u final | 0.129 | 0.198 | 0.253 |
| dire | E | 3 | u robusto | 0.121 | 0.204 | 0.296 |
| dire | E | 3 | φ final | 0.151 | 0.216 | 0.277 |
| dire | E | 3 | salida del llano | 1.5e+03 | 1.5e+03 | 1.5e+03 |
| dire | M | 3 | u final | 0.153 | 0.178 | 0.208 |
| dire | M | 3 | u robusto | 0.162 | 0.181 | 0.227 |
| dire | M | 3 | φ final | 0.142 | 0.215 | 0.254 |
| dire | M | 3 | salida del llano | 500 | 1e+03 | 1e+03 |

## A. Razones pareadas por semilla (valores individuales, mediana y rango)

| problema | razón | métrica | n | por semilla | mediana | rango | hardware del par |
|---|---|---|---|---|---|---|---|
| conv | E/B | u final | 5 | 28:0.13, 29:1.59, 30:0.18, 31:3.51, 32:3.87 | 1.59 | 0.13-3.87 | mismo hardware |
| conv | E/B | u robusto | 5 | 28:0.19, 29:1.08, 30:0.20, 31:1.79, 32:1.08 | 1.08 | 0.19-1.79 | mismo hardware |
| conv | E/B | φ final | 5 | 28:1.25, 29:1.18, 30:1.11, 31:1.18, 32:0.87 | 1.18 | 0.87-1.25 | mismo hardware |
| conv | M/E | u final | 3 | 28:6.80, 29:2.16, 30:5.05 | 5.05 | 2.16-6.80 | hardware distinto en algún par |
| conv | M/E | u robusto | 3 | 28:4.97, 29:2.20, 30:6.35 | 4.97 | 2.20-6.35 | hardware distinto en algún par |
| conv | M/E | φ final | 3 | 28:0.79, 29:0.83, 30:0.87 | 0.83 | 0.79-0.87 | hardware distinto en algún par |
| conv | M/B | u final | 3 | 28:0.92, 29:3.44, 30:0.92 | 0.92 | 0.92-3.44 | hardware distinto en algún par |
| conv | M/B | u robusto | 3 | 28:0.94, 29:2.36, 30:1.26 | 1.26 | 0.94-2.36 | hardware distinto en algún par |
| conv | M/B | φ final | 3 | 28:0.99, 29:0.98, 30:0.96 | 0.98 | 0.96-0.99 | hardware distinto en algún par |
| dire | E/B | u final | 3 | 28:0.71, 29:0.82, 30:1.24 | 0.82 | 0.71-1.24 | mismo hardware |
| dire | E/B | u robusto | 3 | 28:0.62, 29:0.90, 30:1.45 | 0.90 | 0.62-1.45 | mismo hardware |
| dire | E/B | φ final | 3 | 28:0.82, 29:0.82, 30:1.12 | 0.82 | 0.82-1.12 | mismo hardware |
| dire | M/E | u final | 3 | 28:1.19, 29:0.90, 30:0.82 | 0.90 | 0.82-1.19 | hardware distinto en algún par |
| dire | M/E | u robusto | 3 | 28:1.34, 29:0.89, 30:0.77 | 0.89 | 0.77-1.34 | hardware distinto en algún par |
| dire | M/E | φ final | 3 | 28:0.94, 29:1.00, 30:0.92 | 0.94 | 0.92-1.00 | hardware distinto en algún par |
| dire | M/B | u final | 3 | 28:0.85, 29:0.74, 30:1.02 | 0.85 | 0.74-1.02 | hardware distinto en algún par |
| dire | M/B | u robusto | 3 | 28:0.83, 29:0.80, 30:1.12 | 0.83 | 0.80-1.12 | hardware distinto en algún par |
| dire | M/B | φ final | 3 | 28:0.77, 29:0.82, 30:1.03 | 0.82 | 0.77-1.03 | hardware distinto en algún par |

## B. Estabilidad de la conclusión frente a las semillas (conteo combinatorio, no una prueba)

Para cada subconjunto de semillas: victorias del numerador (razón < 1) y mediana de la razón. E/B en el converso (n = 5): subconjuntos de 4 y de 3. M/E y M/B en las 3 semillas de M: se deja una fuera.

| problema | razón | métrica | semillas | subconjuntos | resultado | rango de la mediana |
|---|---|---|---|---|---|---|
| converso | E/B | u final | 4 de 5 | 5 | E gana (más de la mitad de las semillas) en 0 de 5 | 0.88-2.55 |
| converso | E/B | u robusto | 4 de 5 | 5 | E gana (más de la mitad de las semillas) en 0 de 5 | 0.64-1.08 |
| converso | E/B | u final | 3 de 5 | 10 | E gana (más de la mitad de las semillas) en 3 de 10 | 0.18-3.51 |
| converso | E/B | u robusto | 3 de 5 | 10 | E gana (más de la mitad de las semillas) en 3 de 10 | 0.20-1.08 |
| converso | M/E | u final | 2 de 3 | 3 | M gana las 2 semillas en 0 de 3 | 3.61-5.92 |
| converso | M/E | u robusto | 2 de 3 | 3 | M gana las 2 semillas en 0 de 3 | 3.58-5.66 |
| converso | M/B | u final | 2 de 3 | 3 | M gana las 2 semillas en 1 de 3 | 0.92-2.18 |
| converso | M/B | u robusto | 2 de 3 | 3 | M gana las 2 semillas en 0 de 3 | 1.10-1.81 |
| directo | M/E | u final | 2 de 3 | 3 | M gana las 2 semillas en 1 de 3 | 0.86-1.05 |
| directo | M/E | u robusto | 2 de 3 | 3 | M gana las 2 semillas en 1 de 3 | 0.83-1.12 |
| directo | M/B | u final | 2 de 3 | 3 | M gana las 2 semillas en 1 de 3 | 0.80-0.94 |
| directo | M/B | u robusto | 2 de 3 | 3 | M gana las 2 semillas en 1 de 3 | 0.82-0.97 |

## E. Sensibilidad retrospectiva (descriptiva, no es varianza explicada)

Mediana de log10 de la razón sobre la métrica robusta, por factor. **Los factores «patas» y «hardware» están confundidos** (todo M corrió en A100; ver Desviaciones de la spec): el número no los separa.

| problema | factor | n | efecto | rango |
|---|---|---|---|---|
| conv | cuerpo (E/B) | 5 | +0.03 | -0.72 a +0.25 |
| conv | patas (M/E) | 3 | +0.70 | +0.34 a +0.80 |
| dire | cuerpo (E/B) | 3 | -0.05 | -0.21 a +0.16 |
| dire | patas (M/E) | 3 | -0.05 | -0.11 a +0.13 |
| conv | B en T4 (semillas [28]) | 1 | mediana u rob. 0.391 | 0.391 a 0.391 |
| conv | B en H100 (semillas [29, 30]) | 2 | mediana u rob. 0.118 | 0.071 a 0.164 |
| conv | B en A100-80 (semillas [31, 32]) | 2 | mediana u rob. 0.067 | 0.039 a 0.094 |
| conv | E en T4 (semillas [28]) | 1 | mediana u rob. 0.074 | 0.074 a 0.074 |
| conv | E en H100 (semillas [29, 30]) | 2 | mediana u rob. 0.055 | 0.033 a 0.077 |
| conv | E en A100-80 (semillas [31, 32]) | 2 | mediana u rob. 0.086 | 0.070 a 0.102 |

Morris queda diferido hasta que haya cómputo (ver spec).