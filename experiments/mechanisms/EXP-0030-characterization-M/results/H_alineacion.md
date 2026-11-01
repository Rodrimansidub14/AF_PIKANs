# H. Alineación de gradientes por grupo físico y dispersión por arista (post hoc, exploratorio)

Marco: conflicto tipo I (magnitudes distintas) y tipo II (direcciones opuestas), Wang et al. (arXiv:2502.00604, resumen). Mediana entre semillas 28-30 de: `align` = coseno medio por pares entre mec, equil y borde; `razón` = norma mayor / menor entre esos tres grupos; `disp` = décadas entre el percentil 5 y 95 de la norma del gradiente por arista. Entre paréntesis, en c' (lo que ve Adam). 128 puntos por clave; estados guardados.

| problema | paso | brazo | align (c) | razón (c) | disp (c) | disp (c′) | cos mec-equil (c) |
|---|---|---|---|---|---|---|---|
| converse | 0 | B (n=3) | -0.01 | 3037.8 | 3.59 | - | -0.00 |
| converse | 0 | E (n=3) | -0.01 | 3037.8 | 3.59 | 3.59 | -0.00 |
| converse | 0 | M (n=3) | -0.01 | 3037.8 | 3.59 | 3.59 | -0.00 |
| converse | 2000 | B (n=3) | 0.02 | 1292.9 | 3.69 | - | 0.01 |
| converse | 2000 | E (n=3) | 0.03 | 148.5 | 2.63 | 2.63 | 0.01 |
| converse | 2000 | M (n=3) | 0.00 | 202.4 | 2.97 | 2.95 | -0.03 |
| converse | 4000 | B (n=3) | 0.01 | 1366.1 | 3.22 | - | -0.01 |
| converse | 4000 | E (n=3) | 0.02 | 253.5 | 2.70 | 2.73 | 0.01 |
| converse | 4000 | M (n=3) | -0.01 | 1457.8 | 3.39 | 3.36 | -0.00 |
| converse | 10000 | B (n=3) | 0.00 | 2883.8 | 3.42 | - | 0.00 |
| converse | 10000 | E (n=3) | 0.00 | 1490.0 | 3.09 | 3.06 | 0.00 |
| converse | 10000 | M (n=3) | -0.00 | 1160.5 | 3.23 | 3.26 | 0.00 |
| converse | 20000 | B (n=3) | 0.02 | 3837.7 | 3.09 | - | 0.00 |
| converse | 20000 | E (n=3) | 0.05 | 3113.1 | 3.25 | 3.42 | 0.00 |
| converse | 20000 | M (n=3) | -0.01 | 2998.4 | 3.72 | 3.70 | -0.00 |
| direct | 0 | B (n=3) | 0.02 | 1253.7 | 3.39 | - | 0.01 |
| direct | 0 | E (n=3) | 0.02 | 1253.7 | 3.39 | 3.39 | 0.01 |
| direct | 0 | M (n=3) | 0.02 | 1253.7 | 3.39 | 3.39 | 0.01 |
| direct | 2000 | B (n=3) | -0.04 | 11.3 | 2.01 | - | -0.11 |
| direct | 2000 | E (n=3) | -0.01 | 12.0 | 2.73 | 2.69 | -0.05 |
| direct | 2000 | M (n=3) | -0.06 | 5.6 | 2.68 | 2.51 | -0.14 |
| direct | 4000 | B (n=3) | -0.05 | 7.9 | 2.48 | - | -0.14 |
| direct | 4000 | E (n=3) | -0.01 | 8.8 | 2.74 | 2.48 | -0.08 |
| direct | 4000 | M (n=3) | -0.03 | 2.7 | 2.45 | 2.43 | -0.20 |
| direct | 10000 | B (n=3) | 0.03 | 18.3 | 2.48 | - | 0.07 |
| direct | 10000 | E (n=3) | 0.06 | 28.5 | 3.22 | 2.84 | -0.01 |
| direct | 10000 | M (n=3) | -0.14 | 3.1 | 2.64 | 2.56 | -0.24 |
| direct | 20000 | B (n=3) | -0.06 | 46.4 | 3.33 | - | -0.16 |
| direct | 20000 | E (n=3) | -0.00 | 38.5 | 2.74 | 2.71 | -0.05 |
| direct | 20000 | M (n=3) | -0.02 | 5.5 | 2.81 | 2.64 | -0.10 |

## Lectura (descriptiva; n = 3 semillas por celda, estados guardados, 128 puntos por clave)
1. **No hay conflicto tipo II dominante.** El coseno medio entre grupos queda en |align| ≤ 0.06 casi siempre: los gradientes de mecánica, equilibrio y borde son casi ortogonales, no opuestos. Lo que domina es el tipo I (magnitudes): en el converso el grupo mayor pesa 1 300-3 800 veces el menor; en el directo, 5-46 veces.
2. **E reduce la razón de magnitudes en la fase del llano del converso** (pasos 2 000-4 000: B 1 293 y 1 366; E 149 y 254) y la dispersión entre aristas (B 3.7 y 3.2 décadas; E 2.6 y 2.7). A partir del paso 10 000 las diferencias entre brazos desaparecen (razón 1 490-3 837, dispersión 3.1-3.4). Coincide en el tiempo con la ventaja de E en el tiempo de salida del llano (mediana 5 500 frente a 7 000); es una coincidencia temporal descrita, no una prueba de causa.
3. **M no conserva esa reducción en el converso:** razón 202 en el paso 2 000 pero 1 458 en el 4 000 (como B). En el **directo** M baja la razón de magnitudes (3-6 frente a 9-38 de B y E), que es lo que P2 busca, a costa de gradientes algo más opuestos (cos mec-equil −0.10 a −0.24 en M; −0.05 y +0.07 en B/E). Esto es consistente con el efecto pequeño de M sobre el error del directo (0.180 frente a 0.193), sin ser una prueba.
4. **La dispersión entre aristas casi no cambia en c′** (misma cifra en c y c′ para E y M, por ejemplo 2.63 y 2.63 en el paso 2 000): P está normalizada por traza en cada arista (`B/m`), así que no iguala la escala *entre* aristas, solo la forma *dentro* de cada una. Interpretación (no medida): Adam ya normaliza la escala por coordenada, y lo que no puede resolver es la correlación dentro de la arista (bloques casi de rango uno), que es lo que E ataca.
5. **Sanidad:** en el paso 0 B, E y M dan exactamente las mismas cifras (mismo modelo inicial).
