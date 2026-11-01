# EXP-0029: M frente a las predicciones firmadas (Modal A100, 2026-10-26)

Datos: `results/piezo-exp0029-final/` (10 de 12 corridas; **M converso semillas 31 y 32 fallaron por memoria**: CUDA out of memory en A100 de 40 GB con 4 procesos y el registro de bloques completos). B y E de las semillas 28-30 vienen de Kaggle (28) y Modal H100 (29-30); 31-32 de este bloque (A100). l2re_u al paso 20 000.

| problema | semilla | B u / φ | E u / φ | M u / φ |
|---|---|---|---|---|
| converso | 28 | 0.367 / 0.041 | 0.050 / 0.051 | 0.336 / 0.040 |
| converso | 29 | 0.048 / 0.044 | 0.077 / 0.053 | 0.166 / 0.044 |
| converso | 30 | 0.180 / 0.046 | 0.033 / 0.051 | 0.165 / 0.044 |
| converso | 31 | 0.031 / 0.047 | 0.108 / 0.055 | (falló) |
| converso | 32 | 0.135 / 0.047 | 0.521 / 0.040 | (falló) |
| directo | 28 | 0.180 / 0.184 | 0.129 / 0.151 | 0.153 / 0.142 |
| directo | 29 | 0.240 / 0.264 | 0.198 / 0.216 | 0.178 / 0.215 |
| directo | 30 | 0.204 / 0.247 | 0.253 / 0.277 | 0.208 / 0.254 |

Medias (semillas 28-30): converso B 0.198, E 0.053, M 0.222; directo B 0.208, E 0.193, M 0.180. Con n = 5 en B y E del converso: B media 0.152 (mediana 0.135), E media 0.158 (mediana 0.077); E gana a B en 2 de 5 semillas pareadas (28, 30), pierde en 3 (29, 31, 32; E seed 32 = 0.521).

## Predicciones de DISENO_FINAL §3
1. Converso u(M)/u(E) ∈ [0.85, 1.15] y u(M)/u(B) ≤ 0.5: **NO se cumple.** Cocientes M/E: 6.7, 2.2, 5.0; M/B: 0.92, 3.5, 0.92 (mediana 0.92). M no conserva la ventaja de E: se comporta como B.
2. Directo u(M)/u(E) ∈ [0.85, 1.15], ninguna semilla > 1.5×E: cocientes 1.19, 0.90, 0.82. Ninguna > 1.5 (cumple); 2 de 3 fuera del intervalo (una a cada lado): **formalmente falsada**, con M ≤ E en 2 de 3 semillas (media 0.180 frente a 0.193).
3. Converso φ(M)/φ(E) ≤ 1.20: **se cumple** (0.78, 0.83, 0.86; φ de M ≈ la de B, ≈ 0.043).
4. Saltos del directo M ≤ E + 3: **no evaluable**: M tiene registro por paso (64-68 saltos en el directo, 38-39 en el converso); las corridas viejas de E tienen la pérdida cada 100 pasos, no comparable.
5. Costo por paso M ≤ 1.10× E: converso A100, M 0.35 s frente a E 0.32 s (semillas 31-32): 1.09 (cumple por poco).

## Lectura
- La hipótesis H-FINAL **no queda respaldada** en el converso. Los pesos de las patas (b de 0.1-0.2) no hicieron un cambio de «segundo orden».
- **Hallazgo mayor:** con las semillas 31-32 la ventaja de E sobre B en el converso deja de verse (n = 5: medias iguales, E gana en 2 de 5). El resultado «E: 0.198 → 0.053» del bloque de 3 semillas depende de cuáles semillas se miren. La dispersión por semilla de todos los brazos en el converso (0.03-0.52) es del tamaño de las diferencias entre brazos.
- Hardware mixto (T4, H100, A100) y n pequeño: se reporta sin p-valores.
- Pendiente: M converso 31-32 (2 corridas, ≈ 3.5-4 USD en A100 con 2 procesos por GPU).
