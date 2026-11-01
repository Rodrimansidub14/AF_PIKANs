# Ronda 1-C, bloque V (semillas 106–110) y veredicto de la ronda — 2026-10-22

Kernel `rodrimandub/af-pikans-exp-0023-c1-v-block`, 2×T4, 1 660 s, 10/10 trabajos ok. Mismo análisis preregistrado, sin cambios.

| Problema | Brazo | V mediana | V mejora | T1+V mediana (10 sem.) |
|---|---|---|---|---|
| Allen–Cahn d=2e-4 | E | 0.133 | 5/5 | 0.306 (10/10) |
| | F | 0.457 | 5/5 (máx 0.90) | 0.514 (10/10) |
| Fisher–KPP (control) | E | 2.722 | 1/5 | — |
| | F | 2.522 | 1/5 | — |

## Veredicto contra las reglas preregistradas
- **Refutación de (a):** no se cumple. E y F pasan T1 (mediana ≤ 0.90) **y** V (mejora en ≥ 4 de 5; F con máx 0.90 pasa justo). Alcance a la familia Allen–Cahn **no refutado**.
- **Desempate (d), E frente a F en Allen–Cahn d=2e-4:** T1 mediana 0.795, E gana 5/5; V mediana 0.29, E gana 5/5; agrupado 0.57, E gana 10/10 ⇒ **variante E** (condición 2 de EXP-0025 cumplida; `aG` = variante E).
- **Control KPP (descriptivo):** E y F **empeoran** en T1 y en V (medianas 5.3/3.1 y 2.7/2.5; mejoran 1 de 5 en ambos). La predicción «ganancia menor o nula» queda satisfecha con creces: el efecto es negativo.
- **Predicción de dosis (f):** **no respaldada.** Mediana de E agrupada en d=2e-4 = 0.306 frente a ≈ 0.14 en d=1e-3: la ganancia es menor, no mayor, al afinar el frente (el B de d=2e-4 ya tiene un error alto, ≈ 0.07, y quizá el frente está mal resuelto con 8 000 puntos y ancho 16; no se probó). Sin prueba, descriptivo.
- **Región (descriptivo, relevante para H-OBS-3 de EXP-0025):** E reduce más el error **fuera** de la interfaz que en ella (V: 0.209 interfaz frente a 0.118 resto; T1: 0.41 frente a 0.35). F casi no toca la interfaz (0.83–0.94). Es decir, ni E ni F «focalizan» la interfaz.
- **cond(P) (e):** el contraste preregistrado es en Burgers (bloque P) y **no se corrió**. Spearman descriptivo n=5 por problema: ρ entre −0.8 y +0.2, ninguno con p < 0.10. Nada que reportar como evidencia.

## Qué cambia para el camino al mecanismo
1. Hay un efecto grande y replicado en AC (E: 0.31 agrupado) y uno **negativo** en KPP y, por 1-B, en Burgers. El efecto depende del problema, que es la premisa de un selector (EXP-0025).
2. El predictor de cuándo E ayuda queda sin respaldo: cond(P) inicial no discrimina en AC ni KPP; el contraste definitivo (Burgers) sigue sin correrse.
3. La ganancia de E no es espacialmente focal: no se debe llamar «focalización» a E.
