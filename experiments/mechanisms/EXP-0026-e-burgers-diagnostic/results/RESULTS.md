# EXP-0026 v1.1 — resultados — 2026-10-22

Kernel `rodrimandub/af-pikans-exp-0026-diag`, T4×2, 6 700 s. **Ledger incompleto:** 27 trabajos ok y 4 `not_started_cutoff` (el tope de despacho de 1.75 h cayó antes de despachar las últimas semillas de Allen–Cahn). Burgers (factorial 3001–3010, réplica cruzada, lectura de R) quedó completo; **Allen–Cahn solo tiene 6 de 10 semillas** (ausencias, no se imputan). Análisis: `analyze_diag.py`, `analysis.json`.

## Burgers, factorial (10 semillas, pareado)
| Contraste (preregistrado) | Mediana | Mejora | Veredicto |
|---|---|---|---|
| H-DX-1: e(E1000)/e(E1000L), quitar el recálculo al arrancar L-BFGS | 1.036 | 3/10 | **No respaldada** |
| H-DX-2: e(E500)/e(E1000), cadencia 500 | 0.706 | 6/10 | **No respaldada** (mediana ≤ 0.90 pero 6 < 7) |

Frente a B: E1000L 0.212 (8/10, 2 semillas > 2×, máx 6.47); E1000 0.204 (8/10, 2 > 2×, máx 6.91); **E500L 0.166 (10/10, máx 0.83) y E500 0.203 (10/10, máx 0.95), ambos «estables»** según el criterio preregistrado; los de cadencia 1 000 no. Descriptivo: e(E500L)/e(E1000L) 0.766; e(E500L)/e(E500) 0.898.
**Allen–Cahn (n = 6, no evaluable con la regla de n = 10):** todos los brazos E dan ≈ 0.10–0.13 de B; E500/E1000 = 1.10 y E1000/E1000L = 1.01: cadencia y recálculo no importan ahí.

## Réplica cruzada (v1.1-A, Burgers, semillas de 1-B)
- **Reproduce V:** E1000L en 101–105 da mediana **3.13** (1/5 mejora) frente a la de 1-B V (5.70, 1/5). **El fallo de V no es ruido de corrida: sigue a las semillas con el código de 1-B.**
- **Reproduce T1:** E1000L en 1–5 da mediana 0.23 (T1 de 1-B: 0.43).
- **E500 en las semillas de V:** mediana 0.347, 4/5 mejores (pero la semilla 105 empeora 21.6×).
- En 1–5, E500 es peor que E1000L (0.665 frente a 0.23; semilla 1: 12.8×).

## Lectura (exploratoria, post-hoc, no preregistrada)
Agrupando las 20 semillas de Burgers: semillas con razón > 2× frente a B: E1000L 7/20 y E500 2/20 (Fisher p = 0.13, no significativo); E500 gana a E1000L en 11/20 (mediana pareada 0.896). Hay semillas malas para E con ambas cadencias (1 y 105). Es compatible con: (i) el recálculo al arrancar L-BFGS **no** es el factor (H-DX-1 refutada); (ii) la cadencia 500 reduce parte de la cola pero no de forma consistente entre conjuntos de semillas (efecto semilla × configuración); (iii) la cola pesada por semilla es una propiedad del problema y del método, no de una configuración concreta. No hay mecanismo establecido que explique las semillas malas.

## Lectura de R (v1.1-D, solo lectura, semillas 3001–3003)
El pool remuestreado por `RadLever` tiene 3 701–3 777 puntos únicos de 8 192 (≈ 45 %), `effective_candidates` 74–559, 44–46 % de la masa en el 5 % de candidatos de mayor residuo, y **36–46 % de los puntos en la banda del choque (|x| < 0.1) frente a ≈ 10 % uniforme**. Es decir, hay **ambas cosas**: duplicados (≈ 55 % repetidos) y concentración fuerte en el choque. Con estos datos no se separa cuál causa el 33× de R en Burgers. El pool es de 8 192 puntos, no de 9 200.
