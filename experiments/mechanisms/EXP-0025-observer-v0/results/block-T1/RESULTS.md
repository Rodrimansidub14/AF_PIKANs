# EXP-0025 — bloque T1 (semillas 2001–2005) — puntuado el 2026-10-22

Kernel `rodrimandub/af-pikans-exp-0025-block-t1`, T4×2, 3 540 s de pared, 20/20 trabajos ok, 60 corridas (B, R, E, O, O−, O−m × 5 semillas × {Allen–Cahn d=1e-3, Burgers}). O−m corrió en ambos problemas (O eligió a0 en 87.5 % y 80 % de las decisiones). Análisis: `analyze_block.py` (aplica el spec literalmente), `analysis.json`.
**Nota de tope:** el spec fija 1.5 h por bloque; el bloque tardó 59 min en total, dentro del tope.

Razón mediana error(X)/error(B), pareada por semilla:

| Problema | R | E (fixed:aG) | O | O− (uniforme) | O−m |
|---|---|---|---|---|---|
| Allen–Cahn d=1e-3 | 0.654 | **0.204** | 0.605 | 0.216 | 1.000 |
| Burgers | 33.35 | **0.260** | 2.498 | 63.5 | 1.000 |

## Veredictos preregistrados
- **H-OBS-1 (no arrepentimiento): RECHAZADA.** Allen–Cahn: r_O = 0.605 > min(1, 0.654, 0.204) + 0.10 = 0.304. Burgers: r_O = 2.50 > 0.36; además una semilla de O queda en 5.81× B (guarda de cola, 2×) y el error de O en la interfaz llega a 5.85× el de B (guarda de región, 1.25×). Sí se cumple r_O ≤ 0.90 en un problema (Allen–Cahn), pero no basta.
- **H-OBS-2 (el sondeo discrimina): NO RECHAZADA formalmente, con reserva.** El comparador por regla es O−m (O eligió a0 > 50 %): en Allen–Cahn r_O = 0.605 ≤ 0.90·1.000. Pero O−m es casi B (en 3 de 5 semillas de AC y de Burgers eligió solo a0: razón 1.000), así que el comparador es débil. Frente al O− uniforme, O **pierde** en ambos problemas (0.605 frente a 0.216 en AC; 2.50 frente a 63.5 en Burgers solo gana porque O− incluye aR, catastrófico). El resultado es de que el sondeo no elige mejor que el azar útil; no se debe citar H-OBS-2 como éxito.
- **H-OBS-3 (lectura espacial): NO RESPALDADA.** Condición «interfaz − resto ≥ 0.10» en 1/5 (AC) y 0/5 (Burgers) semillas.
- **Guardas:** reloj de O 1.27× y 1.29× el de B (≤ 1.5 ✓); error de Adam O/B media geométrica 0.45 y 0.77 (✓); pérdida finita 5/5 (✓); región: **falla en Burgers**; cola: **falla en Burgers**.

## Qué se ve (descriptivo, sin cambiar el veredicto)
1. **O queda mayormente en a0**: 35 de 40 decisiones en AC y 16 de 20 en Burgers. Cuando elige aG en la decisión 0–2 gana (AC semilla 2001: 0.049 frente a B); cuando no, es B (≈ 1.0). La guarda γ y el sesgo de horizonte corto (Tier 1) explican por qué pocas veces cambia.
2. **La palanca E dentro del observador (aG en cada decisión, K = 500, sin recálculo extra al arrancar L-BFGS) mejora a B en ambos problemas** (AC 0.204, Burgers 0.260; en Burgers 4 de 5 semillas < 0.55, una con L-BFGS detenido a las 5 iteraciones). **Esto contradice 1-B**, donde E (cadencia 1 000, con recálculo al arrancar L-BFGS) empeoró en Burgers (mediana 5.70, 1 de 5). El spec declara el E de EXP-0023 solo verificación cruzada; esta discrepancia es la pista más fuerte del bloque hacia un mecanismo, y **no está explicada**: difieren la cadencia, el recálculo al inicio de L-BFGS y el camino de código.
3. **R (aR) es catastrófico en Burgers (33×) y malo para el azar uniforme que lo incluye (63×).** Sospecha de implementación, no probada con estos datos: `RadLever` muestrea con reemplazo 9 200 puntos desde 8 192 candidatos (`rad.sample_rad`, `replace=True`), lo que repite puntos y pierde cobertura. No se modifica el comparador R (fijado por el spec); queda anotado.
4. Sin el brazo KPP el bloque no prueba «ninguna palanca fija gana en todos»: en estos dos problemas E gana en ambos.
