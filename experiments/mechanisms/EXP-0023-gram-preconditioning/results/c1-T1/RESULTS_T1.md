# Ronda 1-C, bloque T1 (semillas 6–10) — puntuado el 2026-10-22

Kernel `rodrimandub/af-pikans-exp-0023-c1-block`, 2×T4, 1 660 s de pared, 10/10 trabajos ok, 60 salidas. Análisis: `round1C/analyze_round1c.py` (sin cambios), `analysis_*.json`.
Receta: Adam 4000, L-BFGS ≤ 500, RBF ancho 16. Razón = error L2 relativo final de E o F sobre B, pareada por semilla.

| Problema | Brazo | Mediana | Geomedia | Mejora | Máx |
|---|---|---|---|---|---|
| Allen–Cahn d=2e-4 (réplica) | E | 0.355 | 0.362 | 5/5 | 0.47 |
| | F | 0.570 | 0.500 | 5/5 | 0.65 |
| Fisher–KPP (control) | E | 5.311 | 3.840 | 1/5 | 9.42 |
| | F | 3.133 | 3.966 | 1/5 | 34.03 |

- **Réplica (T1):** E y F pasan el umbral (mediana ≤ 0.90). Error de B ≈ 0.065–0.077; E 0.019–0.033.
- **Desempate (d), sobre T1:** E/F por semilla 0.795, 0.881, 0.434, 0.700, 0.937 → mediana 0.795 ≤ 0.90 y E gana 5/5 ⇒ E, **provisional hasta V**.
- **Región (descriptivo):** E reduce el error 0.41× en la interfaz y 0.35× fuera; F 0.94× y 0.42×. F mejora casi solo fuera de la interfaz.
- **Control KPP:** E y F **empeoran** (mediana 5.3× y 3.1×). La predicción preregistrada decía «ganancia menor o nula»; el resultado es peor que nulo. Coherente con el alcance a la familia AC y con 1-B (Burgers); es el caso para un selector.
- **Dosis (f):** mediana de E 0.355 en d=2e-4 frente a ≈ 0.14 en d=1e-3: la ganancia **no crece** al bajar d, contra la predicción de dosis. Descriptivo, sin prueba; el B de d=2e-4 tiene error alto (≈ 0.07).
- **cond(P) (e):** el contraste preregistrado es en Burgers (bloque P), no corrido. Spearman descriptivo en AC y KPP con n=5: sin señal (ρ ≈ −0.1, −0.2).
- Reloj por corrida: B 97 s, E 105 s, F 101 s (≈ +8 % y +4 % en T4).
