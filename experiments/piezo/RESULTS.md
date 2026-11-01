# Piezoelectricidad (bimorfo PVDF, formulación indirecta) — estado (EXP-0009)

Exploratorio. `scientific_accepted=false`. Referencia de trabajo: FEM de Daniel González con `poling_sign=−1` (FEM del mismo grupo; falta contraste de terceros).

**Resumen honesto.** Con nuestra propia formulación mixta de 8 campos (reimplementación propia), ningún brazo RBF-PIKAN salió de la solución nula en 12+ variantes; el MLP solo aprendió la flexión
con pesos adaptativos + separación de escalas de v (u ≈ v ≈ 0.19 al final de Adam, una semilla) y el L-BFGS siempre destruyó la solución. El mismo fallo está documentado por el propio código de Daniel
(`scaling.py`: "the trivial field u = v = 0 is the only field that pays nothing"). Hallazgos: la convención de signos físicamente consistente es e31 y e33 de signo opuesto (los residuos de equilibrio del FEM lo confirman);
la tabla `tab:pvdf` de la metodología coincide con la inversión de e33 del cargador de Daniel, no con la física.

**Con la receta del paper de Daniel** (commit e7856f1, 2026-10-02: 20 000 Adam, float64, formulación por capas con interfaz, SiLU, escalas por campo, remuestreo cada 200 épocas, sin L-BFGS) ejecutada con su MLP y con un tronco RBF-PIKAN
intercambiado (`daniel_arm.py`), semilla del paper 20260728, evaluador propio contra el FEM:

| Brazo (n=5 semillas: 20260728–32) | Parámetros | u | v | φ | Tiempo por corrida |
|---|---|---|---|---|---|
| MLP 3×50 SiLU | 5 758 | 0.136 ± 0.007 | 0.186 ± 0.011 | 0.046 ± 0.000 | ~4 000 s |
| RBF-PIKAN 16 | 5 376 | **0.092 ± 0.021** | **0.104 ± 0.048** | 0.048 ± 0.003 | ~9 400 s |

Diferencia pareada RBF − MLP: u −0.054, −0.003, −0.073, −0.036, −0.055; v −0.104, −0.006, −0.152, −0.083, −0.066 (el RBF gana 5/5 en ambas). Por semilla en `daniel-recipe-per-seed.csv`.

Con la receta correcta el RBF aprende la flexión y queda por debajo del MLP en u y v en las 5 semillas, con mayor dispersión y ~2.4× el costo; la trayectoria del RBF aún mejora al final de las 20 000 épocas (no convergido). El fallo de nuestra reimplementación propia era de formulación (capas con interfaz, escalas por campo, enrutamiento de gradiente), no del RBF.

Detalle completo, tablas por variante y diagnóstico: `EXP-0009-piezo-indirect-rbf/spec.md`. Registro de fallos: `registry/failures.jsonl` (`piezo-null-state-20261003`).
