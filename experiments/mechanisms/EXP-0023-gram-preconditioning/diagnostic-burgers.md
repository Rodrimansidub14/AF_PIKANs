# EXP-0023 — diagnóstico post-hoc de Burgers (2026-10-21) — EXPLORATORIO

Sin cómputo nuevo: `diag_burgers.py` sobre los 20 JSON de Burgers de T1 (semillas 1–5) y V (101–105); salida en `results/diag_burgers.json`. n = 10 semillas por brazo; correlaciones de Spearman con la log-razón final (brazo/B); p-valores descriptivos y **sin corrección por multiplicidad** (9 pruebas en E, 6 en F).

## Qué se podía medir con los logs
Error de B, error de fin de Adam, pérdida de entrenamiento en el paso 2000, cond(P) de E (en el paso 0 y al arrancar L-BFGS) y la densidad de puntos de colocación en el frente (el pool de cada semilla es determinista, se recalculó). **No hay checkpoints**: no existe la Gram de las activaciones entrenadas ni el error por región (el JSON solo trae L2 global y máximo), así que «nitidez del frente» se aproximó por densidad de puntos en el frente, que casi no varía.

## Resultado
- Las semillas que empeoran (E: 1, 102, 103, 104, 105; razones 4.2–11.1) **no comparten ninguna característica medible** entre las disponibles:
  - puntos en el frente (|x| ≤ 5ν, t ≥ 1/π): 87–90 en las 10 semillas (ρ = +0.23, p = 0.52); cerca del frente 572–576 (ρ = −0.51, p = 0.13);
  - error de B (denominador): ρ = −0.22, p = 0.53 (las malas tienen B pequeño y grande: 0.0013 a 0.0075);
  - razón al final de Adam: ρ = −0.21, p = 0.56; razón de la pérdida en el paso 2000: ρ = +0.08, p = 0.83.
- Para F: ninguna correlación con p < 0.15 (error de B ρ = −0.49, p = 0.15; fin de Adam ρ = +0.26; pérdida ρ = −0.20; frente ρ = +0.16).
- Único hallazgo nominal: **cond(P) de la capa 1 en el paso 0 (E): ρ = −0.72, p = 0.019** (a mayor condicionamiento inicial, mejor razón). Entre 15 pruebas se esperan ≈ 0.75 falsos positivos al 0.05 y n = 10, así que **no lo trato como hipótesis**, solo como pista a verificar si se vuelve a medir con una cantidad preregistrada. (La correlación con el error final del brazo es tautológica: es el numerador de la razón.)

## Conclusión
Los logs no sostienen una hipótesis sobre por qué E empeora en Burgers. La variabilidad parece un componente por semilla no explicado por lo medido (coherente con la sensibilidad numérica documentada en el EXP-0024). Para que una ronda siguiente pueda medir algo, habría que **guardar checkpoints y errores por región (frente / resto)** en las corridas nuevas y preregistrar el predictor.
