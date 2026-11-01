# Ronda 1-C, bloque P (Burgers, semillas 11–20) — contraste de cond(P) — 2026-10-22

Kernel `rodrimandub/af-pikans-exp-0023-p-block`, T4×2, 1 220 s, 10/10 trabajos ok, 20 corridas (B y E). Análisis: `round1C/analyze_round1c.py` sin cambios.

## Contraste preregistrado (único)
Spearman entre cond(P) inicial de la capa 1 y log(razón E/B), n = 10, **una cola, dirección esperada negativa**.
- Resultado: **ρ = +0.59** (p de dos colas = 0.074). En la dirección preregistrada (negativa) el p de una cola es ≈ 0.96.
- **Veredicto: NO respaldado** (signo contrario al esperado). El predictor cond(P) inicial **no** anticipa cuándo E ayuda en Burgers.
- Observación que debilita cualquier lectura: cond(P) inicial de la capa 1 casi no varía entre semillas (90.4 a 94.0). Con tan poca varianza, el predictor no tiene dónde discriminar; el ρ positivo no es evidencia de un efecto contrario sino de una señal casi constante.

## E frente a B en Burgers (descriptivo)
Razón E/B por semilla (11–20): 5.14, 0.19, 0.69, 0.03, 0.15, 2.28, 1.18, 0.04, 0.51, 0.25. Mediana **0.383**, geomedia 0.365, mejora **7/10**, máximo 5.13; dos semillas > 2×. Otra muestra más con cola pesada y mediana favorable: junto con el `aG` fijo de EXP-0025 (n = 5) la mediana agrupada es 0.260 con 12 de 15 mejoras y 2 por encima de 2×. En 1-B V (E empeoró, mediana 5.70) fue la excepción.

## Consecuencia
El mecanismo condicional por cond(P) inicial queda **sin respaldo** (decisión abierta 4 de la propuesta de 1-C: no hay umbral que fijar). La hipótesis de trabajo restante para Burgers es variabilidad por semilla con efecto central favorable a E y cola pesada; EXP-0026 contrasta cadencia y recálculo al arrancar L-BFGS.
