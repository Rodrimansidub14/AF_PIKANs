# EXP-0023 / EXP-0024 — cierre de la ronda 1-B (2026-10-21)

## Decisión del investigador (chat, 2026-10-21)
«Sí, aplica la regla de salida a la ronda 1-B, pero acota qué cierras. Cierras el avance a C, no la búsqueda en Allen-Cahn. No he leído el texto de la regla, así que lo siguiente parte de cómo la describes.»

## Texto de la regla firmada (EXP-0024, `T1-plan.md`, 2026-10-20 22:01) y cómo se aplica
Dice: «si la ronda 1-B (precondicionadores) **no pasa T1** con el criterio ya firmado (mediana ≤ 0.90 en ambos problemas), la búsqueda de mecanismo en Burgers/Allen-Cahn termina».
- **T1 sí se pasó** (E y F), así que el disparador literal no ocurrió; lo que falló fue **V** (E: Burgers 5.70, 1/5; F: Burgers 3/5), un caso que la regla no nombra. Se registra como **decisión del investigador**, no como disparo automático de la regla.
- Alcance acotado por el investigador: **se cierra el avance a C** de E y F (no hay finalista de la ronda 1-B). **No se cierra la búsqueda en Allen-Cahn.** Esto estrecha el alcance de la regla firmada («Burgers/Allen-Cahn»); queda anotado aquí para que no se lea como que la regla se aplicó tal cual.

## Para la discusión (mismo peso para el resultado negativo y el positivo)
- **Positivo (Allen–Cahn):** el precondicionamiento por bloque de arista (E) reduce el error final ≈ 7× frente a B (razón mediana 0.136 en T1 y 0.140 en V, 5/5 y 5/5, 10 semillas en dos bloques disjuntos, con reloj +8–10 %); la Gram fija de la base (F) da 0.44 y 0.39 (5/5 y 5/5). J (solo Jacobi) no ayuda de forma consistente (0.884, 3/5 en T1).
- **Negativo (Burgers):** el efecto de E **no replica**: mediana 0.43 (4/5) en T1 y 5.70 (1/5) en V; en 5 de 10 semillas empeora (hasta 11×). F mantiene la mediana (0.49 en ambos bloques) pero mejora solo 4/5 y 3/5, con semillas peores que B. La dependencia del problema es una conclusión del estudio, no una nota al pie.
- **Sin explicación medida:** el diagnóstico post-hoc (`diagnostic-burgers.md`) no encontró un predictor de las semillas malas entre lo disponible en los logs.
- **E frente a F:** en T1/V no se pueden separar de forma general (en Allen–Cahn E es ≈ 3× mejor en error; en Burgers F es más estable). La ganancia de F, que no usa activaciones entrenadas, apunta a la corrección del solape de la base RBF como fuente principal, sin descartar que E añada algo en Allen–Cahn.
- No toqué los `.tex` de ProyectoGrad; estos párrafos están listos para llevarlos a la escritura si lo pedís.

## Estado
Ronda 1-B cerrada sin finalista. Cualquier nueva ronda (1-C) requiere hipótesis, problema de réplica, semillas y desempate fijados **antes** de correr: ver `round1C-proposal.md` (borrador, sin firma).
