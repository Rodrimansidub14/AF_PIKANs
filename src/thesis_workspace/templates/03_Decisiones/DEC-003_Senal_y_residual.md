---
kind: adviser_decision
id: DEC-003
status: needs_adviser
priority: medium
---

# DEC-003 — Señal y residual

## Distinción que debe quedar escrita

Una propiedad del problema, como curvatura u oscilación, no es automáticamente
una cantidad observable disponible para el mecanismo. La señal debe definirse
y justificarse como observable del modelo o de la solución.

Para una PDE de segundo orden, el residual puede contener ya la segunda
derivada. Por eso hay riesgo de colinealidad entre curvatura y residual. La
pregunta no es solo si la señal correlaciona con dificultad, sino qué
información incremental aporta frente al residual en un régimen donde el
residual sea bajo y la solución siga siendo difícil.

## Gate

decisión del asesor: ______________________________

señal aprobada: ___________________________________

información incremental que debe probarse: __________

## Implicaciones de redacción y prueba

- 6.4 debe hablar de propiedades matemáticas, no presentar una propiedad como
  si ya fuera una señal.
- 6.6.3 debe comparar cada candidata con el residual.
- La prueba debe registrar la relación entre la señal, el residual y el
  comportamiento de la solución.
