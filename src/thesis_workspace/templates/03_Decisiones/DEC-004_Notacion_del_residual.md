---
kind: adviser_decision
id: DEC-004
status: needs_adviser
priority: low
---

# DEC-004 — Notación del residual

## Alternativas

El borrador puede usar el residual con forzamiento explícito:

N[u](x) - f(x)

o absorber el forzamiento dentro del operador, como en el protocolo actual.
Ambas convenciones son posibles; el problema operativo es mantener una sola.

## Recomendación para discusión

Usar la forma explícita:

r(x;theta) = N[u_theta](x) - f(x)

porque hace visible el forzamiento y permite explicar el control de Poisson
oscilatorio como un cambio en f, no en el operador.

## Gate

decisión del asesor: ______________________________

convención aprobada: explícita / absorbida / otra

## Superficie afectada

Revisar 6.1, la definición de residual de 6.3, el puente de 6.5, la definición
del mecanismo en 6.6, el banco de 6.7 y cualquier sección de metodología que
reutilice la notación.
