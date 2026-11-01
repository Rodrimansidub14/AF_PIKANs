---
kind: adviser_decision
id: DEC-001
status: needs_adviser
priority: high
---

# DEC-001 — Punto de inyección

## Pregunta

¿El mecanismo interviene en la ponderación del residual, en el muestreo o en
la densidad de la grilla interna?

## Opciones

### A — Ponderación del residual

La afirmación defendible es redistribución del esfuerzo de aprendizaje. Es la
opción de menor riesgo de ejecución y es coherente con la ecuación que ya
aparece en el borrador, pero no equivale por sí sola a redistribuir capacidad.

### B — Modulación del muestreo

Es una variante RAR sobre PIKAN. Se conserva como baseline o fallback. La
afirmación de novedad queda más estrecha: estudiar la interacción entre
muestreo y arquitectura de soporte local.

### C — Densidad de la grilla interna

Es la lectura literal de redistribución de capacidad y la analogía más directa
con AMR. Tiene mayor riesgo de implementación. Antes de escogerla se debe
verificar en la fuente primaria de KAN si existe grid extension y si es
uniforme o programada.

## Gate

decisión del asesor: ______________________________

opción seleccionada: A / B / C

evidencia o condición adicional: ___________________

## Consecuencias que deben actualizarse después del gate

- vocabulario de 6.2, 6.5 y 6.6;
- formulación de la contribución y distancia frente a RAR;
- firma de T y punto de inyección;
- controles y predicciones del banco de 6.7;
- presupuesto relativo de la sección 6.2.
