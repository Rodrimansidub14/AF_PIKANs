---
type: frontier
tags: #frontier #arch #thesis-direction
---

# Causal Physics-Informed Attention

> Máscara de atención gobernada por la condición CFL de la PDE. Fuerza la atención a respetar el cono de luz de la onda — no hay información del futuro influyendo en el pasado.

## Analogía
Causal masking de GPT, pero donde la máscara la dicta la física (velocidad máxima de onda), no la posición en secuencia.

## Gap específico
Wang 2024 introduce causalidad en la pérdida. Nadie la ha introducido en la arquitectura de atención directamente.

## Papers de soporte
- [[Wang_2024_CausalPINNs]]
- [[Vaswani_2017_Attention]]
- [[AC_PKAN_2025]]

## Conceptos relacionados
*(ninguno)*

## Notas de desarrollo
*(vacío)*
