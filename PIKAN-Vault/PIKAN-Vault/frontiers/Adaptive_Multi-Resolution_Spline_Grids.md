---
type: frontier
tags: #frontier #arch #thesis-direction
---

# Adaptive Multi-Resolution Spline Grids

> Dinamizar la grilla de splines de KAN durante entrenamiento, no solo post-hoc. Densificar donde el gradiente de la solución es alto (frentes de onda, singularidades).

## Analogía
Adaptive Mesh Refinement (AMR) de FEM pero aplicado a las grillas internas del KAN.

## Gap específico
Grid extension de Liu 2025 es post-hoc y manual. Ningún paper hace esto de forma diferenciable y end-to-end durante entrenamiento.

## Papers de soporte
- [[Liu_2025_KAN]]
- [[Lu_2021_DeepXDE]]
- [[Kharazmi_2021_hpVPINNs]]

## Conceptos relacionados
*(ninguno)*

## Notas de desarrollo
*(vacío)*
