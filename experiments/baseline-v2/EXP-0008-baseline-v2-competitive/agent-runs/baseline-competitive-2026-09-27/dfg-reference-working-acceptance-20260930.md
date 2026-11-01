# DFG 2D-1 Re=20: FEM Taylor–Hood L3 como referencia de campo **de trabajo** — 2026-09-30

Decisión del investigador (chat, 2026-09-30): "Dale", con la misma vara que se aplicó a Allen–Cahn: referencia de terceros o anclada a terceros. Este archivo **registra** esa decisión; no es un claim ni toca `evals/`, `registry/claims.jsonl` ni umbrales. Aceptación para medir el error de campo de los PINN/PIKAN en el baseline exploratorio; **no** autoriza claims.

## Por qué cumple la vara

1. **Anclada a terceros (FEATFLOW):** el FEM L3 reproduce los valores publicados con Cd a 2.89e-5 relativo, Δp a 1.21e-4 relativo y Cl a 6.41e-8 absoluto (`local-20260930/force-formula-fem-validation.json` y `qois.json`).
2. **Convergencia del campo entre mallas (calculada el 2026-09-30 sobre la malla de puntuación 221×42, 9 203 puntos):** velocidad L1→L2 5.84e-5 y L2→L3 6.93e-6 (razón 8.4); presión L1→L2 2.28e-4 y L2→L3 5.53e-5 (razón 4.1). El error de discretización es unas 10³ veces menor que el error de campo de los modelos (7–9 % en u, v; ~25 % en p).
3. **Criterio de Cl cambiado a absoluto** (por decisión del investigador): el cambio L2→L3 de Cl es 7.06e-5 en absoluto, frente a la tolerancia 5e-4 usada contra FEATFLOW (margen ×7). El criterio relativo original (0.2 %) sobre una cantidad ≈0.0106 no es informativo.

## Condiciones que se mantienen

- La solución la generó un agente (DOLFINx 0.11.0 + Gmsh 4.15.2, P2/P1, Newton, salida do-nothing); su respaldo de terceros son los QoI de FEATFLOW, no un campo publicado. Se declara así en cualquier uso.
- Mi re-evaluación independiente de las fuerzas sobre el campo exportado valida Cd (0.38 %) y Δp (0.50 %), pero **no Cl** (error absoluto 0.007 por extrapolación de mi evaluador en el borde curvo). La validación de Cl descansa en el cálculo nativo de DOLFINx (6.4e-8).
- Presión: se compara con gauge fijado en el punto (2.2, 0.2) de la salida.
- Sigue siendo `DRAFT_UNVERIFIED` / `scientific_accepted=false` en los artefactos; cualquier claim exige el gate `scientific_claim`.
