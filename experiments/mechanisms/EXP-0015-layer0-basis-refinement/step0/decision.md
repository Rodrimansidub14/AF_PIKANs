# Decisión pendiente — bloque GPU del EXP-0015 (tras el paso 0, n = 3)

Preparado por un agente el 2026-10-18. **No decide el agente**; la decisión es del investigador (spec, enmienda del paso 0).

## Evidencia (resumen; detalle en `results.md`)

| Problema | PINN / ajuste C | O11 | U41 | Lectura según la tabla de la enmienda |
|---|---|---|---|---|
| DFG | 34.6× | mejora (0.47, 3/3) | mejora (0.38, 3/3) | ≥ 10×: la representación no limita; bloque GPU de DFG se recorta |
| Burgers | 0.14× | empeora (1.23) | sin efecto (1.07) | ajuste no concluyente sobre techo; **H-BASIS-2 no se sostiene** |
| Heat (control −) | 0.01× | no (0.76, 1/3) | **mejora (0.44, 3/3)** | el efecto de más bases es en parte capacidad |
| NS2d-CG, Allen–Cahn, piezo | 22×, 7×, 274× (n = 1) | — | — | sin confirmar con n = 3 |

## Opciones

1. **Cancelar el bloque GPU de Burgers/Allen–Cahn/Heat del EXP-0015 y pasar a la rama A del EXP-0016** (el error de los PINN es de optimización: traspaso Adam→L-BFGS decidido por la red, o condicionamiento por capa). Es lo que apunta la evidencia: H-BASIS-2 no se sostiene donde estaba la hipótesis primaria y el PINN supera al ajuste en 4 de 6 problemas.
2. **Reducir el bloque GPU a Burgers (4 brazos × 5 semillas, ≈ 2.8 h)** solo para cerrar H-BASIS-1/2/3 con la receta PINN real. Costo: ≈ 2.8 h de GPU para una hipótesis que el paso 0 ya no favorece.
3. **Mantener el bloque completo (≈ 7.2 h).** No recomendado.

Rama B (bases en DFG) solo con un control de parámetros igualados, porque U41 también mejora en Heat y en DFG el PINN está 35× por encima del ajuste.

- [x] Decisión del investigador: **opción 1** — se cancela el bloque GPU del EXP-0015 (Burgers, Allen–Cahn y Heat) y se pasa a la rama A del EXP-0016 — fecha: 2026-10-18 (chat: "Cancela el bloque GPU del EXP-0015"). El EXP-0015 queda cerrado en el paso 0 (CPU, n = 3); H-BASIS-1/2/3 no se prueban con la receta PINN. Rama B (bases en DFG) queda abierta solo con control de parámetros igualados.


> **Nota de terminología (agente, 2026-10-19; no cambia la decisión).** En este documento "PINN" se refiere a la RBF-PIKAN entrenada con la pérdida física (PIKAN-PDE), no al MLP. Todos los valores de la columna "PINN / ajuste C" son de la RBF-PIKAN.
