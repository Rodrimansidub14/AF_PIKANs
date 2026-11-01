# EXP-0015 · Paso 0 — ajuste supervisado de la referencia (resultados, semilla 1)

> **Terminología (2026-10-19).** "PIKAN-PDE" = la RBF-PIKAN entrenada con la pérdida física (el sistema de la tesis). No es el MLP: el MLP (PINN clásico) solo aparece como comparador en los baselines.


Exploratorio. `scientific_accepted=false`. Corrido el 2026-10-18 en CPU (contenedor en la nube, 2 núcleos), por un agente, bajo la enmienda firmada del spec (`../spec.md`, "Enmienda firmada el 2026-10-18").

**Pregunta.** ¿La arquitectura de control *puede representar* la solución con el error que reporta la PIKAN-PDE? Se ajusta cada red directamente a la referencia (MSE, sin término físico). La referencia se usa solo como objetivo de regresión: es un diagnóstico, no un mecanismo.

**Receta (igual para todos los brazos de un problema).** float64; 8 000 puntos de entrenamiento por semilla (mitad uniformes, mitad proporcionales a |∇f| de la referencia, el mismo conjunto para todos los brazos); 2 000 pasos de Adam (lr 1e-3) + 2 000 iteraciones de L-BFGS strong-Wolfe con `tolerance_grad = tolerance_change = 0`; L2 relativo evaluado en la malla completa de la referencia. Brazos de la capa 0: C (11 uniformes), U41 (41 uniformes), O11 (11 por equidistribución del monitor de la referencia, piso 0.3), X11 (espaciado de O11 desplazado; solo Burgers y Allen–Cahn). Arquitecturas = las de control de baseline v2 (Heat RBF16 con centros [−1.05, 1.05] y σ 0.21; Burgers RBF48; AC RBF16; NS RBF56; DFG y piezo RBF16 con 2 entradas).

**Incidencia corregida.** Una primera pasada (`out2`, descartada) dejaba L-BFGS detenido a ~1 000 iteraciones por las tolerancias por defecto de PyTorch (pérdida ~1e-6 < `tolerance_change` 1e-9 relativa). Los números de abajo son de la pasada corregida.

## Resultados (semilla 1)

| Problema | Ajuste C (L2) | PIKAN-PDE baseline (L2) | PIKAN-PDE / ajuste | O11 / C | U41 / C | X11 / C |
|---|---|---|---|---|---|---|
| Piezo (u) | 3.36e-4 | 0.092 | 274× | 1.09 | 1.09 | — |
| DFG Re=20 (u) | 2.36e-3 | 0.070 | 30× | **0.43** | **0.32** | — |
| NS2d-CG (u) | 1.90e-3 | 0.041 (semilla 701) | 22× | 0.89 | **0.60** | — |
| Allen–Cahn | 3.85e-4 | 2.69e-3 | 7× | 1.10 | 0.88 | 1.00 |
| Burgers | 5.28e-4 (aún bajando) | 7.16e-5 | 0.14× | 1.42 | 1.11 | 1.11 |
| Heat | 1.26e-4 | 1.86e-6 | 0.015× | 1.34 | **0.68** | — |

Por componente (v, p/φ) y curvas de error contra iteración: `results/<problema>-<brazo>-seed1.json`.

## Lectura

1. **La capacidad no limita en piezo, DFG, NS2d-CG y Allen–Cahn**: el ajuste supervisado llega a un error entre 7 y 274 veces menor que el de la PIKAN-PDE. El error de la PIKAN-PDE ahí es de optimización o formulación.
2. **Burgers y Heat no son concluyentes** sobre el techo: la PIKAN-PDE tuvo ~10× más presupuesto (Burgers: 24 000 Adam + 1 800 s L-BFGS f64, 16 384 puntos), y en Burgers el ajuste todavía bajaba.
3. **Ubicar las bases (O11) solo ayuda en DFG**; en Burgers (hipótesis primaria H-BASIS-2) empeora 1.42×.
4. **Más bases (U41) ayudan en DFG, NS2d-CG y Heat**. Heat es el control negativo: parte del efecto es capacidad general (U41 tiene +8 % a +26 % de parámetros), no resolución local.
5. Una semilla por brazo: nada de esto es decisión todavía (ver `../../../../docs/plans/2026-10-18-handoff-paso0-exp0016.md`).

## Reproducir

```
cd experiments/mechanisms/EXP-0015-layer0-basis-refinement/step0
python fit_step0.py --problem burgers --arms C,O11,U41,X11 --seeds 2,3 --adam 2000 --lbfgs 2000 --ntrain 8000 --out results
```
`AF_ROOT` es opcional (por defecto, la raíz del repo). Problemas: `heat`, `burgers`, `allen_cahn`, `ns2d_cg`, `dfg`, `piezo`. Un JSON ya existente se salta. Prueba de equivalencia de la capa 0 contra `src/sciml/model.py`: `python equiv_test.py` (diferencias ~1e-16).

---

## Confirmación con n = 3 (semillas 1–3), 2026-10-18

Exploratorio. `scientific_accepted=false`. Mismas receta y semilla 1 de arriba; semillas 2 y 3 corridas en la CPU local del investigador (4 núcleos físicos, 2 hilos por proceso; 17–51 min por corrida), no en el contenedor de la semilla 1. Equivalencia de capa 0 verificada de nuevo antes de correr (`equiv_test.py`, diferencias ~1e-16). Solo se repitieron los tres problemas donde los brazos se separaban (DFG, Burgers, Heat); NS2d-CG, Allen–Cahn y piezo quedan en n = 1. Criterio del spec: mejora si la media geométrica de la razón ≤ 0.80 y ≥ 2/3 semillas ≤ 0.80. Tabla generada por `table_n3.py` (`table_n3.md`).

| Problema | Ajuste C (media geom.) | PIKAN-PDE | PIKAN-PDE / ajuste | O11/C por semilla [media geom.] | U41/C por semilla [media geom.] | X11/C por semilla [media geom.] |
|---|---|---|---|---|---|---|
| DFG (u) | 2.02e-3 | 7.00e-2 | 34.6× | 0.43 / 0.47 / 0.52 [0.47] **sostiene** | 0.32 / 0.46 / 0.38 [0.38] **sostiene** | — |
| Burgers | 5.30e-4 | 7.16e-5 | 0.14× | 1.42 / 1.14 / 1.14 [1.23] no | 1.11 / 0.96 / 1.14 [1.07] no | 1.11 / 0.99 / 0.97 [1.02] no |
| Heat | 1.88e-4 | 1.86e-6 | 0.01× | 1.34 / 1.14 / 0.29 [0.76] no (1/3) | 0.68 / 0.58 / 0.22 [0.44] **sostiene** | — |

### Lectura con n = 3

1. **DFG:** O11 y U41 mejoran de forma consistente (3/3 semillas, media geométrica 0.47 y 0.38). Pero la PIKAN-PDE está 35× por encima del ajuste de C: según la tabla de la enmienda, la representación no limita y el bloque GPU de DFG se recorta. La mejora del ajuste no se traslada a explicar el error de la PIKAN-PDE.
2. **Burgers:** O11/C = 1.23 (empeora en las 3 semillas), U41 = 1.07 y X11 = 1.02. **H-BASIS-2 (primaria) no se sostiene en Burgers** en este diagnóstico. Salvedad: los cuatro brazos terminan cerca de 5e-4 y el ajuste es peor que la PIKAN-PDE (0.14×, con ~10× menos presupuesto), así que el ajuste está limitado por optimización y no mide un techo de representación.
3. **Heat (control negativo): U41 mejora 0.44 en 3/3 semillas.** Parte del efecto de más bases es capacidad general (+8 % a +26 % de parámetros), no resolución local. Cualquier brazo futuro de "más bases" necesita un control con parámetros igualados. O11 en Heat no se sostiene (1/3); la media 0.76 viene de una semilla 3 atípica de C (4.18e-4 frente a 1.26e-4 en las otras), que infla a favor la razón de los otros brazos en esa semilla.
4. **Ruido:** con n = 3 y CV alto entre semillas (p. ej. C en Heat), solo los efectos de DFG y U41-Heat son claros; las razones cercanas a 1 en Burgers no distinguen entre brazos.

Decisión sobre el bloque GPU: ver `decision.md`. Réplica en Kaggle (CPU, 20 trabajos) en curso como contraste: `rodrimandub/af-pikans-step0-confirm`; sus JSON no están incorporados a estas tablas.
