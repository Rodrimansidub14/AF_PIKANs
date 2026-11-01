---
experiment_id: EXP-0012
slug: heat-rad-uniform-lbfgs
status: decided          # draft | preregistered | running | scored | decided
suite: tier2_candidate
problem_id: problem.heat_1d (u_t = 0.1 u_xx, EXP-0007 contract)
model_id: model.rbf_pikan (rbf_fixed, EXP-0008 control)
mechanism_id: mechanism.rad (src/mechanisms/rad.py, power 2, uniform_fraction 0.5) — sin cambios respecto a EXP-0011
hypothesis_id: H-RAD-HEAT-2
preregistered_at: 2026-10-04
preregistered_by: Rodrigo Mansilla (investigador; firma en el chat: "firmo EXP-0012")
---

# EXP-0012 — ¿El déficit de RAD tras L-BFGS venía de los puntos de L-BFGS? (seguimiento de EXP-0011)

Abierto el 2026-10-04 por decisión del investigador ("Opción B: abrir EXP-0012"). Redactado por un agente. **Firmado el 2026-10-04** por el investigador; desde ese momento los umbrales no se mueven.

## Pregunta

En EXP-0011, RAD redujo el error al final de Adam (razón 0.088 frente al control) pero el error tras L-BFGS completo quedó 4.9× por encima del control, y la hipótesis se rechazó por esa guardia. Esa fase de L-BFGS se había corrido sobre los últimos 4 096 puntos RAD, concentrados donde el residuo es alto. **¿Si L-BFGS se corre sobre el pool uniforme del control, el error tras L-BFGS de la rama RAD queda dentro de la guardia (≤ 2× el control)?**

## Hipótesis (falsable)

H-RAD-HEAT-2: partiendo de los checkpoints de fin de Adam de la rama RAD de EXP-0011 (semillas 901–905, sin reentrenar), un L-BFGS completo sobre el **pool uniforme de 4 096 puntos del control** (el de EXP-0008, mismo archivo por semilla) deja el L2 relativo en la malla analítica 512×256 en **≤ 2× el del control tras L-BFGS** (razón RAD+uniforme / control ≤ 2.0), en media geométrica y en al menos 4 de 5 semillas.

## Qué cambia respecto a EXP-0011 (y qué no)

| Elemento | EXP-0011 | EXP-0012 |
|---|---|---|
| Etapa Adam (RAD) | corrida nueva | **reutiliza** los checkpoints `checkpoint-adam-final.pt` de EXP-0011; no se reentrena |
| Puntos de L-BFGS | últimos 4 096 puntos RAD | **pool uniforme del control** (`heat-seed-901..905.npz`, campo `interior`) |
| Optimizador L-BFGS | strong-Wolfe, lr 1, historia 100, bloques de 25 it./125 evaluaciones, parada por plateau/cero progreso/15 000 it. | igual; tope de 700 s por semilla como en el control |
| Comparador | control EXP-0008 (`1.857e-6 ± 1.8e-7` tras L-BFGS) | el mismo |
| Mecanismo | RAD (potencia 2, 50 % uniforme, refresco 1 000) | sin cambios; no se toca `src/mechanisms/` |

## Métrica primaria

| Campo | Valor |
|---|---|
| Métrica | L2 relativo de campo, malla analítica 512×256, **tras L-BFGS completo sobre el pool uniforme** |
| Comparación | pareada por semilla contra el control EXP-0008 tras L-BFGS (`1.949e-6, 1.795e-6, 1.634e-6, 1.792e-6, 2.113e-6`) |
| **Umbral** | razón ≤ 2.0 en media geométrica **y** en ≥ 4 de 5 semillas |

## Métricas de guardia

| Métrica | Restricción |
|---|---|
| L2 al final de Adam (de EXP-0011, sin cambios) | se reporta; no se vuelve a evaluar como criterio |
| Tiempo de GPU de L-BFGS por semilla | ≤ 700 s (tope) |
| Estabilidad | 5/5 corridas con pérdida finita |

## Condición de falsación

Se rechaza H-RAD-HEAT-2 si: (a) la media geométrica de las razones es > 2.0; (b) menos de 4 de 5 semillas tienen razón ≤ 2.0; o (c) alguna corrida no es finita. Si se rechaza, el déficit no se explica por los puntos de L-BFGS; la rama RAD deja el modelo en un estado que L-BFGS no refina al piso del control, y queda como resultado negativo de RAD sobre este control. Un resultado favorable **no** es un claim: pasa al gate `scientific_claim`; y no revierte el veredicto de EXP-0011, que se evaluó con su propia regla.

**Límites declarados antes de correr:** n = 5 semillas pareadas, un solo problema y una sola configuración de RAD; el umbral de 2.0 es el de la guardia original, no un umbral de superioridad. Un resultado entre 1.0 y 2.0 se reporta tal cual, sin tratarlo como mejora sobre el control.

## Presupuesto

| Campo | Valor |
|---|---|
| Suite / tier | tier2_candidate / 2 |
| Backend | Kaggle T4×2, privado |
| Semillas | 901–905 (checkpoints RAD de EXP-0011) |
| Corridas | 5 (solo L-BFGS) |
| Costo máximo | 0.6 h de sesión, tope duro 2 400 s |

## Superficie de edición declarada

- [ ] ninguna: no se modifica `src/mechanisms/`, `evals/`, el modelo base ni la PDE. Solo un runner nuevo en `experiments/heat/EXP-0012-heat-rad-uniform-lbfgs/`.

## Amenazas de validez

- **Reutilización de checkpoints:** la etapa Adam es exactamente la de EXP-0011 (paridad por construcción); no hay nueva ventana para elegir semillas.
- **Pool del control:** se usa el archivo `heat-seed-S.npz` del control con su hash; es el mismo conjunto con el que se obtuvo el comparador.
- **Comparador:** el control de EXP-0008 es el pre-registrado; las réplicas del mismo runner de EXP-0011 mostraron diferencias por semilla (no se usan aquí).
- **No-determinismo de GPU:** una repetición de L-BFGS puede dar valores distintos por semilla; se reporta por semilla, no solo la media.

## Firma

- [x] Firmado por el investigador — fecha: 2026-10-04 — nombre: Rodrigo Mansilla (mensaje de chat "firmo EXP-0012", tras leer el spec completo)
