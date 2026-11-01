---
experiment_id: EXP-0023
slug: gram-preconditioning
version: v2             # firmada el 2026-10-20; v1 (borrador) archivada en spec-v1-draft.md
status: preregistered   # PRUEBAS PREVIAS PENDIENTES: no se lanza GPU hasta que pasen y se registre el presupuesto
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d (desarrollo), problem.allen_cahn_1d (réplica)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn; 11 bases, ancho 1.75 × espaciado), baseline v2
mechanism_id: mechanism.gram_preconditioning (src/mechanisms/gram_preconditioning.py, por escribir y probar)
hypothesis_id: H-PRE-1 (mejora frente al baseline), H-PRE-2 (mejora frente a Jacobi), H-PRE-3 (importa que la Gram sea de las activaciones entrenadas)
gate: EXP-0022 no sostuvo H-COND-1 ni H-COND-2 por la letra; el investigador decidió firmar este experimento de forma explícita (chat 2026-10-20: «firmo 0023 y 0024»), como prevé la regla del 0022
preregistered_at: 2026-10-20
preregistered_by: Rodrigo Mansilla (mensaje de chat "firmo 0023 y 0024", 2026-10-20 16:41, sobre esta v2; registrado por el agente. Los cambios de la v1 a la v2 los había señalado el agente antes en el chat solo en resumen: si objeta alguno antes de lanzar GPU, se enmienda antes de ver datos)
---

# EXP-0023 — Precondicionamiento por la estructura de la representación

> **Terminología.** "PIKAN-PDE" = RBF-PIKAN entrenada con la pérdida física. Este mecanismo **no usa referencia ni residuo**: solo el estado interno (activaciones y jacobianos de la red).

## Cambios respecto de la v1

1. **Jacobi pasa a ser el comparador principal** (H-PRE-2). El EXP-0022 midió que el bloque por arista mejora κ_eff(100) solo 2.9× (Burgers) y 5.9× (Allen–Cahn) sobre Jacobi, y Adam ya hace algo parecido a Jacobi. Un mecanismo que no supera a Jacobi es un reescalado.
2. **Se añade el brazo por entrada** (bloque con todos los coeficientes que leen la misma entrada): en el 0022 redujo κ más que el bloque por arista (0.0039 frente a 0.0084 del crudo en Burgers). Se declara aquí como brazo, no se elige después.
3. **Se elimina el brazo GA** (solo fase de Adam); las dos fases se reportan por separado de todos modos (fin de Adam y tras L-BFGS).
4. **Umbrales con potencia declarada.** Con 5 semillas y los umbrales 0.85/0.90, la calibración del EXP-0018 (`EXP-0018.../calibration.json`) da falsos positivos < 1 % y potencia ≈ 53 % si el efecto verdadero es una razón de 0.6, ≈ 28 % si es 0.7 y ≈ 11 % si es 0.8. **Un resultado negativo significa «sin efecto grande detectable», no «sin efecto».**
5. **Guardias g1 y g3** del EXP-0018 (región crítica y finitud) añadidas. g2 se reemplaza por igualdad de reloj total, con el costo de calcular los bloques incluido.
6. Semillas fijadas aquí (Burgers 801–805, Allen–Cahn 811–815).

## Mecanismo

Para cada arista (ℓ, o, i), los 11 coeficientes se reparametrizan como c = P·c′, con P = (B + εI)^(-1/2) y B el bloque 11×11 de Gauss–Newton de esa arista (definición del EXP-0022, ε = 1e-3 × traza/11). El optimizador mueve c′. Los demás parámetros quedan como en el baseline. B se recalcula cada R = 1 000 pasos en la fase de Adam y una vez al iniciar L-BFGS, porque las activaciones h_ℓ(x) de las capas ocultas cambian al entrenar. Un solo ε y un solo R, sin barrido.

## Brazos

| Brazo | Qué hace | Papel |
|---|---|---|
| **B** | baseline v2 | línea base |
| **J** | solo la escala de Jacobi, 1/√diag(GN) | comparador principal: ¿es solo un reescalado? |
| **E** | bloque por arista, Adam y L-BFGS | mecanismo |
| **I** | bloque por entrada (todos los coeficientes de la misma entrada), Adam y L-BFGS | mecanismo |
| **F** | Gram *fija* con entradas uniformes en [−1, 1], sin recalcular | control: ¿importa que venga de las activaciones entrenadas? |

Igual reloj total, mismos parámetros (§4.4 del protocolo). El costo de calcular los bloques entra al reloj. Corridas completas desde la inicialización (el precondicionamiento actúa también en Adam), cinco semillas por problema.

## Hipótesis y criterios (vinculantes)

Razón pareada r = error(brazo) / error(B) por semilla, error L2 relativo medido **tras L-BFGS** (primaria) y al final de Adam (secundaria).

- **H-PRE-1.** Para E o I: en Burgers **y** Allen–Cahn, mediana de r ≤ 0.85 y mejora en ≥ 4 de 5 semillas; Holm sobre 2 brazos × 2 problemas.
- **H-PRE-2.** El brazo que pasó H-PRE-1 frente a J: mediana de razón ≤ 0.95 y mejora en ≥ 4 de 5 semillas, en ambos problemas.
- **H-PRE-3.** El mismo brazo frente a F: mediana de razón ≤ 0.95 y ≥ 4 de 5, en ambos problemas. Si no se sostiene, la ganancia viene de corregir el solape de la base y no de la representación entrenada.
- **Guardias.** g1: error L2 en la región crítica (frente de Burgers |x| ≤ 5ν, t ≥ 1/π; interfaz de Allen–Cahn |u_ref| < 0.9; definida con la referencia, solo para puntuar) del brazo frente a B, razón en media geométrica ≤ 1.10. g3: parámetros finitos en todas las corridas.

## Regla de decisión (vinculante)

| Resultado | Consecuencia |
|---|---|
| H-PRE-1, H-PRE-2 y H-PRE-3 | Mecanismo candidato. Sigue la prueba de estrés en los seis problemas (preregistro aparte) |
| H-PRE-1 y H-PRE-2, no H-PRE-3 | La representación entrenada no es lo que importa; es el bloque de la base. Se reporta como hallazgo sobre la base RBF; no sostiene el título |
| H-PRE-1 sí, H-PRE-2 no | Es un reescalado que Jacobi iguala. Se reporta; no hay mecanismo de representación |
| H-PRE-1 no | Se cierra el precondicionamiento por bloques de arista y de entrada con este diseño; resultado negativo informativo, acotado por la potencia declarada |

## Límites declarados antes de correr

- El precondicionador es local y se recalcula a intervalos.
- L-BFGS es casi invariante a un reescalado diagonal con historia acumulada; los bloques completos lo alteran, pero el efecto puede ser pequeño.
- El costo de calcular B en el reloj puede borrar una mejora pequeña por iteración.
- n = 5: potencia moderada solo para efectos grandes (ver cambio 4).
- Solo dos problemas; la prueba de estrés ocurre solo si hay candidato.

## Presupuesto

GPU. **Por medir** en las pruebas previas (corridas completas × 5 brazos × 10 semillas). No se lanza hasta registrarlo.

## Pruebas previas (Tier 0/1, **pendientes**)

1. Con P = I el brazo reproduce a B bit a bit.
2. El bloque B de la implementación coincide con el de `EXP-0022.../gn.py` (diferencia máxima < 1e-8).
3. El gradiente respecto de c′ es Pᵀ·(gradiente respecto de c).
4. F no recalcula B; E e I sí, en los pasos fijados.
5. Aislamiento: el mecanismo no importa `evals/` ni recibe referencia (prueba por AST como en el 0022).
6. Presupuesto medido en una corrida completa por problema.
7. Equivalencia CPU/GPU de la reparametrización en una semilla (como la prueba 7 del 0018).

## Superficie de edición declarada

Nuevo: `src/mechanisms/gram_preconditioning.py`, `tests/test_gram_preconditioning.py`, `experiments/mechanisms/EXP-0023-gram-preconditioning/`. Sin cambios en `evals/`, `registry/claims.jsonl`, baselines ni experimentos anteriores.

## Firma

- [x] Preregistro: brazos B/J/E/I/F, criterios (0.85, 0.95, 0.95, 4/5), guardias g1 y g3, ε, R, semillas y regla de decisión — Rodrigo Mansilla — fecha: 2026-10-20 (chat: «firmo 0023 y 0024»)
