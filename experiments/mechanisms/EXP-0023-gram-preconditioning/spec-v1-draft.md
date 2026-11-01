---
experiment_id: EXP-0023
slug: gram-preconditioning
version: v1-draft
status: draft          # CONDICIONADO a EXP-0022: no firmar antes de leer su resultado
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d (desarrollo), problem.allen_cahn_1d (réplica)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn), baseline v2
mechanism_id: mechanism.gram_preconditioning (src/mechanisms/gram_preconditioning.py, por escribir)
hypothesis_id: H-PRE-1 (el precondicionamiento por arista mejora el resultado), H-PRE-2 (importa que la Gram sea la de las activaciones entrenadas)
gate: EXP-0022 con H-COND-1 y H-COND-2 sostenidas
---

# EXP-0023 — Precondicionamiento por la estructura de la representación

> **Terminología.** "PIKAN-PDE" = RBF-PIKAN entrenada con la pérdida física. Este mecanismo **no usa referencia ni residuo**: solo el estado interno (activaciones y jacobianos de la red).

Redactado el 2026-10-20 por un agente. **No está firmado y no debe firmarse antes del resultado del EXP-0022.** Los números marcados «propuesto» se pueden mover antes de firmar.

## Mecanismo

Para cada arista (ℓ, i), los 11 coeficientes se reparametrizan como c = P·c′, con P = (B + εI)^(-1/2) y B el bloque 11×11 de Gauss–Newton de esa arista (definición del EXP-0022). El optimizador mueve c′. Los parámetros restantes (rama `base`, etc.) quedan como en el baseline. B se **recalcula** cada R pasos (R = 1 000 en la fase de Adam, propuesto; una vez al iniciar L-BFGS), porque las activaciones h_ℓ(x) de las capas ocultas cambian al entrenar.

Es la representación la que define el precondicionador: bases locales y activaciones entrenadas fijan qué direcciones del espacio de coeficientes son planas. Un MLP no tiene estos bloques.

## Brazos

| Brazo | Qué hace | Papel |
|---|---|---|
| **B** | baseline v2 sin cambios | línea base |
| **G** | precondicionamiento por arista, Adam y L-BFGS | mecanismo |
| **GA** | precondicionamiento por arista solo en la fase de Adam | separa dónde importa (L-BFGS compensa parte del mal condicionamiento) |
| **J** | solo la escala de Jacobi, 1/√diag(B) | control: ¿basta con la escala? |
| **F** | Gram *fija* con entradas uniformes en [−1.05, 1.05], sin recalcular | control: ¿importa que la Gram venga de las activaciones entrenadas, o solo se corrige el solape de la base? |

El costo de calcular B entra al reloj en todos los brazos que lo calculan (no es oráculo). Mismo reloj total, mismos parámetros (§4.4 del protocolo), región crítica y guardia del error global (§4.5). Un solo ε y un solo R, fijados aquí; sin barrido.

## Hipótesis y criterios (vinculantes al firmar)

Semillas nuevas, disjuntas de las del EXP-0016 y 0017 (Burgers 801–805, Allen–Cahn 811–815, propuesto). Razón pareada r = error(brazo) / error(B) por semilla, **medida tras L-BFGS** (primaria) y al final de Adam (secundaria); el patrón conocido es que las ganancias de Adam se pierden tras L-BFGS.

- **H-PRE-1.** Para G o GA: en Burgers **y** Allen–Cahn, mediana de r ≤ 0.85 tras L-BFGS (propuesto) y mejora en ≥ 4 de 5 semillas; Holm sobre 2 brazos × 2 problemas.
- **H-PRE-2.** El brazo que pasó H-PRE-1 frente a F: mediana de razón ≤ 0.95 y mejora en ≥ 4 de 5 semillas en ambos problemas. Sin esto, la ganancia se debe a la base (solape) y no a la representación entrenada, y no sostiene el título.
- Frente a J: reportado, sin hipótesis. Si J iguala a G, el mecanismo se reduce a un reescalado que Adam ya hace.

## Regla de decisión (vinculante)

| Resultado | Consecuencia |
|---|---|
| H-PRE-1 y H-PRE-2 | Mecanismo candidato. Sigue la prueba de estrés en los seis problemas (preregistro aparte) |
| H-PRE-1 sí, H-PRE-2 no | El efecto viene de corregir el solape de la base. Se reporta como hallazgo sobre la base RBF; el mecanismo guiado por la representación entrenada no se sostiene |
| H-PRE-1 no | Se cierra el precondicionamiento por bloques de arista; con EXP-0022 positivo, queda el costo/beneficio como resultado negativo informativo |

## Límites declarados antes de correr

- El precondicionador es local y se recalcula a intervalos; entre recálculos queda obsoleto.
- L-BFGS es casi invariante a un reescalado diagonal con historia acumulada; el efecto de bloques completos (no diagonales) sí lo altera, pero puede ser pequeño.
- Costo de calcular B en el reloj puede borrar una mejora pequeña por iteración.
- n = 5 por problema: el p exacto por semilla solo llega a 1/32.

## Presupuesto

GPU. A estimar tras las pruebas previas; del orden del EXP-0018 más el costo de B.

## Pruebas previas (Tier 0/1, CPU)

1. Con P = I el brazo reproduce a B bit a bit.
2. El bloque B de la implementación coincide con el de EXP-0022.
3. El gradiente respecto de c′ es Pᵀ·(gradiente respecto de c).
4. F no recalcula B; G sí, en los pasos fijados.
5. Aislamiento: el mecanismo no importa `evals/` ni recibe referencia.

## Superficie de edición declarada

Nuevo: `src/mechanisms/gram_preconditioning.py`, `tests/test_gram_preconditioning.py`, `experiments/mechanisms/EXP-0023-gram-preconditioning/`. Sin cambios en `evals/`, `registry/claims.jsonl`, baselines ni en experimentos anteriores.

## Firma

- [ ] Preregistro (solo tras leer el resultado del EXP-0022): criterios, umbrales (0.85, 0.95, 4/5), ε, R y regla de decisión — Rodrigo Mansilla — fecha: ____
