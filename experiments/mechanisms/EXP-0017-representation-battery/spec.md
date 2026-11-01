---
experiment_id: EXP-0017
slug: representation-battery
version: v1-draft
status: scored           # draft | preregistered | running | scored | decided
suite: tier1_regression  # solo CPU, sobre estados ya guardados
problem_id: problem.burgers_pinnacle_1d (principal), problem.allen_cahn_1d (réplica)
model_id: model.rbf_pikan (RBF48 en Burgers; RBF16 en Allen–Cahn), estados de la etapa A del EXP-0016
mechanism_id: none (diagnóstico en lazo abierto; decide el diseño del EXP-0018)
hypothesis_id: H-LOC-1 (localización), H-ATTR-1 (atribución), H-ATTR-2 (concentración)
preregistered_at: 2026-10-20
preregistered_by: Rodrigo Mansilla (mensaje de chat "firmo", 2026-10-20; registrado por el agente con su OK explícito)
---

# EXP-0017 — ¿Qué parte de la representación puede guiar la focalización? Batería en lazo abierto

> **Terminología.** "PIKAN-PDE" = la RBF-PIKAN entrenada con la pérdida física. Ninguna señal de este experimento lee la referencia; la referencia solo da etiquetas.

Redactado el 2026-10-20 por un agente, tras el cierre del "dónde" interno en el EXP-0016 v2 (H-AWARE-1 falló: el asentamiento de las bases de la capa 0 no localiza el error; el residuo sí, en Burgers). **No está firmado.** Solo CPU, sobre estados ya guardados: no hay corridas nuevas de entrenamiento.

## Restricciones del investigador (2026-10-20)

1. El título del protocolo no cambia: *Mecanismos de Focalización Adaptativa Guiados por Representación Interna en Arquitecturas PIKANs*.
2. Se diseña un mecanismo sí o sí. Este experimento no decide *si* hay mecanismo, decide *cuál*.

## Por qué fallaron las señales anteriores (hipótesis de trabajo)

Las señales probadas hasta ahora (EXP-0013; asentamiento de la capa 0 en EXP-0016) son **señales de actividad**: dicen dónde la red todavía cambia. El paso 0 del EXP-0015 mostró que el error de la PIKAN-PDE es de optimización: la red queda **estancada en un lugar equivocado**. Una señal de actividad no distingue "convergió" de "se trabó", y por eso no puede localizar ese error (en EXP-0016 el mapa de asentamiento fue incluso anticorrelacionado con el error en 10 de 25 checkpoints de Burgers).

Este experimento prueba los otros dos usos de la representación:

| Rol de la representación | Pregunta | Parte |
|---|---|---|
| **Fuente de señal** (lo ya probado, con señales nuevas que no son de actividad) | ¿alguna cantidad interna localiza el error mejor que el residuo? | 1 |
| **Geometría** | ¿la resolución que la red ofrece en x alcanza para la que la solución pide en x? | 1 (señal S5) |
| **Espacio de asignación** | ¿el residuo, atribuido base por base a través de la representación, identifica qué parámetros necesita mover el refinamiento? | 2 |

## Datos (ya existen)

`experiments/mechanisms/EXP-0016-stall-handoff/results/stage-a/stageA/`: para Burgers (semillas 301–305) y Allen–Cahn (601–603), pesos completos en los 5 puntos de traspaso de Adam (`*-adam-k*.pt`) y el estado final de L-BFGS iniciado desde cada uno (`*-lbfgs-k*.pt`), más las trazas de Adam. Total: 40 pares (estado de Adam, estado tras L-BFGS). Mallas de evaluación: las del EXP-0016 (Burgers submalla x[::4], t[::2]; Allen–Cahn completa).

## Parte 1 — Batería de localización (H-LOC-1)

En cada uno de los 40 estados de Adam, sin referencia, sobre la malla de evaluación:

| Señal | Definición | Rol |
|---|---|---|
| S1 | ‖∂u(x)/∂θ‖ (norma del jacobiano de la salida respecto de todos los parámetros, por punto) | cuánto puede moverse la representación en x |
| S2 | densidad de bases activas en las capas ocultas en x: Σ_ℓ≥1 Σ_i Σ_g 1[φ_ℓig(h_ℓi(x)) > 0.5] | resolución disponible en x |
| S3 | curvatura de las funciones de arista evaluada en el punto: Σ_ℓ Σ_i,o |f''_ℓoi(h_ℓi(x))| | dónde las funciones aprendidas están más dobladas |
| S4 | asentamiento de la capa 1 proyectado al dominio a través de h_0(x) (misma definición que en EXP-0016, capa 1, no separable) | actividad en una capa no separable (control de la explicación de arriba) |
| S5 | **desajuste de resolución**: ‖∇_x u(x)‖ / ‖∂h_1(x)/∂x‖ (variación que la solución propia de la red pide, dividida por el estiramiento local que ofrece la primera capa oculta) | geometría: demanda contra oferta |
| R | residuo |r(x)| (línea base, la señal de RAR) | referencia de comparación |

Etiqueta (solo para puntuar): |u − u_ref|(x). Métricas por estado: Spearman(señal, error) y solape entre el 5 % superior de la señal y el 5 % superior del error (azar = 0.05).

**H-LOC-1 se sostiene para una señal S si**, en Burgers **y** en Allen–Cahn, su mediana de Spearman y su mediana de solape superan a las del residuo, con la diferencia pareada por estado significativa por permutación con signo dentro de semilla (10 000 permutaciones), p corregida por Holm sobre 5 señales × 2 métricas × 2 problemas. Se reporta además la mejor señal aunque ninguna pase.

## Parte 2 — Atribución en el espacio de la representación (H-ATTR-1, H-ATTR-2)

**Responsabilidad de cada base** en el estado de Adam k, para cada capa ℓ, entrada i y base g:

A_ℓig(k) = Σ_x |r(x)| · φ_ℓig(h_ℓi(x))  /  Σ_x φ_ℓig(h_ℓi(x))

(el residuo medio que "ve" esa base; sin referencia).

**Lo que el refinamiento realmente movió** entre el estado de Adam k y el final de L-BFGS iniciado en k (dato observado, no un mecanismo):

Δ_ℓig = ‖c_ℓ[:, i, g](L-BFGS) − c_ℓ[:, i, g](k)‖ / (‖c_ℓ[:, i, g](k)‖ + ε)

- **H-ATTR-1:** en Burgers y en Allen–Cahn, la mediana sobre los 40 pares de Spearman(A, Δ) (sobre todas las bases de todas las capas) es ≥ 0.30 **y** supera a la de un control de ocupación O_ℓig = Σ_x φ_ℓig(h_ℓi(x)) (descarta que "las bases más usadas se mueven más"), con la diferencia pareada significativa por permutación con signo dentro de semilla (p < 0.05).
- **H-ATTR-2 (hay dónde focalizar):** en la mediana de los estados, el 20 % de las bases con más responsabilidad concentra ≥ 50 % de la responsabilidad total. Si la responsabilidad es difusa, focalizar en el espacio de parámetros no tiene margen.
- **Techo de la atribución (descriptivo):** la misma A calculada con el error real en lugar del residuo (A*, usa la referencia, etiquetada `oracle_ceiling`). Spearman(A*, Δ) dice cuánto se pierde por usar el residuo.
- **Comprobación de efecto (descriptiva):** en los pares donde el error bajó, ¿bajó más en las regiones donde actúan las bases con mayor A? (error por franja de soporte de cada base de la capa 0, antes y después).

## Árbol de decisión (vinculante al firmar): siempre termina en un mecanismo

| Resultado | Mecanismo del EXP-0018 |
|---|---|
| H-ATTR-1 y H-ATTR-2 se sostienen | **Focalización en el espacio de la representación**: en la fase de refinamiento, el paso de cada base se escala por su responsabilidad A (precondicionamiento diagonal por base: lr_ℓig ∝ (A_ℓig / mediana A)^β, β = 1). Es específico de KAN (bases locales); un MLP no lo admite. Si además alguna señal pasa H-LOC-1, se usa como peso espacial complementario |
| Pasa H-LOC-1, falla H-ATTR | **Focalización espacial guiada por la señal ganadora** (la ganadora reemplaza al mapa de asentamiento en el brazo T del EXP-0016 v2, que ya está especificado) |
| H-ATTR-1 pasa, H-ATTR-2 falla | responsabilidad difusa: el mecanismo usa la atribución como **criterio de cuándo** (traspasar cuando la responsabilidad se concentra), no de dónde |
| Nada pasa | se diseña igual la focalización en el espacio de la representación (primera fila), pero **primero con su techo oráculo** (A* con el error real) en GPU antes de cualquier versión con residuo; el resultado negativo de la Parte 1 se reporta como hallazgo |

Cualquier mecanismo resultante se evalúa con las mismas reglas que el EXP-0016 v2: igual reloj, mismos parámetros (§4.4 del protocolo), región crítica y guardia del error global (§4.5), y prueba de estrés en los seis problemas.

## Límites declarados

- n = 5 semillas en Burgers y 3 en Allen–Cahn; los 5 estados de una semilla comparten la trayectoria de Adam (la inferencia usa la semilla como unidad).
- Δ mide lo que movió L-BFGS, no lo que *debería* haber movido: H-ATTR-1 es una correlación con el refinamiento observado, no una prueba causal. La causal es el EXP-0018.
- S1 es caro en Burgers (29 376 parámetros): se calcula en una submalla fija de 8 192 puntos por estado, la misma para todas las señales en la Parte 1.
- En Allen–Cahn ninguna señal localizó el error hasta ahora (residuo: solape 0.011): es la réplica más exigente.

## Pruebas previas (Tier 0/1, CPU)

- señales y responsabilidad en un módulo que no importa `evals` ni la referencia (prueba de aislamiento);
- unitarias de A y Δ con redes sintéticas (una base con error concentrado en su soporte debe tener la mayor A);
- reproducir con este código el residuo y el solape del residuo del EXP-0016 (Burgers 0.455 / 0.393) antes de calcular nada nuevo.

## Presupuesto

Solo CPU. Estimación: Parte 1 dominada por S1 (jacobiano por punto) en Burgers, del orden de 1–3 h en 4 núcleos; Parte 2 minutos. Sin GPU.

## Superficie de edición declarada

- [x] scripts de señales y análisis dentro de `experiments/mechanisms/EXP-0017-representation-battery/`.
- [ ] No se toca `evals/`, el modelo base, la PDE, las recetas ni los estados del EXP-0016.

## Firma

- [x] Firmado por el investigador — fecha: 2026-10-20 — nombre: Rodrigo Mansilla (chat: "firmo")
- [x] Aprobado el uso de la referencia como etiqueta y en el techo A* (`oracle_ceiling`) — fecha: 2026-10-20 (incluido en la firma; sin etiquetas el experimento no se puede puntuar)
