---
experiment_id: EXP-0025            # ID provisional: el 0019 está reservado; 0020–0021 no existen; confirmar al firmar
slug: observer-v0
version: v1-draft
status: draft            # draft | preregistered | running | scored | decided — el investigador lo cambia a `preregistered`
suite: tier2_candidate
problem_id: problem.allen_cahn_1d (primario; d=1e-3 y, si 1-C lo firma, d=2e-4), problem.burgers_pinnacle_1d (donde E falla)
model_id: model.rbf_pikan (RBF16 en Allen–Cahn; RBF48 en Burgers), baseline v2, receta Adam 4000 / L-BFGS 500 como EXP-0023/0024
mechanism_id: mechanism.observer_v0 (src/mechanisms/observer_v0.py, por escribir y probar)
hypothesis_id: H-OBS-1 (no arrepentimiento), H-OBS-2 (el sondeo discrimina), H-OBS-3 (lectura espacial)
gate: depende del cierre de la ronda 1-C de EXP-0023 (da el finalista E/F y la lectura de cond(P)). La fase 0 (implementar y probar en CPU) no depende de ella.
preregistered_at:
preregistered_by:
---

# EXP-0025 — Observador con sondeo: un controlador que elige palanca durante el entrenamiento

## Pregunta

¿Un controlador que, mientras la red entrena, prueba cada palanca candidata en una continuación corta sobre puntos retenidos y se queda con la que
ayuda, iguala o supera a la **mejor palanca fija de cada problema** sin saber de antemano cuál es?

## Contexto: lo que este repositorio ya cerró (NO re-proponer)

| Familia | Experimento | Resultado registrado | Consecuencia para este diseño |
|---|---|---|---|
| Puerta λ por Spearman entre residuo y estructura | EXP-0014 | Posposición propuesta: λ solo pesa cuando las señales coinciden y equivale a la ablación de estructura (G≈S). | **No usar acuerdo entre señales como puerta.** El discriminador es el sondeo. |
| Ubicar/refinar bases de la capa 0 | EXP-0015 | Sin efecto robusto; U41 mejora también Heat (capacidad). | Reubicación de bases queda **fuera**. |
| Asentamiento de bases y traspaso Adam→L-BFGS | EXP-0016 | El mapa interno localiza el error peor que el residuo; el techo del traspaso no se distingue de ruido (p = 0.23). Cerrado. | «Cuándo pasar a L-BFGS» queda **fuera**. |
| Batería de señales internas | EXP-0017 | Ninguna señal interna supera al residuo; S1 era artefacto de la restricción dura. | Las señales internas **no se usan para localizar**. |
| Foco por responsabilidad por base | EXP-0018 | Con el error real tampoco hay techo. Familia cerrada. | Fuera. |
| Diagnóstico y precondicionamiento | EXP-0022, 0023, 0024 | Mal condicionamiento real (κ_eff ≈ 2 000–2 700). En la ronda 1-B, E/F replican en Allen–Cahn (10/10 semillas) y **no** en Burgers (V: 5.70, 1/5). | Hay una palanca con respaldo parcial: la corrección de Gram. |

Lo consistente en todo el recorrido: el residuo es la mejor localizadora entre las señales sin referencia y el error de la PIKAN-PDE es de
optimización más que de capacidad o localización. Y ninguna palanca fija gana en todos los problemas (E: sí en Allen–Cahn, no en Burgers;
focalización por residuo en calor: mejoró una etapa y empeoró la otra). **Ese es el caso para un selector.**

## Hipótesis

Sean `r_X` la razón mediana del error relativo L2 tras L-BFGS de X frente a la **réplica de B en la misma sesión**, pareada por semilla.

- **H-OBS-1 (no arrepentimiento).** En cada problema, `r_O ≤ min(1, r_R, r_E) + ε` (ε = 0.10 `[ABIERTO]`), y ninguna semilla de O supera 2× el error de B. Además, `r_O ≤ 0.90` en al menos un problema.
- **H-OBS-2 (el sondeo discrimina).** `r_O ≤ 0.90 · r_{O−}` en al menos un problema, donde O− elige la palanca al azar en cada decisión (sin sondeo).
- **H-OBS-3 (lectura espacial; no gobierna el paso).** Razón O/B en la región de interfaz (5 % de mayor |u_x| de la referencia, solo en la evaluación) menor que en el resto del dominio por un margen `[ABIERTO]`. Si no, el texto de la tesis no llama focalización a O.

## Diseño del mecanismo

**Palancas (v0: dos, para que la atribución sea posible).**
- `aR` — muestreo ∝ |r|² con piso uniforme u = 0.5 `[ABIERTO]`, presupuesto de puntos fijo (reutilizar `src/mechanisms/rad.py`; no modificarlo).
- `aG` — corrección de Gram (reutilizar `src/mechanisms/gram_preconditioning.py`; variante E o F según el finalista de 1-C `[ABIERTO]`).
- `a0` — continuar sin cambio (referencia de la comparación).

**Ciclo de decisión** cada `K` pasos de Adam (`K = 500` `[ABIERTO]`; con 4000 pasos hay 8 decisiones):
1. Guardar instantánea (θ y estado del optimizador).
2. Para cada `a ∈ {a0, aR, aG}`: continuar `n_p` pasos (`n_p = 25` `[ABIERTO]`) desde la misma instantánea, con la misma semilla de lotes.
3. Evaluar en el bloque `H_d` (fresco para la decisión d) el residuo medio, global y por celda.
4. Elegir `a ≠ a0` solo si mejora el residuo medio ≥ δ respecto de `a0` (`δ` `[ABIERTO]`) **y** ninguna celda crítica empeora más de γ (`γ` `[ABIERTO]`). Celdas críticas = el 5 % de mayor residuo en `H` (sin referencia). Si ambas pasan, la de mayor mejora.
5. Restaurar la instantánea y aplicar la elegida durante los `K` pasos siguientes.

**Costo.** Extra ≈ 3 · n_p / K = 15 % de pasos (+ evaluación en `H`). Objetivo: reloj ≤ 1.5× el de B. Medir en Tier 0.

**Puntos retenidos `H_d`.** Un bloque nuevo de 8192 puntos Sobol por decisión d, de un flujo propio, disjunto del entrenamiento y de los bloques de las demás decisiones (reutilizar un `H` fijo infla la mejora estimada: el observador elegiría varias veces sobre el mismo conjunto). **No es la `eval_grid` de `evals/`**: el mecanismo no la lee ni importa `evals`.

**Observaciones registradas por decisión.** step, acción elegida, mejora por acción, celdas que empeoraron, tiempo. Si el observador elige `a0` en > 90 % de las decisiones se reporta como «inerte».

## Métrica primaria

| Campo | Valor |
|---|---|
| Métrica | error relativo L2 tras L-BFGS (grader de `evals/`) |
| Dirección | minimize |
| Comparador | réplica de B en la misma sesión, pareada por semilla |
| **Delta que cuenta** | `r_O ≤ 0.90` y H-OBS-1 según el texto de arriba |

## Métricas de guardia

| Métrica | Restricción |
|---|---|
| Error al final de Adam | razón O/B, media geométrica ≤ 1.25 |
| `wall_time_sec` | ≤ 1.5× B, incluido el costo del observador |
| `stability_score` | 5/5 corridas con pérdida finita |
| Región crítica | ninguna semilla con empeoramiento > γ en la solución final |
| Cola | ninguna semilla de O con error > 2× B |

## Condición de falsación

Se **rechaza H-OBS-1** si en cualquier problema `r_O > min(1, r_R, r_E) + ε`, o si alguna semilla pasa 2× B, o si se viola una guardia.
Se **rechaza H-OBS-2** si `r_O > 0.90 · r_{O−}` en ambos problemas. Si O ≈ O− el sondeo no discrimina y el valor del observador no es el sondeo.
Mejora del error global con empeoramiento local en regiones críticas ⇒ no respaldada (protocolo §4.5). Un rechazo no se rescata bajando umbrales.

## Brazos y presupuesto

| Brazo | Qué es |
|---|---|
| B | control (réplica en la misma sesión) |
| R | `fixed:aR` dentro de `observer_v0` (mismo camino de código que O) |
| E (o F) | `fixed:aG` dentro de `observer_v0`, finalista de 1-C. El E de EXP-0023 (cadencia 1000) queda solo como verificación cruzada, no como comparador de H-OBS-1 |
| O | observador con sondeo |
| O− | observador sin sondeo: elige al azar, uniforme, entre `{a0, aR, aG}` en cada decisión. Se reporta la frecuencia de acciones de O por problema; si O elige `a0` en más de la mitad de las decisiones, se añade el brazo O−m (misma frecuencia de acciones que O en ese problema) y H-OBS-2 se evalúa contra O−m |

| Campo | Valor |
|---|---|
| Suite / tier | tier2_candidate / 2 |
| Backend | local CPU (fase 0) → Kaggle T4 (bloque, tras Tier 0/1) |
| Semillas | T1 = 21–25, V = 121–125 `[ABIERTO]`; deben ser disjuntas de las usadas en 1-B (301–305 / 601–605 / otras) y 1-C (6–10, 106–110, 11–20) |
| Corridas máximas | 5 brazos × 2 problemas × 5 semillas = 50 por bloque |
| Costo máximo | `[ABIERTO]`; referencia: 50 corridas de 1-B tardaron 2 860 s en T4; O cuesta ~1.5× por corrida |

## Superficie de edición declarada

- [x] `src/mechanisms/observer_v0.py` (nuevo)
- [ ] `src/mechanisms/rad.py`, `src/mechanisms/gram_preconditioning.py`: **solo se importan, no se modifican**
- [ ] campos de `focalization.bounded_edit_surface` del manifiesto: ninguno nuevo en v0

No puede tocar `evals/`, el modelo base, ni la definición de la PDE. No puede importar `evals`
(`tests/test_mechanism_isolation.py`).

## Tier 0 / pruebas previas (CPU, sin GPU; las ejecuta el agente)

1. Aislamiento: el módulo no importa `evals`.
2. Al rechazar una acción, θ y el estado del optimizador se restauran **exactamente** (comparación bit a bit).
3. El presupuesto de puntos de colocación es constante en todas las decisiones.
4. Determinismo dada una semilla; mismo flujo de lotes en las continuaciones del sondeo.
5. Los bloques `H_d` son disjuntos entre sí, del conjunto de entrenamiento y de la `eval_grid`.
6. Sobrecosto medido en Heat o Fisher-KPP (pocos pasos): proporción de pasos extra y de reloj.
7. Si una palanca no puede medirse, el sondeo devuelve `null` con razón; el gate lo trata como no concluyente (regla dura 5).
8. La variante F de `aG` usa `_set` y `_fixed_blocks` de `GramPreconditioner` (métodos privados): una prueba falla fuerte si cambian de nombre o firma.
9. Tier 1 (CPU, minutos): con la receta real (Adam 4000, L-BFGS 500) y dos semillas, medir la frecuencia de acciones de O, la tasa de inercia (`a0` > 90 %) y el reloj con K = 500, n_p = 25. Sin medir error contra palancas fijas: eso lo responde el bloque.

## Amenazas de validez

- **Leakage entrenamiento/evaluación:** `H` es un flujo propio; la evaluación usa la `eval_grid` fija de `evals/`. Verificar en Tier 0.
- **El residuo retenido no es el error.** Precedente: la focalización por residuo mejoró una etapa y empeoró la otra en calor. El sondeo puede aceptar cambios que no ayudan al error real; por eso H-OBS-1 se mide con el grader de `evals/`, no con el residuo de `H`.
- **Sesgo de horizonte corto:** con `n_p` = 25 el sondeo favorece lo que mejora rápido (transitorio de Adam) y no lo que ayuda al final. Mitigación parcial: el brazo O− y la guardia de Adam.
- **Dependencia de 1-C:** la palanca `aG` depende del finalista. Sin finalista, O queda con una sola palanca y H-OBS-2 pierde sentido.
- **Potencia:** 5 semillas solo detectan efectos grandes; la guardia de cola es explícita por eso.
- **Costo:** si el sondeo supera 1.5× B, el resultado es un negativo de costo aunque el error mejore.

## Firma

El pre-registro entra en vigor cuando el investigador cambia `status` a `preregistered` y añade fecha. A partir de ese momento los umbrales no se mueven.

## Enmiendas al borrador (2026-10-21, antes de firmar)

1. `H` pasa de conjunto fijo a un bloque fresco `H_d` por decisión (sesgo de selección).
2. Los comparadores R y E del bloque corren por el mismo camino de código que O (`fixed:aR`, `fixed:aG`); el E de EXP-0023 es verificación cruzada.
3. O− se fija como elección uniforme entre `{a0, aR, aG}`; se añade O−m si O resulta mayormente inerte.
4. Pruebas 8 y 9 añadidas a Tier 0/1.

## Abiertos antes de firmar (2026-10-21)

- El finalista E o F de la ronda 1-C de EXP-0023.
- ε, las semillas 21–25 y 121–125, y el costo máximo.

Los demás parámetros marcados `[ABIERTO]` en el texto (K, n_p, δ, γ, piso uniforme u) tienen hoy valores por defecto técnicos del módulo, sin ajustar con datos.
