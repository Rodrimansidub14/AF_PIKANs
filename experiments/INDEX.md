# Índice de experimentos (por problema)

Estructura: `experiments/<problema>/<EXP-NNNN-slug>/{spec.md, manifest.json, decision.md, notebooks/, results/, figures/, checkpoints/}`.
Los notebooks ejecutables viven en `notebooks/<problema>/<EXP>/` (la ontología fija `notebooks/` como raíz auditada). Rutas viejas → nuevas: `docs/MOVED.md` en la raíz.

| Problema | Experimentos | Resultados recientes (EXP-0008, 2026-09/10) |
|---|---|---|
| `heat/` | EXP-0001, EXP-0002 (Heat+Burgers), EXP-0005 (solo notebooks), EXP-0007 | `results/heat-lbfgs-redo-20260930` |
| `burgers/` | EXP-0006 | — |
| `allen-cahn/` | EXP-0003 | `results/ac-rerun-20260930` (vs `Allen_Cahn.mat`) |
| `navier-stokes/` | EXP-0004 (DFG) | `ns2d-cg-20260930`, `dfg-daniel-cfg-20260930`, `dfg-long-20261001` |
| `piezo/` | — (por abrir) | — |
| `baseline-v2/` | EXP-0008 (plan, evaluadores, recetas; transversal) | — |

Figuras: pendientes (se generan en la fase de análisis). Crudo: `var/agent_runs/`.

**EXP-0011** (`heat/EXP-0011-heat-rad-adam`): primer experimento de mecanismo (RAD sobre el control RBF de Heat). Firmado 2026-10-03; puntuado 2026-10-04: primario cumplido, guardia de L-BFGS violada → rechazada por la regla pre-registrada; `decision.md` firmado el 2026-10-04.

**Baselines congelados:** v1 (2026-10-03) y v2 (2026-10-04, supersede a v1 en Burgers y piezo), en `evals/baselines/`.

**EXP-0012** (`heat/EXP-0012-heat-rad-uniform-lbfgs`): seguimiento de EXP-0011 — ¿el déficit tras L-BFGS venía de correr L-BFGS sobre los puntos RAD? Abierto 2026-10-04 por decisión del investigador. Firmado y puntuado 2026-10-04: rechazada (geomean 2.025 > 2.0; 2/5 semillas ≤ 2.0); `decision.md` firmado el 2026-10-04.

## Estado de los EXP (2026-10-04)

| EXP | Estado | Decisión |
|---|---|---|
| 0001 | decidido | superseded (nunca se corrió) |
| 0002 / 0005 | cerrados | registro exploratorio |
| 0003 | cerrado | registro exploratorio (referencia AC vigente: `Allen_Cahn.mat`) |
| 0004 (DFG) | borrador | abierto: se retoma en la fase de DFG |
| 0006 | cerrado | rechazado (resultado negativo exploratorio de RAD en Burgers) |
| 0007 | cerrado | registro exploratorio (L-BFGS truncado supersedido) |
| 0008 | cerrado | superseded por los baselines v1 y v2 |
| 0009 | cerrado | baseline de piezo registrado en v2 |
| 0011 | decidido | **rechazada** por su regla pre-registrada |
| 0012 | decidido | **rechazada**; hilo RAD en Heat cerrado |

## Patrón registrado (EXP-0011 y EXP-0012)

En el control RBF de Heat, **RAD mejora una etapa y empeora la otra**: ~11× menos error al final de Adam, y un estado que tras L-BFGS completo queda 2–5× por encima del control (4.9× con L-BFGS sobre los puntos RAD, 2.0× sobre el pool uniforme). Reglas derivadas para el primer mecanismo: se evalúa en las dos etapas, con L-BFGS sobre un pool fijo declarado de antemano, y la guardia de L-BFGS es vinculante.

**EXP-0013** (`diagnostics/EXP-0013-run-diagnosis`): diagnóstico por corrida para definir un mecanismo de awareness (solo CPU). Firmado 2026-10-05; puntuado 2026-10-05 (248 checkpoints, 272 señales; `results.md`); `decision.md` pendiente (humano).

**EXP-0013** decidido 2026-10-05: aceptado como campaña de exploración inicial (`diagnostics/EXP-0013-run-diagnosis/decision.md`).

**EXP-0014** (`mechanisms/EXP-0014-self-gated-sampler`): primera iteración de mecanismo (muestreador autogobernado por la concordancia residuo–estructura). **Borrador sin firmar** (2026-10-05); mecanismo implementado y probado (20 pruebas); se lanza tras la renovación de cuota del 2026-10-10.

**EXP-0015** (`mechanisms/EXP-0015-layer0-basis-refinement`): ¿el espaciado de bases de la capa 0 limita donde la estructura es más fina? **Borrador sin firmar** (2026-10-05); se lanza tras la renovación de cuota.

## Actualización 2026-10-18

- **EXP-0015 v2** (`mechanisms/EXP-0015-layer0-basis-refinement`): **firmado** por el investigador el 2026-10-18, con la enmienda del paso 0 (ajuste supervisado en CPU, en los seis problemas) y la aprobación del brazo oráculo. El bloque GPU queda bloqueado hasta leer el paso 0.
- **EXP-0014**: posposición propuesta (ver `registry/proposals.jsonl`), pendiente de decisión.
- Corrección de este índice: los `decision.md` de EXP-0011 y EXP-0012 estaban firmados desde el 2026-10-04.

- **Paso 0 del EXP-0015** corrido el 2026-10-18 (n = 1, seis problemas): `mechanisms/EXP-0015-layer0-basis-refinement/step0/results.md`. Siguiente paso y dirección del EXP-0016: `docs/plans/2026-10-18-handoff-paso0-exp0016.md`.

- **EXP-0015** cerrado en el paso 0 (CPU, n = 3) el 2026-10-18: bloque GPU cancelado por el investigador (`mechanisms/EXP-0015-layer0-basis-refinement/step0/decision.md`).
- **EXP-0016** (`mechanisms/EXP-0016-stall-handoff`): ¿puede la red decidir cuándo pasar de Adam a L-BFGS? Rama A. **Borrador sin firmar** (2026-10-18); etapas A (techo, GPU) y B (señal en lazo abierto, CPU) con una firma; C y D requieren una segunda.
- **EXP-0016 v2** firmado el 2026-10-19 (etapas A y B). Falta la aprobación explícita del uso de la referencia (oráculo y etiquetas de B1) y la prueba de equivalencia en GPU antes del bloque grande; la etapa C requiere segunda firma.
- **EXP-0016 v2** puntuado el 2026-10-19 (etapas A y B, `mechanisms/EXP-0016-stall-handoff/results.md`): H-AWARE-1 y H-HANDOFF-2 no se sostienen; decisión pendiente (`decision.md`).
- **EXP-0017** puntuado el 2026-10-20 (`mechanisms/EXP-0017-representation-battery/results.md`): H-LOC-1, H-ATTR-1 y H-ATTR-2 no se sostienen; S1 es artefacto de la restricción dura; decisión pendiente (`decision.md`).
- **EXP-0018** (`mechanisms/EXP-0018-responsibility-ceiling`): techo oráculo del foco por responsabilidad. v2-draft del 2026-10-20 con pruebas previas en CPU hechas; **sin firmar**.
- **EXP-0018 v2** firmado el 2026-10-20 (criterios, umbrales y uso de la referencia como oráculo). Pendiente la prueba 7 en T4 antes del bloque completo.

## Actualización 2026-10-20 (tarde)

- EXP-0016 y EXP-0017 cerrados en negativo. EXP-0018 (techo oráculo de la responsabilidad) firmado v2 el 2026-10-20. Borradores sin firmar: EXP-0022 (diagnóstico de condicionamiento, CPU), EXP-0023 (precondicionamiento por bloque de arista, condicionado al 0022) y EXP-0024 (protocolo de túnel de viento). El número 0019 queda reservado para la sustitución de A* por un estimador (ver spec del 0018). Ver `registry/proposals.jsonl`.
- **EXP-0018** puntuado el 2026-10-20 (`mechanisms/EXP-0018-responsibility-ceiling/results.md`): H-CEIL-1 no se sostiene (con el error real no hay techo); se cierra la familia «foco por responsabilidad por base»; decisión pendiente (`decision.md`).
- **2026-10-20 (22:01) EXP-0024:** idea A cerrada como especifica de Allen-Cahn (firmado); regla de salida firmada: si la ronda 1-B no pasa T1, termina la busqueda de mecanismo en Burgers/Allen-Cahn. Ver `mechanisms/EXP-0024-wind-tunnel-protocol/T1-plan.md`.
- **2026-10-20 (23:54) EXP-0023:** enmienda 1 firmada (normalizacion de P, criterio 7b, resolucion de budget.md). Ver `mechanisms/EXP-0023-gram-preconditioning/amendment-1.md`.

- **2026-10-21 (noche) EXP-0025** (`mechanisms/EXP-0025-observer-v0`): ¿un observador que elige palanca con un sondeo iguala a la mejor palanca fija por problema? **(Firmado el 2026-10-21 23:57, ver línea siguiente.)** No re-propone las familias cerradas de EXP-0014 a 0018. Depende de la ronda 1-C de EXP-0023; la fase 0 (CPU) no. Ver `registry/proposals.jsonl`.
- **2026-10-21 (23:57) EXP-0025:** firmado v1 por el investigador (valores cerrados en la sección «Cerrado al firmar» del spec). GPU bloqueada hasta las pruebas 1–9, la ronda 1-C puntuada y el presupuesto en `budget.md`. Semillas T1 2001–2005 / V 2101–2105.
- **2026-10-22 EXP-0023 ronda 1-C:** firmada v1 por delegación del investigador (umbrales sin cambio); **no lanzada** (falta kernel T1/V de 1-C, aprobación de campaña y acceso a Kaggle). Sin puntuación hasta que existan datos. Ver `round1C-proposal.md`.
- **2026-10-22 EXP-0023 ronda 1-C puntuada (T1+V):** alcance a la familia Allen–Cahn no refutado (E 0.306, F 0.514 agrupado, 10/10); variante **E** fijada para EXP-0025; control KPP: E y F empeoran; dosis no respaldada; cond(P) sin evidencia (bloque P no corrido). Ver `results/c1-V/RESULTS_V.md`.
- **2026-10-22 EXP-0025 Tier 1:** reloj O/B 1.34–1.35 (≤ 1.5), `budget.md` escrito; condiciones previas 1–3 cumplidas, la 4 (aprobación de campaña) pendiente. Observador casi inerte (15/16 decisiones a0); la guarda γ vetó `aG` en una de dos semillas. Ver `EXP-0025-observer-v0/budget.md`.
- **2026-10-22 EXP-0025 bloque T1 puntuado:** H-OBS-1 **rechazada**; H-OBS-2 no rechazada solo frente a O−m (≈ B), O pierde frente al O− uniforme en AC; H-OBS-3 no respaldada. Hallazgo: aG fijo (E, K = 500, sin recálculo en L-BFGS) mejora B en AC (0.204) y Burgers (0.260), contra 1-B en Burgers (5.70): discrepancia sin explicar. Ver `EXP-0025-observer-v0/results/block-T1/RESULTS.md`.
- **2026-10-22 EXP-0026** (`mechanisms/EXP-0026-e-burgers-diagnostic`): ¿por qué E ayuda en Burgers en un bloque y daña en otro? Factorial cadencia {500, 1000} × recálculo al arrancar L-BFGS {sí, no} + B, 10 semillas nuevas (3001–3010). **Firmado v1 por el investigador.** H-DX-1 y H-DX-2 con umbral 0.90 y ≥ 7/10.
- **2026-10-22 EXP-0023 bloque P puntuado:** contraste de cond(P) en Burgers **no respaldado** (ρ = +0.59, dirección contraria; cond(P) casi constante entre semillas). E/B en Burgers: mediana 0.383, 7/10. Ver `results/c1-P/RESULTS_P.md`.
- **2026-10-22 EXP-0026 puntuado:** H-DX-1 y H-DX-2 **no respaldadas** (el recálculo al arrancar L-BFGS no es el factor; la cadencia 500 no pasa el conteo 7/10). El fallo de 1-B V **se reproduce con el código y las semillas** (no es ruido de corrida). Allen–Cahn incompleto (6/10 por el tope de despacho). Ver `EXP-0026-e-burgers-diagnostic/results/RESULTS.md`.

## Actualización 2026-10-23 a 2026-10-26

- **2026-10-23 EXP-0027** (`mechanisms/EXP-0027-seed-predictor`): ¿predice una señal temprana qué semillas sufren con E en Burgers? Puntuado con los datos de EXP-0026 (CPU, n = 20): P1 (E500, c500) **no respaldado** (ρ = +0.180, p = 0.445). Ver `results.md`.
- **2026-10-23 EXP-0028** (`mechanisms/EXP-0028-failure-attribution`): atribución de fallos del mecanismo en el banco (Allen–Cahn, Burgers, Fisher–KPP, NS2d-CG, DFG) y en piezo. **Firmado v1.1** el 2026-10-23. Fases 1a–1e (`results/phase1*`, puntuación en `results/*_scored.md`) y piezo (`results/piezo-*`, `results/piezo_vs_predictions.md`). **Sin `decision.md`** (la decisión es del investigador).
- **2026-10-24/26 EXP-0029** (spec en `mechanisms/EXP-0028-failure-attribution/EXP-0029_spec.md`, sin carpeta propia): brazo M (E + balance de grupos P2 + arranque denso P5 + guarda P4), firmado en el chat el 2026-10-26 («firmo N, no hacen falta pruebas, córrelo»). Corrido en Modal (A100) el mismo día: `results/piezo-exp0029-final/` (10 JSON: M converso 28–30, M directo 28–30, B y E converso 31–32). Faltan M converso 31–32 (OOM en A100-40GB) y B/E directo 31–32. Lectura: `results/EXP0029_vs_predicciones.md`, `LECCIONES_EXP0029.md`, `results/T1_T2_resultados.md`, diseño en `DISENO_FINAL.md`. **Sin `decision.md`.**
- **2026-10-26 EXP-0030** (`mechanisms/EXP-0030-characterization-M`): caracterización descriptiva y post hoc de B, E y M a partir de los datos guardados de EXP-0029 (CPU, sin corridas nuevas). Decisión `inconclusive`, sin claims. Hallazgo metodológico: el efecto de las patas de M frente a E está confundido con el hardware (M solo en A100). Ver `results/RESUMEN.md`.
- **Cómputo cerrado el 2026-10-26** por decisión del investigador (sin crédito en Modal): se reporta lo encontrado. Pendiente solo escritura (Resultados y Discusión en `ProyectoGrad`). Estado consolidado para redactar: `docs/plans/2026-10-26-estado-para-redaccion.md`.
- **2026-10-26 firmas:** `decision.md` de EXP-0028 (`accepted_exploratory`) y `decision-EXP-0029.md` (`not_supported`, con la confusión de hardware declarada); claims C1 a C7 aceptados en `registry/claims.jsonl`; licencia Apache-2.0 (código) y CC BY 4.0 (documentos y datos derivados).
- **Registro y trazabilidad:** `registry/runs.jsonl` quedó completo hasta EXP-0030 (24 entradas añadidas el 2026-10-26, `recorded_by: agent_backfill_2026-10-26`; los backends de las entradas añadidas se infirieron de los kernels y scripts, verificar si se citan). W&B: proyecto `AF_PIKANS` de la entidad `man22611-uvg`; ver `mechanisms/EXP-0028-failure-attribution/wandb_tools/README.md` y `sync_log.json`.
