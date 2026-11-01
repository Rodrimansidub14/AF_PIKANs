---
experiment_id: EXP-0008
slug: baseline-v2-competitive
status: decided            # draft | preregistered | running | scored | decided
suite: tier2_candidate
problem_id: [heat_1d, burgers_pinnacle_1d, allen_cahn_1d, dfg_2d]
model_id: rbf_fixed      # + escalera de capacidad (capacity-ladder.md) y controles MLP/B-spline
mechanism_id: baseline_none
hypothesis_id: H-BASE-V2
preregistered_at:
preregistered_by:
---

# EXP-0008 — Baseline competitivo v2 (receta de optimización + capacidad + evaluadores)

> **Estado: BORRADOR para revisión humana.** Ningún archivo de este paquete está
> firmado. No autoriza cómputo. `scientific_accepted=false`,
> `reference_accepted=false`. Los umbrales numéricos de esta página son
> *propuestos*; solo son vinculantes cuando el investigador firma (gate
> `hypothesis`) y congela los baselines citados (gate `eval_integrity`).

Paquete completo (léase en este orden):

| Archivo | Contenido | Punto del análisis |
|---|---|---|
| `spec.md` (este) | pregunta, hipótesis, gates numéricos, falsadores, amenazas, aprobaciones | 6, 7 |
| [`recipe-v2.md`](recipe-v2.md) | receta Adam warmup+decay → L-BFGS full-batch, closure, checkpoints, dry-run local | 1 |
| [`capacity-ladder.md`](capacity-ladder.md) | escalera ancho/bases/hidden, controles MLP/B-spline, regla de selección | 3 |
| [`evaluators-and-references.md`](evaluators-and-references.md) | split V/E/externo, FEM PINNacle 101×11, scorer σn de DFG, `gmsh`/FEM, aceptación de referencias | 2, 5 |
| [`signals-and-records.md`](signals-and-records.md) | señales internas por problema, registro por corrida, esquema `registry/`, backfill | 6 |
| [`rad-gate-and-v2.md`](rad-gate-and-v2.md) | estudio de señal SS-1 y rediseño RAR-D (condicional, fuera del alcance de aprobación) | 4 |
| [`fast-track.md`](fast-track.md) | **ruta acelerada recomendada** (~2 GPU-h; continuación L-BFGS desde checkpoints existentes), con la evidencia de la sonda en CPU | 1–7 |
| [`work-plan.md`](work-plan.md) | diseño completo de respaldo (~17.5 GPU-h): paquetes, tramos, presupuesto, go/no-go | 1–7 |
| [`manifest.json`](manifest.json) | contrato máquina de todo lo anterior | — |

## 1. Pregunta

**¿Cuánto de la brecha entre el baseline RBF-PIKAN v1 y las recetas
reproducidas de la literatura (PINNacle, DeepXDE) se debe a la receta de
optimización y a la capacidad de la red, y no a la arquitectura?**

Y, como entregable: **¿alcanza la RBF-PIKAN, con la receta v2 y bajo un
evaluador declarado, la *banda* de las recetas reproducidas en Heat, Burgers,
Allen–Cahn y DFG Re=20?**

## 2. Diagnóstico que motiva el experimento

Evidencia ya medida (fuentes en `var/agent_runs/`; cifras de
`baseline-competitive-2026-09-27/scorecard-current.csv` y de
`burgers-rad-exp006-v4/.../analysis/paired-runs.csv`):

| Observación | Cifra | Lectura |
|---|---|---|
| Heat, RBF con L-BFGS (S2) vs solo Adam (v4) | 2.06e-6 vs 1.68e-4 (~80×) | el optimizador, no la arquitectura, movió 2 órdenes |
| Burgers CH 1025×129, RBF v1 vs MLP PINNacle Adam→L-BFGS | 3.95e-2 vs 5.66e-4 (~70×) | brecha isocontrato ya medida, mismo evaluador |
| Burgers, MLP PINNacle solo Adam vs Adam→L-BFGS | 9.08e-3 → 5.66e-4 (~16×) | L-BFGS es el factor dominante en la receta publicada |
| Allen–Cahn, RBF: Adam vs +240 s L-BFGS | 1.65e-2 → 9.16e-3 (1.8×) | L-BFGS ayuda aun con presupuesto mínimo |
| DFG 301–305, RBF: Δp antes/después de L-BFGS | 34% → 9.9% | idem. **Actualización WP-E3:** con el scorer σn el Cd pasa de 1.46 (74 % de error, artefacto del scorer tangencial) a 5.12 (8.2 %) |
| Burgers, RAD sobre RBF v1 | ×5.9 peor (IC95 2.8–12.1), 5/5 parejas | mecanismo aplicado sobre base no convergida y con diseño frágil |
| Parámetros: RBF v1 vs MLP PINNacle | 3.6k vs 40.8k | la comparación de la literatura tiene 11× más capacidad |

Dos hechos adicionales de proceso: `registry/*.jsonl` está vacío (0 líneas) y
cuatro fallos triviales de kernel (título/slug, `base64`, `gmsh`,
`KeyError: seeds`) consumieron la reserva de infraestructura que faltó para la
referencia FEM de DFG.

## 3. Hipótesis (falsables)

| ID | Afirmación | Problema |
|---|---|---|
| H1 (receta) | Con arquitectura, pools, semillas e iniciales idénticos a v1, la receta v2 reduce el error CH L2 relativo al menos 4× (media geométrica pareada). | Burgers |
| H2 (capacidad) | Si tras H1 sigue existiendo brecha frente a la banda PINNacle, algún rung de la escalera reduce el error ≥2× frente a R0 bajo la misma receta (≥1.5× es el mínimo para *elegir* un rung; entre 1.5× y 2× el resultado es inconcluyente). | Burgers, Allen–Cahn |
| H3 (transferencia) | La receta congelada en Burgers transfiere a Allen–Cahn: mejora ≥2× sobre el L-BFGS de 240 s de S3. | Allen–Cahn |
| H4 (regresión) | La receta v2 conserva el resultado de S2 en Heat (no lo empeora >5×). | Heat |
| H5 (evaluador) | Con un scorer σn único, las cohortes DFG 301–305 y 501–505 son coherentes entre sí (Cd y Δp dentro de 2× de error relativo). | DFG |
| H6 (evaluador externo) | El RBF v1 puntuado en el archivo FEM 101×11 de PINNacle muestra un error entre 1× y 30× el error CH, como en la reproducción PINNacle (23×). | Burgers |

H5 y H6 son mediciones de bajo costo (CPU) que fijan el suelo del resto.

## 4. Diseño (resumen; detalle en los documentos enlazados)

Cuatro etapas y dos semillas-roles disjuntos:

| Etapa | Qué | Semillas | Qué puntúa |
|---|---|---|---|
| **A — Selección (dev)** | A1: receta en Burgers; A2: escalera de capacidad en Burgers y AC. | **dev**: 611–613 (Burgers, A1), 611–612 (Burgers A2), 621–622 (AC A2) | Solo la grilla **V** (`evals.checkpoint_validation`, 129×65) y diagnósticos internos. **No** se abre E. |
| **B — Congelación** | El investigador congela receta y rung ganador en `manifest.json` (versión nueva). | — | Regla de selección fijada de antemano (§6). |
| **C — Confirmación** | Receta y rung congelados, cinco semillas **pareadas con v1** (mismos pools/inicializaciones): Burgers 201–205, AC 401–405, Heat 901–905, DFG 301–305. | pareadas | **E** (mallas protegidas actuales) + evaluadores **externos** independientes de la selección. |
| **D — Módulos condicionales** | RAR-D (tras SS-1); bases aprendibles; causal-training en AC. **No** están en el alcance de aprobación de este preregistro. | — | — |

Por qué las semillas dev son distintas de las de confirmación: la selección
sobre semillas de confirmación filtraría información de E hacia la receta.
Por qué las de confirmación son las de v1: permite una diferencia **pareada**
v2−v1 (mismos puntos, misma inicialización) con mucha menos varianza que
comparar cohortes (los errores v1 de Burgers van de 0.021 a 0.083 según la
semilla).

## 5. Métricas y gates propuestos

### 5.1 Baselines de referencia que hay que congelar (gate `eval_integrity`)

Las cifras salen de artefactos existentes; no se editan, se referencian con
id. Regla 4 del proceso: no se modifica un baseline congelado.

| ID propuesto | Contenido | Valor (media ± DE muestral, n) | Fuente |
|---|---|---|---|
| `burgers_pinnacle_1d@rbf_fixed.v1-adam20k` | RBF v1, CH 1025×129, L2 rel. | 3.955e-2 ± 2.565e-2 (5; sem. 201–205) | `burgers-rad-exp006-v4/.../paired-runs.csv` |
| `burgers_pinnacle_1d@pinnacle_adam_lbfgs.repro` | MLP 5×100, Adam 5k→L-BFGS, CH 1025×129 | 5.659e-4 ± 1.680e-4 (5) | `scorecard-current.csv` |
| ídem, evaluador FEM 101×11 (`burgers1d.dat`) | mismo modelo | 1.3264e-2 ± 2.88e-5 (5); publicado 1.33e-2 ± 5.3e-5 (3) | ídem |
| `heat_1d@rbf_fixed.s2-adam-lbfgs` | RBF, 20k Adam + L-BFGS, 512×256 | 2.061e-6 ± 2.31e-7 (5; 901–905) | ídem |
| `heat_1d@deepxde_1.12.2_a0.1` | receta DeepXDE reproducida | 5.807e-4 ± 4.04e-4 (5; sem. 11–55, **no pareadas**) | ídem |
| `allen_cahn_1d@rbf_fixed.s3-lbfgs240` | RBF, L-BFGS 240 s, ETDRK4 candidata | 9.164e-3 ± 2.66e-3 (5; 401–405) | ídem |
| `allen_cahn_1d@deepxde_official` | receta DeepXDE oficial vs `.mat` | 2.005e-2 ± 6.12e-3 (5) | ídem |
| `dfg_2d@rbf_fixed.v1-301-305.sigma-n` | RBF, post-L-BFGS, QoI vs FEATFLOW, scorer σn único | Cd 5.124 ± 0.053 (8.2 %), Δp 0.1059 (9.9 %), Cl 0.039 ± 0.099 | `var/agent_runs/exp0008/wp-e3/` (WP-E3, hecho). **Sustituye** al Cd 1.463 (74 %) archivado, que provenía del scorer tangencial |

### 5.2 Gates por problema

Convención: **G1** = efecto de receta (pareado v2 vs v1, mismas semillas y
pools); **G2** = banda competitiva (frente a una receta *reproducida* bajo
evaluador declarado); **G3** = paridad/estirar (solo se reporta, no decide).
**Guardas** impiden ganar la primaria rompiendo otra cosa. Todos exigen 5/5
semillas finitas: una corrida divergente cuenta como resultado.

**Burgers canónico** (evaluador primario: CH 1025×129 = E; externo: FEM 101×11)

| Gate | Condición |
|---|---|
| G1 | razón geométrica pareada `L2(v2)/L2(v1) ≤ 0.25` y límite superior del IC95 t (log-razón, n=5) `< log 0.5` |
| G2 | FEM 101×11: media ≤ **1.99e-2** (1.5× la reproducción) y peor semilla ≤ 3.0e-2; **y** CH 1025×129: media ≤ **1.13e-3** (2× la reproducción); ambos, sobre el rung elegido |
| G3 | FEM media ≤ 1.33e-2 y CH media ≤ 5.7e-4 (paridad con la receta publicada) |
| Guardas | RMSE de campo en el frente < v1 (0.2161); error máx. de IC/BC ≤ 2e-2; tiempo por corrida ≤ 3× v1 (≤ ~3100 s) |

Nota sobre el evaluador FEM 101×11: en la reproducción PINNacle el mismo
modelo da CH 5.66e-4 y FEM 1.3264e-2 (Adam→L-BFGS), y CH 9.08e-3 y FEM 1.72e-2
(solo Adam). El archivo FEM se comporta como un **piso ≈ 1.3e-2** respecto de
Cole–Hopf: la parte FEM de G2 mide comparabilidad con la tabla publicada, no
discrimina modelos por debajo de ese piso. **La condición CH es la
vinculante.** WP-E1 mide el piso directamente (`evaluators-and-references.md §2`).

**Heat 1D** (E = 512×256; externo: malla tutorial DeepXDE 256×201)

| Gate | Condición |
|---|---|
| G1 (regresión) | media RBF ≤ **1.0e-5** (5× el S2) en 5/5 semillas |
| G2 | media RBF ≤ 5.8e-4 (media DeepXDE reproducida) con el mismo presupuesto de pasos (20 000 Adam + L-BFGS ≤ 15 000 iter.) |
| Guarda de completitud | un brazo que no alcanza los 20 000 pasos Adam se marca `incomplete` y **no entra a rankings** (caso S2 B-spline) |

**Allen–Cahn 1D** (E = 512×256 sobre ETDRK4/FD8192 candidatas; externo: `.mat` DeepXDE 201×101)

| Gate | Condición |
|---|---|
| G1 | media (mejor rung) ≤ **4.6e-3** (= 0.5× `s3-lbfgs240`) |
| G2 | media ≤ **2.0e-3** (10× mejor que la receta DeepXDE oficial reproducida), en ambas referencias |
| G3 | ≤ 1e-3; AEH-PINN (8.12e-5, 100k épocas, otro evaluador) **no** es meta de este experimento |
| Guardas | traza IC/BC ≤ 1e-12 (transformación dura); simetría par en x (error de evenness) ≤ error de campo; residual normalizado ≤ v1 |
| Condición previa | referencia aceptada por el humano antes de **enunciar** cualquier claim AC (no antes de correr) |

**DFG 2D-1 Re=20** (QoI vs FEATFLOW: Cd 5.57953523384, Cl 0.010618948146, Δp 0.11752016697)

| Gate | Condición |
|---|---|
| G0 | WP-E3: reprocesar 301–305 y 501–505 con **un** scorer σn (H5) antes de cualquier decisión |
| G1 | Cd y Δp con error relativo ≤ 10 %, `|Cl−0.01062| ≤ 0.03` (media de 5 semillas) |
| G2 | Cd y Δp ≤ 5 %, `|ΔCl| ≤ 0.02`; error de campo u,v,p vs FEM solo se **reporta** hasta que la referencia FEM esté aceptada |
| Ejecución | condicionada al resultado de G0 (si 501–505 ya cumple G1 con el scorer correcto, v2 se corre para cerrar G2 y las señales) |

### 5.3 Qué cuenta como "baseline competitivo" (entregable siguiente)

Por problema, un veredicto en la escala ya usada
(`comparable_to_literature` / `screening_only` / `not_comparable`), con:

- Burgers: `comparable_to_literature` si G2 se cumple en el rung elegido y en
  R0 se reporta la brecha; **no** implica SOTA (vRBA queda fuera).
- Heat: `comparable_to_literature` si G1 y G2 se cumplen y todos los brazos son
  `complete`.
- Allen–Cahn: `screening_only` mientras la referencia no esté aceptada,
  aunque G1/G2 se cumplan.
- DFG: `screening_only` mientras no exista referencia FEM aceptada.

Un `comparable_to_literature` no es un claim; el claim lo decide el humano
(gate `scientific_claim`).

## 6. Regla de selección de la etapa A (fijada antes de ver datos)

1. Métrica de selección: media geométrica sobre semillas dev del L2 relativo en
   la grilla **V** tras L-BFGS. E no se abre en la etapa A.
2. Se elige la configuración de menor métrica. Si dos quedan dentro de 10 %,
   gana la de menor costo (segundos de GPU por corrida, mediana).
3. Una configuración solo es elegible si 3/3 (2/2) semillas dev terminan
   finitas y ninguna semilla empeora >2× respecto a la configuración v1
   evaluada en las **mismas** semillas dev (`C1`, misma receta que v1).
4. La escalera de capacidad puede *no* ganar: si ningún rung mejora ≥1.5×
   sobre R0 en ambos problemas, se congela R0 y H2 queda falsada para esa
   escala.
5. Empates o resultados no concluyentes (n=2–3) se resuelven a favor de la
   receta/rung más simple y de menor costo, y se declaran como tales; no se
   añaden semillas dev después de ver el ranking.

## 7. Condiciones de falsación

- **H1 falsada** si la razón geométrica pareada en Burgers es ≥ 0.5 o el IC95
  incluye 1. Consecuencia: la receta no es la palanca principal; la
  capacidad/arquitectura pasan a primer plano (H2) y se revisa el diagnóstico.
- **H2 falsada** si ningún rung mejora ≥1.5× sobre R0 en ambas semillas dev
  de cada problema; **inconcluyente** si el mejor está entre 1.5× y 2×;
  **apoyada** si es ≥2× en ambas semillas. Consecuencia de la falsación: la
  brecha residual no es de capacidad; queda abierta la geometría de bases
  (hipótesis separada).
- **H3 falsada** si en AC la mejora sobre S3 es <1.5×.
- **H4 falsada** si el RBF v2 en Heat queda >2e-5.
- **H5 falsada** si tras el scorer único Cd difiere >2× entre cohortes: el
  scorer no explica la discrepancia y aparece una causa de entrenamiento
  (protocolo 301–305 vs 501–505) que hay que aislar antes de DFG-v2.
- El experimento **no se rescata bajando umbrales**: un rechazo se corrige con
  un protocolo nuevo y un nuevo preregistro (proceso §5).

## 8. Presupuesto (resumen; desglose en `work-plan.md`)

| Tramo | Backend | Estimado de sesión | Tope duro |
|---|---|---:|---:|
| T0 dry-run, reescoreos, FEM, registro | CPU local | 0 GPU-h | — |
| B0 micro-benchmark de tiempos | Kaggle T4×2 | ~0.25 h | 0.30 h |
| T1 A1 receta Burgers (12 corridas dev) | Kaggle T4×2 | ~2.3 h | 3.0 h |
| T2 A2 escalera (22 corridas dev, incluye chequeo AC) | ídem | ~4.5 h (3.7–6.4) | 6.5 h |
| T3a C Burgers (20 corridas) | ídem | ~3.9 h | 5.0 h |
| T3b C Allen–Cahn (15 corridas [+5]) | ídem | ~3.0 h [+0.7 h] | 4.0 h |
| T3c C Heat, RBF (5 corridas) | ídem | ~0.5 h | 1.0 h |
| T4 C DFG (10 corridas, condicional a G0) | ídem | ~3.0 h | 4.0 h |
| **Núcleo (~90 corridas)** | | **~17.5 h** | **~23.8 h** |

Los estimados son calibrables (±50 %) con el micro-benchmark WP-B0; **cada
tramo requiere su propia aprobación de presupuesto** (Tier 2) y no se lanza el
siguiente sin registrar el anterior. Cuota: último valor verificado por CLI
15.75 h (antes de S2); estimación posterior 14.4 h **no verificada**;
refresco anunciado 2026-10-03. Se recomienda no lanzar T1+ antes del refresco.

## 9. Superficie de edición declarada

Agente (tras firma y aprobaciones de §11):

- `experiments/baseline-v2/EXP-0008-*` (este paquete), `var/agent_runs/exp0008/`.
- Código nuevo en `src/sciml/` (módulos `recipe.py`,
  `lbfgs.py`, `preflight.py`, `external_scorers` **en borrador** bajo `var/`),
  con SDD y pruebas en `tests/`. Requiere aprobación `code_surface`
  (§11, punto 2) porque toca `model.py` (hidden/width) y `train.py`.
- `registry/runs.jsonl` (rol `experiment_runner`) y `registry/failures.jsonl`
  (rol `evaluator`).

No puede tocar: `evals/`, `registry/claims.jsonl`, `context/`, `workflow/`,
la definición de las PDE ni los pools de v1 (se leen, no se regeneran).

## 10. Amenazas de validez

- **E ya fue vista.** Las mallas E de Heat/Burgers/AC se han usado para puntuar
  v1 y las campañas de cierre. Ninguna decisión de receta salió de un ranking
  de E, pero la exposición existe: por eso la selección va sobre V + semillas
  dev, y la confirmación añade evaluadores **externos** (FEM PINNacle, `.mat`
  DeepXDE, malla tutorial, FEATFLOW) que no intervinieron en nada.
- **Leakage de residual/referencia.** Entrenamiento y selección no importan
  `evals`; el scoring es un job aparte posterior al congelado.
- **Confusión de factores en la etapa A1.** C1..C4 cambian schedule, lr y
  batch; solo C1 vs C2 aísla el schedule y solo el par (Adam-end, L-BFGS-end)
  dentro de cada corrida aísla L-BFGS. Esto se declara; la etapa A no
  pretende atribución causal fina, solo elegir una receta.
- **Comparación no isocosto.** La receta v2 gasta más cómputo que v1.
  Se reportan curvas error-vs-pasos y error-vs-segundos por fase, y el gate de
  tiempo (≤3× v1). Un resultado mejor a mayor costo no es "mejor arquitectura".
- **Capacidad vs. paridad de parámetros.** La escalera rompe la igualdad de
  parámetros con el MLP de 3.6k; por eso los MLP se corren en dos tamaños
  (3.7k y 40.8k) y se reporta parámetros/costo junto a cada error.
- **n pequeño.** 2–3 semillas dev deciden; 5 confirman. Sin claims
  estadísticos de superioridad general.
- **Referencias no aceptadas** (AC, DFG): los scores son `candidate_reference`.
- **Determinismo.** `deterministic=True` y `CUBLAS_WORKSPACE_CONFIG` no
  garantizan identidad bit a bit entre dispositivos; se guardan versiones y
  hashes de pools.
- **Estimados de tiempo.** Son extrapolaciones (v1 Burgers ≈ 1020 s/20k pasos;
  PINNacle L-BFGS ≈ 1949 s para 5 semillas en T4×2); se calibran antes de
  comprometer tramos.

## 11. Aprobaciones que se solicitan (todas humanas)

| # | Gate | Qué se aprueba |
|---|---|---|
| 1 | `hypothesis` | Firma de este preregistro (hipótesis, semillas, gates §5, regla §6, presupuesto §8). Cambia `status` a `preregistered`. |
| 2 | `code_surface` | Modificar `model.py` (`hidden`, `width_scale`), `train.py` (fases de receta) y añadir módulos de receta/L-BFGS/preflight. |
| 3 | `eval_integrity` | Congelar los baselines de §5.1 con sus ids; fijar los umbrales de §5.2; alojar los scorers externos (FEM PINNacle, σn DFG) en `evals/` (el agente entrega borrador en `var/`, el humano lo incorpora). |
| 4 | presupuesto Tier 2 | Aprobación **por tramo** (T1…T4). |
| 5 | rol de registro | Quién escribe `registry/proposals.jsonl` (hoy ningún rol de `agent-policy.toml` tiene ese permiso). |
| 6 | cómputo local | Autorizar FEM de referencia DFG y reescoreos en **CPU local** (instrucción previa: entrenamientos de baseline en GPU; esto no es entrenamiento). |
| 7 | lectura externa | Descargar en solo lectura `https://raw.githubusercontent.com/i207M/PINNacle/6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf/ref/burgers1d.dat` (hash esperado `3b6d31d8…5eb`) para WP-E1. Se anuncia antes de ejecutar; sin escritura fuera del repositorio. |

## 12. Firma

El preregistro entra en vigor cuando el investigador cambia `status` a
`preregistered` y añade fecha. A partir de ahí los umbrales no se mueven.
`decision.md` lo escribe solo el investigador tras ver el scorecard.
