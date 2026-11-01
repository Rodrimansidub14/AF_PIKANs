# Proceso de ingeniería de AF_PIKANs

Este documento fija el proceso que gobierna el repositorio. Si algo en otro
archivo lo contradice, gana este.

Aprobado por el investigador el 2026-09-22.

---

## 1. Por qué no basta con SDD

Spec-Driven Development asume que el spec determina la salida: escrita la
especificación, la implementación correcta es deducible y los tests deciden si
se llegó. Eso vale para un harness, un parser de manifiestos o un adaptador de
Kaggle.

No vale para un experimento. En investigación la salida no se deduce del spec:
se descubre. Un experimento cuyo resultado se puede predecir no es un
experimento.

Por eso el repositorio opera **dos regímenes** con procesos distintos.

| | Régimen determinista | Régimen empírico |
|---|---|---|
| Objeto | harness, pipeline, graders, tooling LaTeX | mecanismos, modelos, claims |
| Proceso | **SDD**: spec → plan → tests → implementación | **EDD**: pre-registro → suite → scorecard → decisión humana |
| Corrección | decidible | estadística y provisional |
| Quién decide | los tests | el investigador, sobre evidencia |
| Autonomía del agente | total | propone y ejecuta; no acepta |
| Artefactos | `specs/tooling/`, `tests/` | `experiments/`, `evals/`, `registry/` |

## 2. Eval-driven development aplicado a SciML

El régimen empírico sigue el patrón de eval-driven development de la ingeniería
de modelos, con una adaptación decisiva.

| EDD en post-training de LLMs | Equivalente aquí |
|---|---|
| eval set (casos) | `evals/problems/` — PDEs con solución de referencia |
| grader / juez | `evals/graders/` — residuo físico, L2, error en región crítica |
| rúbrica | dirección + umbral + alcance por métrica |
| golden set / regresión | `evals/baselines/` congelados |
| leaderboard | `registry/runs.jsonl` |
| "no hay feature sin eval" | no hay experimento sin suite, métrica y delta declarados antes |
| iteración de prompt/modelo | iteración de mecanismo dentro de la superficie acotada |
| meta-eval del juez | `evals/meta/` — MMS y pruebas de sanidad del grader |

### La adaptación decisiva: el grader es gratis y exacto

En post-training el grader es el problema difícil. "¿Esta respuesta es buena?"
no tiene ground truth, y de ahí salen los reward models, los jueces LLM,
RLAIF y Constitutional AI: toda esa maquinaria existe para *fabricar* una
señal de verdad que no viene con el problema.

En SciML la señal viene con el problema. El operador de la PDE mide el residuo
**sin etiquetas**, y los casos canónicos tienen solución analítica.

**Consecuencia normativa: este proyecto no usa AI graders.** Sustituir un
operador diferencial exacto por un juez aprendido sería cambiar un verificador
exacto por uno ruidoso. El componente "AI for AI" va en el proposer, nunca en
el verifier.

### La amenaza de validez principal

En un PINN el grader y la loss son el mismo operador evaluado de dos maneras.
Si se mide el residuo sobre los mismos puntos de colocación con los que se
entrenó, hay leakage: un mecanismo de focalización puede ganar la métrica
concentrando puntos donde luego se lo mide, sin resolver mejor la PDE.

Para una tesis cuyo objeto de estudio *es* la política de muestreo adaptativo,
esto no es un detalle: es la amenaza de validez central. Las defensas son
estructurales, no de buena voluntad:

1. Cada problema declara una `eval_grid` **densa, fija y determinista**,
   generada desde `problem.yaml` y nunca desde el mecanismo.
2. `evals/` es de solo lectura para todo agente.
3. Un mecanismo no puede importar `evals` (verificado por
   `tests/test_mechanism_isolation.py`).

## 3. Arquitectura: proposer / verifier

```
Investigador → objetivo · hipótesis · presupuesto
     ↓
PROPOSER    literatura · hipótesis · mecanismo · manifiesto · análisis
     ↓      escribe: src/mechanisms/ · experiments/ · var/agent_runs/
INFRA       local CPU · Kaggle · Colab
     ↓      produce: run + artefactos
VERIFIER    evals/ — problemas · graders · suites · baselines
     ↓      SOLO LECTURA. Emite scorecard, nunca "accepted"
REGISTRY    registry/ — runs · proposals · failures · claims
     ↓
Investigador → gate científico → claim → escritura
     ↺
```

**La regla central:** un agente puede poner el build en verde; solo un humano
puede hacer que un claim sea verdadero.

En los laboratorios frontera la separación proposer/verifier es una norma
cultural. Aquí es una restricción **ejecutada**: `.senpi/extensions/` bloquea
por ruta y los tests verifican el bloqueo.

## 4. Tiers de ejecución

La restricción vinculante de este proyecto es **GPU-hora**, no throughput de
agentes. Las cifras de los laboratorios (jornadas-agente por jornada-humana)
salen de organizaciones con cómputo esencialmente ilimitado; aquí hay cuota de
Kaggle y Colab. El loop agent-first solo es viable si Tier 0/1 absorbe la gran
mayoría de las iteraciones.

| Tier | Costo | Contenido | Autoridad |
|---|---|---|---|
| 0 smoke | segundos, CPU | corre, hay artefactos, los graders son sanos (MMS) | agente, libre |
| 1 regresión | minutos, CPU | Heat 1D, presupuesto corto, 2 semillas | agente, libre |
| 2 candidato | horas, GPU | Burgers / Allen-Cahn, ≥5 semillas | presupuesto aprobado |
| 3 tesis | caro | Navier-Stokes | gate `scientific_claim` |

## 5. El ciclo de un experimento

```
pregunta
  → spec.md            pre-registro: suite, métrica, delta, falsación
  → FIRMA HUMANA       status: preregistered. Los umbrales quedan fijos
  → mecanismo          src/mechanisms/, dentro de la superficie declarada
  → Tier 0             ¿el aparato de medición funciona?
  → Tier 1/2           corrida contra la suite
  → scorecard          technical_pass, nunca "accepted"
  → registry           runs.jsonl · proposals.jsonl · failures.jsonl
  → decision.md        EDR. Solo lo escribe el investigador
  → claims.jsonl       claim → run → artefacto → sección LaTeX
  → escritura
```

### La regla eval-first

No se abre un experimento sin declarar antes: **qué suite lo evalúa, qué
métrica tiene que moverse, cuánto tiene que moverse para contar, y qué
resultado lo falsaría.** Sin las cuatro no hay experimento, hay una corrida
exploratoria; esas van a `notebooks/` con rol `exploration` y no producen
evidencia.

### Por qué el pre-registro va antes

Si el umbral se fija después de ver los resultados, el umbral no mide nada:
mide la corrida que ya ocurrió. Es la misma razón por la que `evals/` está
protegido.

Un experimento rechazado **no se rescata bajando el umbral**. Se corrige el
mecanismo o el protocolo y se vuelve a pre-registrar.

## 6. Reglas duras

1. `evals/` es de solo lectura para todo agente, siempre.
2. Un mecanismo no importa `evals`.
3. Los umbrales se derivan de baselines congelados, no se escriben a mano.
4. Un baseline congelado no se edita; se supersede con otro id.
5. Ausencia de medición no es cero. Un grader que no pudo medir devuelve
   `null` con razón, y el gate lo trata como no concluyente.
6. El error global no sustituye una métrica local cuando la hipótesis habla de
   regiones críticas.
7. `accepted` nunca lo escribe un agente.
8. Todo claim traza a una corrida, una corrida traza a un manifiesto, un
   manifiesto traza a un pre-registro firmado.

## 7. Dónde vive cada cosa

| Ruta | Régimen | Contenido |
|---|---|---|
| `specs/tooling/` | determinista | specs y planes SDD del tooling |
| `specs/conventions/` | ambos | convenciones transversales |
| `tests/` | determinista | pruebas del tooling y de las guardias |
| `evals/` | empírico | el verificador. Protegido |
| `experiments/` | empírico | pre-registros, manifiestos, EDRs |
| `registry/` | empírico | memoria append-only del loop |
| `src/mechanisms/` | empírico | superficie escribible del proposer |
| `src/research_pipeline/`, `src/harness_loop/` | determinista | ejecución y gates |
| `context/`, `workflow/` | control | ontología, política, routing |
| `notebooks/` | exploración | sin derecho a producir evidencia |
| `var/`, `artifacts/` | — | regenerable, no versionado |

## 8. Linaje (para citar en la tesis)

- **DSRM** (Peffers et al.) — método de investigación general. Ya citado.
- **Pre-registro** — de la metodología experimental; es el "spec" del régimen
  empírico.
- **Eval-driven development** — práctica de ingeniería de modelos.
- **Benchmark-driven SciML** — PDEBench, PINNacle: suites de evaluación para
  exactamente esta clase de problema.
- **Proposer–verifier / generate-and-verify** — AlphaEvolve, FunSearch.
- **AI-led AI R&D** — reportes públicos de laboratorios frontera sobre agentes
  que escriben código experimental, ejecutan corridas y analizan resultados
  bajo supervisión humana.
- **Goodhart e integridad de evals** — justificación de la separación
  proposer/verifier y del held-out determinista.

La síntesis defendible: *arquitectura proposer–verifier, dirigida por evals,
para investigación SciML asistida por agentes, con autoridad científica
retenida por el investigador.*

## 9. Estado actual

Hecho:

- Separación de repositorios (2026-09-22): el código y la experimentación
  viven en `AF_PIKANs/`; la escritura, en un repositorio privado aparte que
  `src/thesis_workspace/paths.py` resuelve vía `$THESIS_ROOT` o `../ProyectoGrad`.
- `evals/` con Heat 1D (solución analítica verificada por MMS), siete graders
  y 17 meta-evals que pasan.
- Cuatro suites declaradas; Tier 0 y Tier 1 ejecutables.
- `src/mechanisms/` con el mecanismo nulo y el contrato de mecanismos.
- Frontera proposer/verifier ejecutada y cubierta por tests.
- `experiments/EXP-0001` pre-registrado en borrador.
- `registry/` inicializado.

Bloqueante inmediato:

**No hay ningún baseline congelado.** Por eso `pipeline.py gate` devuelve
`threshold_not_defined` en todas las métricas y `technical_pass` es `False` en
cualquier corrida: las cinco métricas de `acceptance_gates` en
`src/harness_loop/experiment_manifest.yaml` declaran `direction` y `required`
pero ninguna declara `threshold`. No es un bug: es el verificador negándose a
aprobar por omisión. Se resuelve cerrando EXP-0001.

Pendiente:

- Firmar EXP-0001 y congelar `heat_1d@model.pikan.baseline`.
- Conectar el harness a las suites (hoy el manifiesto es global y mutable;
  debe pasar a `experiments/<ID>/manifest.yaml`).
- Problemas de Tier 2: Burgers, Allen-Cahn.
