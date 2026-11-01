# AF_PIKANs

Infraestructura de investigación para el trabajo de graduación sobre
mecanismos de focalización adaptativa en arquitecturas PIKAN.

Este repositorio contiene **el código y la experimentación**. La escritura del
documento (LaTeX, figuras, plantilla, bibliografía, presentaciones) vive en un
repositorio privado aparte, por defecto `../ProyectoGrad`.

El proceso de ingeniería que gobierna el repositorio está en
**[`specs/proceso-de-ingenieria.md`](specs/proceso-de-ingenieria.md)**. Empezar ahí.

## Arquitectura

```
Investigador → objetivo · hipótesis · presupuesto
     ↓
PROPOSER    literatura · hipótesis · mecanismo · manifiesto · análisis
     ↓
INFRA       local CPU · Kaggle · Colab
     ↓
VERIFIER    evals/ — solo lectura para todo agente
     ↓
REGISTRY    registry/ — runs · proposals · failures · claims
     ↓
Investigador → gate científico → claim → escritura
```

Un agente puede poner el build en verde; solo un humano puede hacer que un
claim sea verdadero.

## Límites del repositorio

| Ruta | Responsabilidad |
|---|---|
| `specs/` | Proceso, specs SDD del tooling y convenciones |
| `evals/` | **Verificador: problemas, graders, suites, baselines. Protegido** |
| `experiments/` | Pre-registros, manifiestos y decisiones por experimento |
| `registry/` | Memoria append-only del loop |
| `src/mechanisms/` | Superficie de código escribible por el agente |
| `src/research_pipeline/` | Validación, ejecución local y remota, gates |
| `src/harness_loop/` | Manifiestos y ejecución controlada de corridas |
| `src/graph_engineering/` | Validación de contratos y grafo de PIKAN-Vault |
| `src/thesis_workspace/` | Puente hacia el repositorio de escritura |
| `context/` | Ontología, contrato de notebooks y política de agentes |
| `workflow/` | Routing de prompt a rol |
| `notebooks/` | Exploración interactiva, sin derecho a producir evidencia |
| `tests/` | Pruebas del tooling y de las guardias de la arquitectura |
| `.senpi/extensions/` | Adaptador local que aplica los contratos en cada prompt |
| `PIKAN-Vault/` | Base de conocimiento y notas de investigación |
| `tools/` | Verificación del repo: convenciones, cifras citadas, traza, filtro de notebooks |
| `var/`, `tmp/`, `artifacts/` | Regenerables, ignorados por Git |

## Trazabilidad y reproducibilidad

- **De la pregunta al dato:** `experiments/TRAZA.md` y `experiments/status.json` (generados) cruzan cada experimento con su estado, decisión, bloques de corridas del registro y notebooks. Cada bloque está en `registry/runs.jsonl`; las curvas de las corridas, en el proyecto W&B `AF_PIKANS`.
- **De la cifra al dato:** `python tools/reproduce_numbers.py` recalcula las cifras de piezo citadas en la tesis solo desde los JSON versionados y falla si una no coincide.
- **Convenciones:** nombres de experimentos, resultados, notebooks y ramas en [`specs/conventions/convencion-nombres.md`](specs/conventions/convencion-nombres.md). Los notebooks se versionan sin salidas (`python tools/setup_git.py` una vez por clon).
- **Qué no está aquí y cómo verificar:** [`docs/REPRODUCIR.md`](docs/REPRODUCIR.md). Estado para redactar: [`docs/plans/2026-10-26-estado-para-redaccion.md`](docs/plans/2026-10-26-estado-para-redaccion.md).

## El repositorio de escritura

`src/thesis_workspace/paths.py` resuelve su ubicación en este orden:

1. la variable de entorno `THESIS_ROOT`;
2. `../ProyectoGrad`;
3. nada — y entonces las pruebas que lo necesitan se **saltan**, no fallan.

Esto permite clonar y testear este repositorio sin el repositorio privado.

## Comandos

Validar contratos y construir el grafo:

```powershell
py -3.12 -m src.graph_engineering validate --root .
py -3.12 -m src.graph_engineering audit-notebooks --root .
py -3.12 -m src.graph_engineering build --root . --out var/context
```

Tier 0 — meta-evals del verificador (los graders miden lo que dicen medir):

```powershell
py -3.12 -m unittest discover -s evals/meta -p "test_*.py"
```

Pruebas del tooling y de las guardias:

```powershell
py -3.12 -m unittest discover -s tests -t .
node --test tests/senpi_context_core.test.mjs tests/senpi_extension_smoke.test.mjs
```

Pipeline de experimentos:

```powershell
py -3.12 src/research_pipeline/pipeline.py validate --manifest src/harness_loop/experiment_manifest.yaml
py -3.12 src/research_pipeline/pipeline.py run-local --manifest src/harness_loop/experiment_manifest.yaml --command py -3.12 src/research_pipeline/examples/train_stub.py
py -3.12 src/research_pipeline/pipeline.py gate --manifest src/harness_loop/experiment_manifest.yaml --summary <run_summary.json>
```

## Estado

`gate` devuelve hoy `threshold_not_defined` en todas las métricas porque no
hay ningún baseline congelado en `evals/baselines/`. Es el comportamiento
correcto: el verificador se niega a aprobar por omisión. Se resuelve cerrando
`experiments/EXP-0001-heat1d-baseline`.

## Flujo con Senpi

La extensión local valida la ontología, actualiza el grafo de `PIKAN-Vault`,
enruta el prompt contra `workflow/index.toml` e inyecta el contexto antes de
cada turno. El router selecciona flujo y superficie según el lenguaje natural:

```text
resume las fuentes relevantes para el marco teórico
propone un mecanismo de focalización guiado por residuo
ejecuta el manifiesto en dry-run
compara las métricas de las corridas
```

No es necesario usar `/workflow` ni `research-role=...`; queda como override.
En Senpi, confía el proyecto si se solicita y ejecuta `/reload`; luego
`/research-context` muestra el estado del contexto derivado.
