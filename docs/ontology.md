# Ontología y graph engineering de investigación

## Propósito

Este repositorio usa una ontología para convertir preguntas de investigación en notebooks, corridas y evidencia trazable. La ontología no es un catálogo decorativo: define qué debe declarar un experimento, qué puede hacer un agente y qué resultado requiere aprobación humana.

La fuente declarativa está en `context/`. `PIKAN-Vault/PIKAN-Vault` es el corpus de literatura y conceptos. Los mapas generados se escriben en `var/context/`.

## Ciclo operativo

```text
pregunta
  → literatura y conceptos
  → hipótesis falsable
  → manifiesto de experimento
  → notebook parametrizado
  → corrida reproducible
  → métricas y artefactos
  → gates
  → decisión humana
  → claim o siguiente hipótesis
```

El notebook es una interfaz de investigación. El manifiesto fija los valores de la corrida y el harness conserva su registro. El vault aporta fuentes y conceptos, pero no convierte una nota en evidencia experimental.

## Contratos

| Archivo | Declara |
|---|---|
| `context/research-ontology.toml` | entidades, relaciones, vocabularios e invariantes |
| `context/notebook-contract.toml` | roles, metadatos y salidas mínimas |
| `context/agent-policy.toml` | permisos, rutas de escritura y gates humanos |
| `workflow/index.toml` | catálogo declarativo para seleccionar workflow y rol por intención |
| `src/harness_loop/experiment_manifest.yaml` | valores concretos de una iteración |

El primer nivel de autonomía es `L2_human_verified_ai_execution`. Los agentes pueden preparar trabajo y ejecutar tareas acotadas; el investigador conserva la autoridad sobre el código central, la aceptación del experimento y la escritura de claims.

## Graph engineering

La herramienta `src.graph_engineering` construye un grafo determinista de notas y recursos:

```powershell
py -3.12 -m src.graph_engineering validate --root .
py -3.12 -m src.graph_engineering audit-notebooks --root .
py -3.12 -m src.graph_engineering build --root . --out var/context
```

El comando `build` produce:

- `var/context/research-graph.json`: nodos, relaciones y diagnósticos;
- `var/context/notebook-audit.json`: estado de cada notebook;
- `var/context/validation.json`: errores y advertencias de contratos;
- `var/context/GRAPH_REPORT.md`: resumen legible para revisión humana.

### Reglas del grafo

1. Cada nota recibe un ID estable `vault:<ruta-relativa-sin-extensión>`.
2. Los wikilinks se resuelven por ruta, nombre de archivo o título.
3. Un destino único crea `links_to`.
4. Un destino faltante o ambiguo queda en el diagnóstico.
5. `_templates/` no se considera conocimiento curado.
6. La herramienta no ejecuta ni modifica notas o notebooks.
7. Las relaciones semánticas como `supports` o `evaluates` deben declararse; no se infieren desde proximidad textual.

## Roles de agentes

| Rol | Producto permitido | Decisión que conserva el investigador |
|---|---|---|
| `literature_scout` | evidencia y enlaces propuestos | promover una fuente al corpus curado |
| `hypothesis_planner` | hipótesis y manifiesto borrador | activar hipótesis y gates |
| `notebook_engineer` | notebook dentro de la superficie autorizada | modificar PDE o modelo central |
| `experiment_runner` | corrida y artefactos | ejecutar backend costoso o externo |
| `evaluator` | comparación y decisión borrador | aceptar el resultado |
| `research_synthesizer` | borrador de informe o LaTeX | publicar un claim |

## Ejecución desde Senpi

El repositorio incluye una extensión local en
`.senpi/extensions/pikan-research-context.ts`. Senpi la carga únicamente para
este proyecto y la ejecuta antes de cada prompt. La extensión proyecta los
contratos en `var/context/senpi-context.json`, carga el grafo actualizado,
selecciona el workflow de `workflow/index.toml` y añade al contexto del modelo
el rol, las fases, las rutas permitidas, los gates y los nodos relevantes para
la pregunta.

El router usa `literature_scout` como fallback conservador. En el uso normal el
usuario escribe lenguaje natural:

```text
adapta el baseline Heat en el notebook
```

La extensión selecciona `notebook_engineer` automáticamente. La sintaxis
`research-role=...` permanece como override explícito, no como requisito.

La política nativa de filesystem de Senpi niega escrituras fuera del proyecto,
en los contratos canónicos, en `workflow/`, en el vault, en `var/context/` y en
los archivos de control de Senpi. Esto evita que el agente promocione
conocimiento o cambie la ontología y el enrutador por accidente. Un fallo de
validación deja las escrituras bloqueadas hasta que se corrija el contexto.

Para activarla: confiar el proyecto si Senpi lo solicita y ejecutar `/reload`.
`/research-context` fuerza una comprobación y muestra el estado del grafo.

## Primer flujo: Heat 1D

El primer baseline debe declarar en `metadata.research`:

```json
{
  "notebook_id": "nb.heat.baseline.v1",
  "role": "baseline",
  "problem_id": "problem.heat_1d",
  "model_id": "model.pikan.baseline",
  "hypothesis_id": "hyp.canon.baseline",
  "source_of_truth": "manifest"
}
```

La auditoría actual identifica los notebooks históricos como `unclassified`, porque todavía no tienen ese bloque. Esa clasificación es deliberada y no modifica el contenido existente. La siguiente intervención puede adaptar una sola variante Heat como baseline canónico y dejar las demás como exploración histórica.

## Validación de una afirmación

Antes de llevar un resultado al marco teórico, debe existir:

- una hipótesis con falsación;
- un manifiesto con problema, modelo, semillas y presupuesto;
- una corrida registrada;
- métricas con alcance y dirección;
- artefactos reproducibles;
- comparación contra baseline cuando corresponda;
- decisión humana explícita y limitaciones visibles.
