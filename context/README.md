# Contexto y ontología de investigación

`context/` contiene los contratos declarativos que gobiernan el ciclo de research automation del proyecto.

| Archivo | Responsabilidad |
|---|---|
| `research-ontology.toml` | Entidades, relaciones, vocabularios e invariantes científicas. |
| `notebook-contract.toml` | Roles, metadatos y salidas mínimas de notebooks. |
| `agent-policy.toml` | Permisos de agentes y gates de aprobación humana. |

El catálogo operativo de rutas vive en `workflow/index.toml`. Es una
proyección de intención: selecciona el workflow y rol más probable, pero no
duplica permisos ni concede gates.

La fuente de conocimiento enlazado es `PIKAN-Vault/PIKAN-Vault`. La herramienta de graph engineering lee sus notas, resuelve wikilinks y escribe únicamente derivados en `var/context/`.

Los manifiestos de `src/harness_loop/` contienen valores concretos de una corrida. No reemplazan la ontología.

## Flujo

```text
pregunta → literatura → hipótesis → manifiesto → notebook → run
  → métricas/artefactos → gates → decisión humana → claim/documento
```

El nivel de autonomía inicial es `L2_human_verified_ai_execution`: un agente puede leer, proponer y ejecutar tareas acotadas. La aprobación científica, la promoción de conocimiento canónico y cualquier cambio fuera de la superficie declarada requieren intervención humana.

Este contexto pertenece únicamente a `ProyectoGrad`. No importa archivos, datos, contratos ni estado de otros repositorios.

## Comandos

```powershell
py -3.12 -m src.graph_engineering validate --root .
py -3.12 -m src.graph_engineering audit-notebooks --root .
py -3.12 -m src.graph_engineering build --root . --out var/context
```

`build` también genera `var/context/senpi-context.json`, un paquete derivado
que proyecta roles, gates, rutas y salud del grafo para el adaptador de Senpi.

## Integración con Senpi

La extensión local `.senpi/extensions/pikan-research-context.ts` se carga solo
para este repositorio. Antes de cada prompt valida los contratos, comprueba el
manifiesto, actualiza el grafo cuando cambian las fuentes, enruta el prompt
contra `workflow/index.toml` e inyecta el contexto seleccionado al system
prompt.

El usuario puede escribir directamente, por ejemplo:

```text
adapta el baseline Heat en el notebook
```

El router selecciona `notebook_engineer` sin que el prompt tenga que declarar
el rol. `research-role=...` sigue disponible como override explícito. La
extensión bloquea las herramientas de archivo ante escrituras en `context/`,
`workflow/`, `PIKAN-Vault/PIKAN-Vault`, `.senpi/`, `.git/` y `var/context/`,
además de cualquier ruta fuera de este repositorio.

En Senpi, confía el proyecto si se solicita y ejecuta `/reload`. El comando
`/research-context` muestra el estado del paquete derivado. Esta integración
no modifica la instalación global de Senpi ni el repositorio `obs-data`.

Los resultados derivados se pueden borrar y regenerar. Las notas del vault no se editan durante estos comandos.
