# Contexto y ontología de investigación

`context/` contiene los contratos declarativos que gobiernan el ciclo de research automation del proyecto.

| Archivo | Responsabilidad |
|---|---|
| `research-ontology.toml` | Entidades, relaciones, vocabularios e invariantes científicas. |
| `notebook-contract.toml` | Roles, metadatos y salidas mínimas de notebooks. |
| `agent-policy.toml` | Permisos de agentes y gates de aprobación humana. |

Desde 2026-09-22 la política declara la arquitectura proposer/verifier:
`evals/` (el verificador) es de solo lectura para todo rol,
`registry/claims.jsonl` solo lo escribe el investigador, y `src/mechanisms/`
es la superficie de código que el proposer puede escribir sin gate. El proceso
completo está en `specs/proceso-de-ingenieria.md`.

El catálogo operativo de rutas vive en `workflow/index.toml`. Es una
proyección de intención: selecciona el workflow y rol más probable, pero no
duplica permisos ni concede gates.

La fuente de conocimiento enlazado es `PIKAN-Vault/PIKAN-Vault`. La herramienta de graph engineering lee sus notas, resuelve wikilinks y escribe únicamente derivados en `var/context/`; la promoción de notas desde `var/agent_runs` al vault es una operación separada y requiere aprobación humana explícita.

Las lecturas de archivos fuera de `AF_PIKANs` pueden autorizarse cuando la
tarea las necesita, siempre en modo solo lectura y con aviso previo de la ruta
al investigador. Las escrituras fuera del repositorio permanecen bloqueadas.

La excepción remota aprobada el 2026-09-20 permite conectar Kaggle, subir notebooks
privados y ejecutar campañas previamente aprobadas. No autoriza escrituras de
archivos fuera del repositorio ni otros servicios. El adaptador remoto verifica
el servicio permitido y la aprobación del paquete; los gates científicos siguen
siendo humanos. La autenticación queda al final de la preparación local.

Los manifiestos de `src/harness_loop/` contienen valores concretos de una corrida. No reemplazan la ontología.

## Flujo

```text
pregunta → literatura → hipótesis → manifiesto → notebook → run
  → métricas/artefactos → gates → decisión humana → claim/documento
```

El nivel de autonomía inicial es `L2_human_verified_ai_execution`: un agente puede leer, proponer y ejecutar tareas acotadas. La aprobación científica, la promoción de conocimiento canónico y cualquier cambio fuera de la superficie declarada requieren intervención humana.

Este contexto pertenece únicamente a `AF_PIKANs`. No importa archivos, datos, contratos ni estado de otros repositorios.

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
`workflow/`, `.senpi/`, `.git/` y `var/context/`, además de cualquier ruta
fuera de este repositorio. La escritura en `PIKAN-Vault/PIKAN-Vault` queda
declarada únicamente para `literature_scout` y requiere aprobación humana
explícita. Las lecturas externas se permiten en modo solo lectura y la
extensión informa la ruta antes de ejecutarlas.

En Senpi, confía el proyecto si se solicita y ejecuta `/reload`. El comando
`/research-context` muestra el estado del paquete derivado. Esta integración
no modifica la instalación global de Senpi ni el repositorio `obs-data`.

Los resultados derivados se pueden borrar y regenerar. Los comandos de
validación y construcción no editan las notas del vault; la promoción es una
operación separada y aprobada.
