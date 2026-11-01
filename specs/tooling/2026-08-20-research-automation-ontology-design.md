# Ontología de automatización de investigación para PIKAN

## Estado

Propuesta aprobada para implementación el 2026-08-20.

## Objetivo

Establecer un contrato declarativo para diseñar, ejecutar y auditar notebooks de investigación PIKAN mediante agentes, manteniendo la decisión científica bajo aprobación humana y usando `PIKAN-Vault` como corpus de conocimiento enlazado.

## Límites

- Este diseño aplica únicamente a `ProyectoGrad`.
- `obs-data` no es dependencia, consumidor, fuente de archivos ni destino de cambios.
- `PIKAN-Vault` se lee como corpus de notas; sus archivos no se reescriben automáticamente.
- La ontología no convierte un resultado experimental en una afirmación científica por sí sola.
- Los mapas, auditorías y grafos derivados se escriben en `var/context/` y no son fuente de verdad.

## Arquitectura

La fuente declarativa se divide en tres contratos pequeños:

- `context/research-ontology.toml`: entidades, relaciones, vocabularios e invariantes científicas.
- `context/notebook-contract.toml`: metadatos, roles y salidas mínimas de cada notebook.
- `context/agent-policy.toml`: roles de agentes, permisos, artefactos permitidos y gates humanos.

El código de `src/graph_engineering/` valida esos contratos y construye un grafo determinista a partir de las notas Markdown de `PIKAN-Vault`, los notebooks Jupyter y los manifiestos de experimento. `var/context/` recibe el JSON, el reporte Markdown y el inventario de notebooks.

## Modelo de investigación

Las entidades principales son:

- `paper`, `concept`, `domain` y `frontier`: conocimiento curado en el vault.
- `problem`: PDE, dimensiones, dominio, condiciones y solución de referencia.
- `model`: PINN, PIKAN y variantes.
- `focalization_mechanism`: señal, política y parámetros de focalización.
- `hypothesis`: afirmación, efecto esperado y condición de falsación.
- `experiment`: combinación declarada de problema, modelo, hipótesis, semillas y presupuesto.
- `run`: ejecución concreta de un experimento.
- `metric`: medida con dirección, unidad y alcance global o local.
- `artifact`: configuración, curva, mapa, métrica o nota generada.
- `evidence`: fuente que respalda una observación o claim.
- `decision`: aceptación, rechazo o aplazamiento con justificación.
- `notebook`: interfaz reproducible de exploración, baseline, candidato, análisis o reporte.

Las relaciones previstas incluyen `links_to`, `supports`, `extends`, `evaluates`, `instantiates`, `uses`, `tests`, `produces`, `derived_from` y `compares_with`. El grafo determinista solo infiere `links_to` desde wikilinks y conexiones declaradas en metadatos; las demás relaciones requieren declaración explícita.

## Ciclo de research automation

```text
pregunta → literatura → hipótesis → manifiesto → notebook → run
  → métricas/artefactos → gates → decisión humana → claim/documento
```

El manifiesto vigente en `src/harness_loop/experiment_manifest.yaml` es una instancia de este modelo. La ontología estandariza sus nombres y significados, pero no duplica sus valores de ejecución.

## Roles de agentes y autoridad

El nivel inicial es `L2_human_verified_ai_execution`:

- un agente de literatura puede extraer evidencia y proponer enlaces;
- un agente de hipótesis puede redactar propuestas falsables;
- un agente de notebooks puede modificar únicamente la superficie declarada;
- el harness puede ejecutar y registrar corridas;
- un agente evaluador puede calcular gates y redactar decisiones;
- ningún agente puede aprobar un claim científico, modificar la ontología canónica o escribir en `obs-data`.

Los gates humanos son: aprobar el alcance, aprobar la hipótesis/manifiesto, aprobar cambios al código central y aceptar el resultado para escritura científica.

## Graph engineering con PIKAN-Vault

El grafo se construye de forma reproducible:

1. leer frontmatter y títulos de las notas Markdown;
2. asignar un identificador estable `vault:<ruta-relativa-sin-extensión>`;
3. extraer wikilinks `[[...]]`;
4. resolver enlaces por ruta, nombre de archivo o título normalizado;
5. registrar enlaces faltantes y ambiguos sin inventar nodos;
6. agregar notebooks como nodos auditables;
7. escribir `research-graph.json` y `GRAPH_REPORT.md` en `var/context/`.

Las notas bajo `_templates/` se excluyen del corpus. Un enlace no resuelto es una advertencia visible, no una relación inventada.

## Contrato mínimo de notebook

Los notebooks de `baseline`, `candidate`, `analysis` y `reporting` deben declarar un bloque de metadatos `research` con `notebook_id`, `role`, `problem_id`, `model_id`, `hypothesis_id` y `source_of_truth`. Los notebooks de `exploration` requieren como mínimo `notebook_id` y `role`.

Una corrida experimental debe producir, mediante el harness, métricas, configuración, resumen de ejecución, evidencia visual pertinente y una nota de decisión. El inventario inicial puede reportar notebooks históricos sin esos metadatos; la auditoría no los modifica.

## Invariantes

- Una afirmación experimental requiere una corrida y artefactos trazables.
- Una hipótesis debe declarar una condición de falsación.
- Baseline y candidato deben compararse con el mismo problema, presupuesto y protocolo de semillas.
- El error global no sustituye las métricas de residual o error en regiones críticas cuando la hipótesis las requiere.
- `0`, vacío y ausencia de medición no son equivalentes.
- Una nota de literatura puede respaldar una motivación, pero no sustituye evidencia experimental.
- La ausencia de un wikilink no prueba ausencia conceptual.
- El agente puede proponer; la aprobación humana decide.

## Criterios de aceptación

- Los tres contratos TOML validan sin errores.
- El grafo de `PIKAN-Vault` se genera sin modificar el vault.
- El reporte expone tipos de notas, enlaces resueltos, enlaces faltantes, huérfanos y notebooks sin metadatos.
- El validador detecta tipos, roles y valores de vocabulario inválidos.
- El primer corte identifica Heat 1D como candidato a baseline sin imponer cambios destructivos a notebooks existentes.
- Las pruebas del grafo y del contrato pasan con la biblioteca estándar de Python.
