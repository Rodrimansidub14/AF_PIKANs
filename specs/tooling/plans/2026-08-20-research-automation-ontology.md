# Research Automation Ontology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar una ontología declarativa y un grafo determinista de `PIKAN-Vault` que gobiernen la creación, ejecución y auditoría de notebooks mediante agentes.

**Architecture:** Tres contratos TOML en `context/` declaran conceptos científicos, contrato de notebooks y política de agentes. `src/graph_engineering/` contiene el código puro para validarlos, leer el vault, resolver wikilinks, auditar notebooks y escribir artefactos derivados en `var/context/`. Las notas del vault y los notebooks existentes se leen sin reescritura automática.

**Tech Stack:** Python 3.12+ estándar (`tomllib`, `json`, `argparse`, `unittest`), TOML, Markdown frontmatter simple, Jupyter `.ipynb` JSON y PowerShell para verificación local.

**Spec:** `docs/superpowers/specs/2026-08-20-research-automation-ontology-design.md`

## Global Constraints

- `obs-data` no se lee como entrada de ejecución, no se modifica y no aparece como dependencia del proyecto.
- `PIKAN-Vault/PIKAN-Vault` es un corpus de lectura; la herramienta no reescribe sus notas.
- La ontología canónica vive en `context/`; los mapas generados viven en `var/context/`.
- Las relaciones no explícitas se reportan como faltantes o ambiguas; no se inventan enlaces.
- Los agentes operan en nivel `L2_human_verified_ai_execution`; las decisiones científicas requieren aprobación humana.
- Las pruebas nuevas usan la biblioteca estándar y deben demostrar un fallo antes de implementar cada comportamiento.

---

### Task 1: Crear los contratos declarativos y su documentación

**Files:**
- Create: `context/research-ontology.toml`
- Create: `context/notebook-contract.toml`
- Create: `context/agent-policy.toml`
- Create: `context/README.md`
- Create: `notebooks/README.md`
- Modify: `.gitignore`

**Interfaces:**
- `research-ontology.toml` expone `meta`, `paths`, `invariants.global`, `node_type`, `relation_type` y `vocab`.
- `notebook-contract.toml` expone `metadata`, `role` y `output`.
- `agent-policy.toml` expone `meta`, `role`, `gate` y `prohibited_path`.
- Los tres archivos deben usar identificadores estables en inglés y documentación en español.

- [ ] **Step 1: Escribir el contenido TOML mínimo aprobado en la especificación**

Incluir tipos `paper`, `concept`, `domain`, `frontier`, `problem`, `model`, `focalization_mechanism`, `hypothesis`, `experiment`, `run`, `metric`, `artifact`, `evidence`, `decision` y `notebook`; vocabularios para capas, roles, estados, familias de modelo y etapas de problema; e invariantes sobre evidencia, baseline, semillas y aprobación humana.

- [ ] **Step 2: Ejecutar una carga TOML deliberadamente antes del validador**

Run: `py -3.12 -c "import tomllib; [tomllib.load(open(p, 'rb')) for p in ['context/research-ontology.toml','context/notebook-contract.toml','context/agent-policy.toml']]; print('TOML: PASS')"`

Expected: `TOML: PASS`.

- [ ] **Step 3: Documentar los límites y comandos**

`context/README.md` explicará la fuente de verdad, el flujo de agentes, la relación con `PIKAN-Vault` y la prohibición de tocar `obs-data`. `notebooks/README.md` explicará roles, metadatos y salidas.

- [ ] **Step 4: Ignorar salidas derivadas**

Agregar `/var/context/` y `src/graphify-out/` a `.gitignore`; no ignorar los contratos ni los notebooks.

- [ ] **Step 5: Verificar estructura**

Run: `py -3.12 -c "from pathlib import Path; assert all(Path(p).exists() for p in ['context/research-ontology.toml','context/notebook-contract.toml','context/agent-policy.toml','context/README.md','notebooks/README.md']); print('CONTRACT_FILES: PASS')"`

Expected: `CONTRACT_FILES: PASS`.

### Task 2: Crear pruebas fallidas del parser y validador

**Files:**
- Create: `tests/test_graph_engineering.py`

**Interfaces:**
- Tests importarán `parse_frontmatter`, `extract_wikilinks`, `resolve_links`, `validate_contracts` y `audit_notebooks` from `src.graph_engineering`.

- [ ] **Step 1: Escribir pruebas de comportamiento**

Cubrir:

```python
def test_resolves_wikilink_by_unique_vault_stem(): ...
def test_reports_missing_and_ambiguous_wikilinks(): ...
def test_parses_tags_and_frontmatter(): ...
def test_rejects_invalid_notebook_role(): ...
def test_audits_missing_research_metadata_without_editing_notebook(): ...
```

Use `tempfile.TemporaryDirectory` para crear un vault pequeño y un `.ipynb` mínimo. Las pruebas deben comparar estructuras de salida, no detalles internos de implementación.

- [ ] **Step 2: Ejecutar pruebas y comprobar que fallan por código ausente**

Run: `py -3.12 -m unittest -v tests.test_graph_engineering`

Expected: FAIL with import errors for `src.graph_engineering` or its missing functions, not with a fixture or syntax error.

### Task 3: Implementar el núcleo de lectura y resolución del vault

**Files:**
- Create: `src/graph_engineering/__init__.py`
- Create: `src/graph_engineering/vault.py`

**Interfaces:**
- `parse_frontmatter(text: str) -> dict[str, object]`
- `extract_wikilinks(text: str) -> list[str]`
- `collect_vault(vault_root: Path) -> list[dict[str, object]]`
- `resolve_links(nodes: list[dict[str, object]]) -> dict[str, object]`

- [ ] **Step 1: Implementar parser mínimo de frontmatter**

Leer solo el bloque inicial delimitado por `---`, soportar pares `key: value`, listas separadas por comas y tags `#tag`, y devolver `{}` cuando una nota no tenga frontmatter.

- [ ] **Step 2: Ejecutar pruebas del parser**

Run: `py -3.12 -m unittest -v tests.test_graph_engineering.GraphEngineeringTests.test_parses_tags_and_frontmatter`

Expected: PASS.

- [ ] **Step 3: Implementar extracción de wikilinks**

Extraer `[[target]]`, quitar alias después de `|` y encabezados después de `#`, preservar el orden y eliminar duplicados por nota.

- [ ] **Step 4: Implementar colección de nodos**

Excluir `_templates/`, asignar IDs `vault:<ruta-relativa-sin-extensión>`, usar `title` de frontmatter o primer encabezado, inferir `type` como `unclassified` cuando falte y conservar ruta, campos y tags.

- [ ] **Step 5: Implementar resolución honesta**

Resolver por ruta relativa, stem o título normalizado. Crear edges `links_to` solo con un destino único; registrar `missing_links` y `ambiguous_links` como listas independientes.

- [ ] **Step 6: Ejecutar toda la prueba de vault**

Run: `py -3.12 -m unittest -v tests.test_graph_engineering`

Expected: PASS for all vault parser and resolver tests.

### Task 4: Implementar contratos de notebooks y política de agentes

**Files:**
- Create: `src/graph_engineering/contracts.py`
- Modify: `src/graph_engineering/__init__.py`

**Interfaces:**
- `load_contracts(root: Path) -> dict[str, dict]`
- `validate_contracts(root: Path) -> dict[str, list[str]]`
- `audit_notebooks(root: Path, contract: dict) -> dict[str, object]`

- [ ] **Step 1: Implementar carga TOML y validación estructural**

Validar IDs únicos, presencia de `meta.schema_version`, tipos de nodo requeridos, vocabularios no vacíos, roles de notebook y roles de agente con `approval_required`. Devolver `{errors, warnings}` sin escribir archivos.

- [ ] **Step 2: Ejecutar prueba de rol inválido**

Run: `py -3.12 -m unittest -v tests.test_graph_engineering.GraphEngineeringTests.test_rejects_invalid_notebook_role`

Expected: PASS after the validator reports the invalid role.

- [ ] **Step 3: Implementar auditoría de notebooks**

Recorrer `notebooks/**/*.ipynb`, leer JSON sin ejecutar celdas y producir por notebook: ID de ruta, título, metadatos `research`, rol, campos faltantes y estado `ready|unclassified|invalid`. No escribir sobre el `.ipynb`.

- [ ] **Step 4: Ejecutar prueba de no modificación**

Run: `py -3.12 -m unittest -v tests.test_graph_engineering.GraphEngineeringTests.test_audits_missing_research_metadata_without_editing_notebook`

Expected: PASS and original notebook bytes unchanged.

- [ ] **Step 5: Ejecutar validación real del repositorio**

Run: `py -3.12 -c "from pathlib import Path; from src.graph_engineering.contracts import validate_contracts; result=validate_contracts(Path('.')); print(result); raise SystemExit(1 if result['errors'] else 0)"`

Expected: no contract errors; warnings may identify historical notebooks without metadata.

### Task 5: Construir el grafo de investigación y CLI

**Files:**
- Create: `src/graph_engineering/build.py`
- Create: `src/graph_engineering/__main__.py`
- Modify: `src/graph_engineering/__init__.py`
- Modify: `.gitignore`

**Interfaces:**
- `build_graph(root: Path) -> dict[str, object]`
- `render_report(graph: dict[str, object], contracts: dict[str, object]) -> str`
- CLI commands: `validate`, `audit-notebooks`, `build`.

- [ ] **Step 1: Implementar construcción del grafo**

Combinar nodos y edges del vault con nodos de notebooks y el manifiesto `src/harness_loop/experiment_manifest.yaml`. Las conexiones declaradas en metadatos de notebook deben usar IDs de vault; enlaces no resueltos deben conservarse en el reporte, no como edges.

- [ ] **Step 2: Implementar reporte legible**

Incluir conteos por tipo, edges resueltos, enlaces faltantes, enlaces ambiguos, nodos huérfanos, notebooks por estado, roles de agentes y gates humanos.

- [ ] **Step 3: Implementar CLI sin dependencia externa**

Usar `python -m src.graph_engineering validate|audit-notebooks|build`; `build --out var/context` escribirá `research-graph.json`, `notebook-audit.json`, `validation.json` y `GRAPH_REPORT.md`.

- [ ] **Step 4: Ejecutar build sobre PIKAN-Vault**

Run: `py -3.12 -m src.graph_engineering build --root . --out var/context`

Expected: outputs created under `var/context/`; no file under `PIKAN-Vault/PIKAN-Vault` changes.

- [ ] **Step 5: Auditar el grafo generado**

Run: `py -3.12 -c "import json; from pathlib import Path; g=json.loads(Path('var/context/research-graph.json').read_text(encoding='utf-8')); assert g['nodes']; assert not any('obs-data' in str(x) for x in g.values()); print(len(g['nodes']), len(g['edges']))"`

Expected: non-empty graph and no `obs-data` reference.

### Task 6: Verificar graph engineering y documentar el primer flujo Heat 1D

**Files:**
- Modify: `docs/ontology.md`
- Modify: `notebooks/README.md`
- Create: `var/context/` generated outputs

**Interfaces:**
- Documentation must show the lifecycle `question → evidence → hypothesis → manifest → notebook → run → decision`.
- Heat 1D must be identified as the first baseline candidate without changing existing notebook content.

- [ ] **Step 1: Ejecutar la auditoría Graphify del código de soporte**

Run code-only graph audit on `src/graph_engineering` and `src/harness_loop`, preserving its generated output under `var/`.

- [ ] **Step 2: Confirmar PIKAN-Vault coverage**

Run: `py -3.12 -m src.graph_engineering build --root . --out var/context`

Expected: report includes papers, concepts, domains, frontiers and current notebook inventory.

- [ ] **Step 3: Ejecutar pruebas y checks finales**

Run:

```powershell
py -3.12 -m unittest discover -s tests -p "test_*.py" -v
py -3.12 -m compileall -q src
py -3.12 -m src.graph_engineering validate --root .
py -3.12 -m src.graph_engineering audit-notebooks --root .
```

Expected: tests and compilation pass; validation has no errors; audit reports existing unclassified notebooks without modifying them.

- [ ] **Step 4: Revisar límites y estado Git**

Run: `git status --short --branch` and `git diff -- obs-data`

Expected: only this repository contains changes; there is no path or diff for `obs-data`.
