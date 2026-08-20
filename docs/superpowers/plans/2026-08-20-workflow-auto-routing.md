# Automatic Workflow Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Senpi select the appropriate research workflow and role from ordinary natural-language prompts while preserving explicit overrides, human gates, and repository boundaries.

**Architecture:** Add a small TOML workflow registry under `workflow/`. Python graph engineering validates the registry and projects a safe, relative routing pack into `var/context/senpi-context.json`. The tested Senpi JavaScript core scores the projected workflows, and the local extension injects only the selected workflow while continuing to enforce the existing role-aware filesystem policy.

**Tech Stack:** Python 3.12 standard library (`tomllib`, `unittest`), Node.js ESM, local Senpi extension API, TOML, JSON, Markdown.

**Spec:** `docs/superpowers/specs/2026-08-20-workflow-auto-routing-design.md`

## Global Constraints

- The change lives only in `C:/Users/rodri/Documents/ProyectoGrad`.
- `obs-data` remains outside scope and is not read or modified.
- `PIKAN-Vault/PIKAN-Vault` remains read-only for agents; derived outputs stay under `var/context/`.
- The global Senpi installation is not modified.
- `context/` and `workflow/` are canonical read-only control paths for agents.
- Preserve `L2_human_verified_ai_execution`, human authority, and all existing scientific gates.
- Production code follows red-green-refactor: each new behavior gets a failing test before implementation.

---

## File Map

- Create `workflow/index.toml`: machine-readable workflow IDs, roles, triggers, exclusions, priorities, stages, and gate IDs.
- Create `workflow/README.md`: concise human explanation of automatic routing and the optional explicit override.
- Create `src/graph_engineering/workflows.py`: load, validate, and safely project the workflow registry.
- Modify `src/graph_engineering/contracts.py`: include workflow-registry validation in project validation.
- Modify `src/graph_engineering/context.py`: include the relative workflow projection in the Senpi context pack.
- Modify `tests/test_graph_engineering.py` and create `tests/test_workflows.py`: cover registry validation and context projection.
- Modify `.senpi/extensions/pikan-research-context-core.mjs`: add deterministic workflow selection and workflow rendering; make `workflow/` readable and protected.
- Modify `tests/senpi_context_core.test.mjs`: cover inferred routes, fallback, explicit override, ambiguity, and rendered phases.
- Modify `.senpi/extensions/pikan-research-context.ts`: watch `workflow/`, select the active workflow automatically, and pass it to the system context.
- Modify `tests/senpi_extension_smoke.test.mjs`: prove a natural-language notebook prompt routes without `research-role=...`.
- Create `AGENTS.md`: minimal project entrypoint pointing agents to `context/` and `workflow/`.
- Modify `README.md`, `context/README.md`, `docs/ontology.md`, and `.senpi/extensions/README.md`: document automatic use and the remaining override syntax.

---

### Task 1: Registry validation and context projection

**Files:**
- Create: `workflow/index.toml`
- Create: `workflow/README.md`
- Create: `src/graph_engineering/workflows.py`
- Modify: `src/graph_engineering/contracts.py`
- Modify: `src/graph_engineering/context.py`
- Test: `tests/test_workflows.py`
- Test: `tests/test_graph_engineering.py`

**Interfaces:**
- `load_workflow_registry(root: Path) -> dict[str, Any]`
- `validate_workflow_registry(root: Path, agent_contract: dict[str, Any]) -> dict[str, list[str]]`
- `project_workflow_registry(root: Path, agent_contract: dict[str, Any]) -> dict[str, Any]`
- `build_context_pack(...)['workflow']` exposes only `default`, `ambiguity_policy`, and safe workflow items.

- [ ] **Step 1: Write the failing Python tests.** Add a focused registry test with the expected six workflows and a projection test with no absolute paths:

```python
def test_projects_workflows_and_references_existing_roles(self) -> None:
    contracts = load_contracts(PROJECT_ROOT)
    projection = project_workflow_registry(PROJECT_ROOT, contracts["agent"])

    self.assertEqual(projection["default"], "literature")
    self.assertEqual(projection["ambiguity_policy"], "conservative")
    self.assertEqual(
        {item["id"] for item in projection["items"]},
        {"literature", "hypothesis", "notebook", "run", "evaluation", "synthesis"},
    )
    self.assertEqual(
        {item["role"] for item in projection["items"]},
        {role["id"] for role in contracts["agent"]["role"]},
    )
    self.assertNotIn(str(PROJECT_ROOT), json.dumps(projection))

def test_rejects_workflow_with_unknown_role(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry = root / "workflow"
        registry.mkdir()
        (registry / "index.toml").write_text(
            '[meta]\nschema_version = 1\ndefault = "bad"\nambiguity_policy = "conservative"\n'
            '\n[[workflow]]\nid = "bad"\nrole = "missing_role"\npriority = 1\n'
            'triggers = ["bad"]\nexcludes = []\nstages = ["inspect"]\ngate_ids = []\n',
            encoding="utf-8",
        )

        result = validate_workflow_registry(root, {"role": [{"id": "literature_scout"}]})

        self.assertTrue(any("missing_role" in error for error in result["errors"]))
```

- [ ] **Step 2: Run the focused tests and verify the expected RED state.**

Run:

```powershell
py -3.12 -m unittest tests.test_workflows tests.test_graph_engineering.GraphEngineeringTests.test_context_pack_projects_workflows -v
```

Expected: import or attribute failures because the registry module, file, and context projection do not yet exist.

- [ ] **Step 3: Add the minimal registry and loader.** Create `workflow/index.toml` with these exact IDs and roles:

```toml
[meta]
schema_version = 1
default = "literature"
ambiguity_policy = "conservative"

[[workflow]]
id = "literature"
role = "literature_scout"
priority = 10
triggers = ["literatura", "fuente", "fuentes", "paper", "concepto", "marco teorico", "vault", "resume", "explica"]
excludes = ["notebook", "ejecuta", "entrena", "metricas"]
stages = ["orient", "extract_evidence", "propose_links"]
gate_ids = ["scientific_claim"]

[[workflow]]
id = "hypothesis"
role = "hypothesis_planner"
priority = 20
triggers = ["hipotesis", "falsacion", "pregunta de investigacion", "manifiesto", "presupuesto", "semillas"]
excludes = ["ejecuta", "corre", "metricas finales"]
stages = ["define_question", "draft_hypothesis", "draft_manifest"]
gate_ids = ["scope", "hypothesis"]

[[workflow]]
id = "notebook"
role = "notebook_engineer"
priority = 30
triggers = ["notebook", ".ipynb", "adapta", "crear notebook", "baseline", "candidato"]
excludes = ["solo explica", "solo resume"]
stages = ["inspect_contract", "edit_declared_surface", "audit"]
gate_ids = ["scope", "code_surface"]

[[workflow]]
id = "run"
role = "experiment_runner"
priority = 40
triggers = ["ejecuta", "corre", "entrena", "run-local", "corrida", "manifiesto"]
excludes = ["solo planifica", "solo explica"]
stages = ["validate_manifest", "run_local", "record_metrics"]
gate_ids = ["hypothesis"]

[[workflow]]
id = "evaluation"
role = "evaluator"
priority = 35
triggers = ["evalua", "evaluar", "metricas", "compara", "comparar", "resultado", "gate"]
excludes = ["solo literatura"]
stages = ["read_runs", "compute_metrics", "evaluate_gates"]
gate_ids = ["scientific_claim"]

[[workflow]]
id = "synthesis"
role = "research_synthesizer"
priority = 25
triggers = ["redacta", "escribe", "informe", "latex", "marco teorico", "claim", "sintetiza"]
excludes = ["sin evidencia", "sin resultados"]
stages = ["read_evidence", "draft_report", "link_sources"]
gate_ids = ["scientific_claim"]
```

Implement `src/graph_engineering/workflows.py` with standard-library TOML loading, duplicate-ID checks, required metadata checks, default-workflow checks, role-reference checks, unique trigger checks within each item, and a projection that drops unknown fields and absolute paths.

- [ ] **Step 4: Wire workflow validation and projection into the existing Python path.**

`validate_contracts` must append workflow errors/warnings to its existing result. `build_context_pack` must include:

```python
"workflow": project_workflow_registry(root, agent)
```

Do not add workflow data to the vault graph nodes; it belongs only in the derived context pack.

- [ ] **Step 5: Run the focused Python tests and verify GREEN.**

Run:

```powershell
py -3.12 -m unittest tests.test_workflows tests.test_graph_engineering -v
```

Expected: all focused tests pass, including the pre-existing vault, graph, notebook, and context-pack tests.

- [ ] **Step 6: Commit the registry/projection slice.**

```powershell
git add workflow src/graph_engineering/workflows.py src/graph_engineering/contracts.py src/graph_engineering/context.py tests/test_workflows.py tests/test_graph_engineering.py
git commit -m "feat: add declarative research workflow registry"
```

### Task 2: Tested automatic workflow selector

**Files:**
- Modify: `.senpi/extensions/pikan-research-context-core.mjs`
- Test: `tests/senpi_context_core.test.mjs`

**Interfaces:**
- `selectWorkflow(prompt, workflowContext, roles) -> { workflow, role, explicit, confidence, matchedTriggers, error? }`
- `renderSystemPrompt({ pack, graph, role, workflow, prompt, ready, error }) -> string`

- [ ] **Step 1: Add failing Node tests before changing the core.** Add a fixture with the six projected items and assert automatic selection:

```javascript
test("infers notebook workflow from a natural-language prompt", () => {
    const result = selectWorkflow("adapta el baseline Heat en el notebook", PACK.workflow, PACK.roles);

    assert.equal(result.workflow.id, "notebook");
    assert.equal(result.role.id, "notebook_engineer");
    assert.equal(result.explicit, false);
    assert.deepEqual(result.matchedTriggers.sort(), ["adapta", "baseline", "notebook"].sort());
});

test("falls back conservatively when no workflow trigger matches", () => {
    const result = selectWorkflow("revisa este asunto general", PACK.workflow, PACK.roles);

    assert.equal(result.workflow.id, "literature");
    assert.equal(result.role.id, "literature_scout");
    assert.equal(result.confidence, "fallback");
});

test("explicit role remains an override over inferred workflow", () => {
    const result = selectWorkflow("research-role=evaluator adapta el notebook", PACK.workflow, PACK.roles);

    assert.equal(result.workflow.id, "evaluation");
    assert.equal(result.role.id, "evaluator");
    assert.equal(result.explicit, true);
});

test("rendered context includes the active workflow and phases", () => {
    const rendered = renderSystemPrompt({
        pack: PACK,
        graph: PACK,
        role: PACK.roles[1],
        workflow: PACK.workflow.items[2],
        prompt: "adapta el baseline",
        ready: true,
    });

    assert.match(rendered, /Workflow activo:.*notebook/);
    assert.match(rendered, /inspect_contract/);
});
```

- [ ] **Step 2: Run the focused Node test and verify RED.**

```powershell
node --test tests/senpi_context_core.test.mjs
```

Expected: `selectWorkflow is not defined` or equivalent missing-workflow assertions while the pre-existing tests continue to identify the current API.

- [ ] **Step 3: Implement deterministic selection.** Add normalization using the existing accent-insensitive `normalizeText`, match triggers as normalized phrases, discard workflows with matching exclusions, rank by match count then priority, and use `literature`/the first role as a fail-safe when there is no match. Return `confidence` values `explicit`, `high`, `medium`, or `fallback`.

The explicit path must call the existing `parseResearchDirective`; unknown roles must retain its error and never produce a write-enabled route. The inferred path must resolve the selected workflow's role against `roles`; a missing role is an error and falls back to the default role.

- [ ] **Step 4: Extend prompt rendering without duplicating the entire registry.** Add these lines before the relevant-node section:

```text
Workflow activo: `notebook` — confidence `high`
Fases: `inspect_contract`, `edit_declared_surface`, `audit`
Gates del workflow: `scope`, `code_surface`
Triggers detectados: `adapta`, `baseline`, `notebook`
```

If the context is blocked, keep the existing `Context unavailable` notice and do not imply that the route grants permission.

- [ ] **Step 5: Run all Node core tests and verify GREEN.**

```powershell
node --test tests/senpi_context_core.test.mjs
```

- [ ] **Step 6: Commit the selector slice.**

```powershell
git add .senpi/extensions/pikan-research-context-core.mjs tests/senpi_context_core.test.mjs
git commit -m "feat: infer research workflow from prompts"
```

### Task 3: Senpi extension integration and protected workflow path

**Files:**
- Modify: `.senpi/extensions/pikan-research-context.ts`
- Modify: `tests/senpi_extension_smoke.test.mjs`

**Interfaces:**
- `before_agent_start` invokes `selectWorkflow` for every prompt without requiring a directive.
- The filesystem policy allows read/enumeration of `workflow/` and denies writes to it for every role.

- [ ] **Step 1: Extend the smoke test with a failing natural-language assertion.** Replace the directive-bearing prompt in the integration test with:

```javascript
prompt: "adapta el baseline Heat en el notebook",
```

and assert:

```javascript
assert.match(result.systemPrompt, /Workflow activo:.*notebook/);
assert.match(result.systemPrompt, /notebook_engineer/);
```

Add policy assertions that a role can read `workflow/index.toml` and cannot write it.

- [ ] **Step 2: Run the smoke test and verify RED.**

```powershell
node --test tests/senpi_extension_smoke.test.mjs
```

Expected: the existing extension context lacks a workflow route and the new assertions fail.

- [ ] **Step 3: Connect the extension to the projected workflow pack.**

Modify `before_agent_start` to compute:

```javascript
const routing = selectWorkflow(event.prompt, pack.workflow, pack.roles ?? []);
state.roleId = routing.role.id;
state.directiveError = routing.error;
const role = routing.role;
const context = renderSystemPrompt({
    pack,
    graph,
    role,
    workflow: routing.workflow,
    prompt: event.prompt,
    ready: state.ready && !routing.error,
    error: routing.error ?? state.error,
});
```

Keep `emptyPack()` compatible by including a conservative literature workflow when no generated pack is available.

- [ ] **Step 4: Add `workflow` to the source fingerprint and protect it.**

Add `workflow` to `WATCH_PATHS`. In the pure core, include the resolved workflow root in protected paths; allow reads through the same unconditional generated/control read branch as `var/context`, but reject writes before checking role write paths. Add `workflow/` to the Bash mutation guard.

- [ ] **Step 5: Run both Node suites and verify GREEN.**

```powershell
node --test tests/senpi_context_core.test.mjs tests/senpi_extension_smoke.test.mjs
```

- [ ] **Step 6: Commit the Senpi integration slice.**

```powershell
git add .senpi/extensions/pikan-research-context.ts .senpi/extensions/pikan-research-context-core.mjs tests/senpi_context_core.test.mjs tests/senpi_extension_smoke.test.mjs
git commit -m "feat: auto-route Senpi research context"
```

### Task 4: Repository entrypoint and documentation

**Files:**
- Create: `AGENTS.md`
- Modify: `README.md`
- Modify: `context/README.md`
- Modify: `docs/ontology.md`
- Modify: `.senpi/extensions/README.md`

- [ ] **Step 1: Write the root entrypoint and update the user-facing flow.** `AGENTS.md` must be short and operational:

```markdown
# ProyectoGrad agent entrypoint

Use `context/` as the research-control source and discover the active route
from `workflow/index.toml`. Prompts are routed automatically from natural
language; do not require the user to paste workflow instructions or declare a
role for ordinary tasks. Keep the explicit `research-role=...` syntax only as
an override. Never modify `context/`, `workflow/`, `PIKAN-Vault/PIKAN-Vault`,
`.senpi/`, `var/context/`, or any path outside this repository.
```

- [ ] **Step 2: Document examples without manual flags.** Include these examples in the appropriate docs:

```text
resume las fuentes relevantes para el marco teórico
adapta el baseline Heat en el notebook y audítalo
ejecuta el manifiesto en dry-run
compara las métricas de las corridas
redacta un borrador del apartado usando evidencia aceptada
```

Explain that `research-role=...` remains available only when the automatic route needs to be overridden.

- [ ] **Step 3: Check documentation for contradictions.** Remove statements saying that a write-oriented prompt must always declare `research-role=...`; replace them with the automatic-routing behavior and the explicit-override exception.

- [ ] **Step 4: Run Markdown/config presence checks.**

```powershell
Test-Path AGENTS.md
Test-Path workflow/index.toml
Test-Path workflow/README.md
rg -n "automatic|automát|research-role|workflow/index" AGENTS.md README.md context/README.md docs/ontology.md .senpi/extensions/README.md workflow/README.md
```

- [ ] **Step 5: Commit the documentation slice.**

```powershell
git add AGENTS.md README.md context/README.md docs/ontology.md .senpi/extensions/README.md workflow/README.md
git commit -m "docs: document automatic workflow routing"
```

### Task 5: Full verification and security audit

**Files:**
- Verify: all files above and generated `var/context/` outputs.

- [ ] **Step 1: Run the complete Python test suite.**

```powershell
py -3.12 -m unittest discover -s tests -p "test_*.py" -v
```

Expected: exit code 0 and no failing tests.

- [ ] **Step 2: Run the complete Node test suite.**

```powershell
node --test tests/*.test.mjs
```

Expected: exit code 0 and no failing tests.

- [ ] **Step 3: Validate contracts, manifest, and build the context pack.**

```powershell
py -3.12 -m src.graph_engineering validate --root .
py -3.12 src/research_pipeline/pipeline.py validate --manifest src/harness_loop/experiment_manifest.yaml
py -3.12 -m src.graph_engineering build --root . --out var/context
```

Inspect `var/context/senpi-context.json` and require `workflow.default == "literature"`, six workflow items, valid role references, and no absolute project path.

- [ ] **Step 4: Exercise the actual router with representative prompts.** Load the pure core with Node and assert these mappings:

```text
"resume las fuentes del marco teorico"       → literature / literature_scout
"plantea una hipotesis falsable"             → hypothesis / hypothesis_planner
"adapta el baseline Heat en el notebook"    → notebook / notebook_engineer
"ejecuta el manifiesto en dry-run"           → run / experiment_runner
"compara las metricas de las corridas"       → evaluation / evaluator
"redacta el apartado con evidencia aceptada" → synthesis / research_synthesizer
"revisa este asunto general"                 → literature / fallback
```

- [ ] **Step 5: Audit protected boundaries and diff.**

```powershell
git status --short
git diff --name-only HEAD
git diff --name-only -- C:/Users/rodri/Documents/IMP/obs-data
git diff --name-only -- C:/Users/rodri/.senpi
git diff --name-only -- PIKAN-Vault/PIKAN-Vault
```

The final diff may contain only this repo's registry, code, tests, docs, and the prior design/plan artifacts. `var/context/` is derived and ignored. Any result showing an external or protected canonical path requires stopping and correcting the change before completion.

- [ ] **Step 6: Mark the plan complete only after fresh verification evidence.** Record the exact test counts and command exit codes in the handoff; do not claim completion from static inspection alone.

## Plan self-review

- Registry creation, validation, projection, and source fingerprinting cover the declarative-discovery requirement.
- Selector tests cover all six workflows, fallback, explicit override, and prompt rendering.
- Extension tests cover automatic use and workflow read/write policy.
- Documentation covers natural-language use and preserves the override escape hatch.
- Full verification covers Python, Node, contracts, manifest, context build, route examples, and all requested boundaries.
- No placeholder steps or undefined interfaces remain; every implementation step has a concrete file, command, or data shape.

