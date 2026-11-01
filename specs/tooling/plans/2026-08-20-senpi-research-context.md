# Senpi Research Context Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the project ontology and PIKAN-Vault graph to every Senpi prompt in this repository through a project-local extension with validated context and filesystem guardrails.

**Architecture:** Extend the existing Python graph-engineering CLI with a derived `senpi-context.json` pack. Add a project-local Senpi TypeScript extension that refreshes this pack, injects a compact system context through `before_agent_start`, and applies role-aware filesystem policies using a small tested JavaScript core.

**Tech Stack:** Python 3.12 standard library (`tomllib`, `unittest`), Node.js ESM, Senpi extension API, TOML/YAML/JSON contracts.

**Spec:** `specs/tooling/2026-08-20-senpi-research-context-design.md`

## Global Constraints

- Never read or modify `C:/Users/rodri/Documents/IMP/obs-data` from the adapter.
- Never write to `PIKAN-Vault/PIKAN-Vault`; graph outputs belong only under `var/context/`.
- Do not modify the global Senpi installation under `C:/Users/rodri/.senpi`.
- Keep the adapter project-local in `.senpi/extensions/`.
- Preserve `L2_human_verified_ai_execution` and all existing approval gates.
- Production code follows test-first red-green-refactor cycles.

---

### Task 1: Project context-pack projection

**Files:**
- Create: `src/graph_engineering/context.py`
- Modify: `src/graph_engineering/__main__.py`
- Test: `tests/test_graph_engineering.py`

**Interfaces:**
- `build_context_pack(root: Path, graph: dict[str, Any], contracts: dict[str, dict[str, Any]]) -> dict[str, Any]`
- The `build` CLI writes `var/context/senpi-context.json` next to the existing derived graph files.

- [ ] **Step 1: Write failing tests** for a context pack that exposes ontology identity, all agent roles, all gates, declared paths, graph counts, notebook audit counts, and no absolute filesystem paths.
- [ ] **Step 2: Run `py -3.12 -m unittest tests.test_graph_engineering -q` and verify the new assertions fail because the context pack does not exist.
- [ ] **Step 3: Implement the projection and make the `build` command serialize it with stable sorted JSON.
- [ ] **Step 4: Run the focused test again and verify it passes.

### Task 2: Tested Senpi policy core

**Files:**
- Create: `.senpi/extensions/pikan-research-context-core.mjs`
- Create: `tests/senpi_context_core.test.mjs`

**Interfaces:**
- `parseResearchDirective(prompt, roles) -> { role, explicit, error? }`
- `isContained(root, candidate) -> boolean`
- `filesystemDecision({ root, role, operation, canonicalPath }) -> { allow: true } | { allow: false, reason: string }`
- `selectRelevantNodes(graph, prompt, limit) -> node[]`
- `renderSystemPrompt({ pack, graph, role, prompt, ready, error }) -> string`

- [ ] **Step 1: Write Node tests** for default conservative role, explicit valid/invalid role directives, outside-root denial, protected-write denial, role read/write scopes, generated-context reads, relevant-node selection, and blocked-context wording.
- [ ] **Step 2: Run `node --test tests/senpi_context_core.test.mjs` and verify the tests fail because the module does not exist.
- [ ] **Step 3: Implement the pure policy/context functions without filesystem or process side effects.
- [ ] **Step 4: Run the Node tests and verify they pass.

### Task 3: Project-local Senpi extension

**Files:**
- Create: `.senpi/extensions/pikan-research-context.ts`
- Create: `tests/senpi_extension_smoke.test.mjs`

**Interfaces:**
- Default export: Senpi extension factory receiving `pi`.
- Registers `research-context` command, `before_agent_start`, `session_start`, `tool_call`, and a factory-time filesystem policy.

- [ ] **Step 1: Write a smoke test** that loads the extension through Senpi's installed TypeScript loader or a direct transpile-free import path and asserts registration of the required hooks/policy/command.
- [ ] **Step 2: Run the smoke test and verify it fails because the extension does not exist.
- [ ] **Step 3: Implement Python command execution with `py -3.12`, fallback interpreters, timeout, source fingerprint caching, and fail-closed write state.
- [ ] **Step 4: Implement `before_agent_start` context injection and role selection using the tested core module.
- [ ] **Step 5: Implement native filesystem policy and the best-effort Bash guard for protected mutations.
- [ ] **Step 6: Run the smoke test and verify it passes.

### Task 4: Documentation and full verification

**Files:**
- Modify: `context/README.md`
- Modify: `docs/ontology.md`
- Modify: `README.md`
- Create: `.senpi/extensions/README.md`

- [ ] **Step 1: Document the prompt directive, default role, protected paths, generated outputs, and `/reload` activation.
- [ ] **Step 2: Run Python unit tests, Node tests, Python compilation, contract validation, manifest validation, and graph build.
- [ ] **Step 3: Verify the actual project-local extension files are present and the global Senpi directory and `obs-data` Git status remain untouched.
