# Senpi Research Context Integration Design

## Goal

Every Senpi prompt launched in `ProyectoGrad` receives the project's research-automation contracts and current PIKAN-Vault graph, while the built-in filesystem tools enforce the declared research surface.

## Constraints

- `C:/Users/rodri/Documents/IMP/obs-data` is out of scope and must not be read or modified by the adapter.
- `PIKAN-Vault/PIKAN-Vault` remains read-only for agents; graph generation writes only derived files under `var/context/`.
- The global Senpi installation under `C:/Users/rodri/.senpi` is not modified.
- The adapter must use only the repository's existing Python standard-library tooling and Senpi's native extension API.
- Human approval remains required for ontology changes, core-model/PDE changes, expensive runs, claims, and promotion of canonical knowledge.

## Architecture

The integration has two layers:

1. `src.graph_engineering` emits `var/context/senpi-context.json` alongside the existing graph, audit, validation, and report artifacts. The JSON is a derived context pack built from the TOML contracts and current graph; it contains roles, gates, paths, and graph health counters, not copied vault prose.
2. `.senpi/extensions/pikan-research-context.ts` is auto-discovered by Senpi for this project. Its `before_agent_start` handler validates/refreshes the derived context, selects a role from an explicit prompt directive, and appends a compact system context. Its factory-time filesystem policy denies out-of-project access and writes to canonical or generated-control paths.

The prompt directive is optional for read-only work. Without one, the adapter uses the conservative `literature_scout` role. A write-oriented task must explicitly include, for example, `research-role=notebook_engineer`.

## Runtime flow

```text
Senpi prompt
  → parse research-role directive
  → validate contracts and manifest
  → refresh var/context when sources changed
  → load senpi-context.json + research-graph.json
  → append role, paths, gates, graph summary, and invariants
  → agent uses read/write tools
  → filesystem policy checks every built-in tool path
  → human reviews gates and scientific decisions
```

The extension caches a source fingerprint for the current session. It rebuilds only when contract files, notebooks, the vault, or the experiment manifest change, or when derived outputs are absent.

## Filesystem policy

- Any read, enumeration, or write outside the project root is denied. This prevents the agent from accessing `obs-data` through Senpi's built-in file tools.
- Writes to `context/`, `PIKAN-Vault/PIKAN-Vault`, `.git/`, `.senpi/`, and `var/context/` are denied.
- Reads and enumeration are limited to the active role's `read_paths`, plus the generated `var/context/` pack.
- Writes are limited to the active role's `write_paths` and remain inside the project root.
- A `tool_call` guard blocks obvious Bash mutations targeting `obs-data` or protected canonical paths. This is a convenience guard; Senpi's native filesystem policy is the enforcement boundary for built-in file tools, not arbitrary shell commands.

If validation or graph refresh fails, the extension injects a blocked-context notice and denies all agent writes until the issue is corrected.

## Testing and activation

- Python unit tests cover context-pack contents and role/gate projection.
- Node tests cover role parsing, path containment, write/read decisions, node selection, and prompt rendering.
- The extension is syntax-checked and loaded with a minimal fake Senpi API in a smoke test.
- The user activates the project-local extension by trusting the project if Senpi asks and running `/reload`.

