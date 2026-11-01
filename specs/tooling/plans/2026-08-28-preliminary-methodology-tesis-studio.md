# Preliminary Methodology in Tesis-Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an editable preliminary methodology in the existing Tesis-Studio Obsidian note, detailed enough to guide a reproducible PIKAN focalization study without claiming that pending experiments already occurred.

**Architecture:** Preserve the one-note `j-capitulos.tex` topology and the existing Markdown-to-staged-LaTeX contract. Replace the generic methodology prose with a two-level structure: a common experimental protocol followed by repeated domain-specific procedures, borrowing the compact procedural rhythm of the adviser thesis and the explicit method/baseline/ablation/reproducibility pattern of landmark machine-learning papers.

**Tech Stack:** Obsidian Markdown, protected LaTeX metadata comments, LaTeX math blocks, existing `src.thesis_workspace` Markdown-to-LaTeX bridge, Python `unittest` verification.

**Spec:** `docs/superpowers/specs/2026-08-28-obsidian-thesis-workspace-and-chapter-6-design.md`

## Global Constraints

- Modify only `var/agent_runs/obsidian-bootstrap/Tesis-Studio/01_Secciones/70_Metodologia_y_Plan_de_Trabajo.md` during implementation.
- Preserve all frontmatter keys and values, the `# Metodología` and `# Plan de trabajo` chapter boundaries, existing protected LaTeX blocks, and the `fig-cronograma-gantt` block.
- Keep active `.tex` files, `PIKAN-Vault/PIKAN-Vault`, `context/`, `workflow/`, `.senpi/`, and bibliography authority files unchanged.
- Use present tense for definitions, past tense only for decisions or procedures evidenced as completed, and future tense or explicit Obsidian TODO callouts for pending experimental execution.
- Do not invent hyperparameters, thresholds, run counts, hardware, software versions, FEM settings, results, or acceptance decisions.
- Reuse only citation keys already present in `99_Bibliografia.md`; retain the protected citation-marker plus Obsidian-wikilink format.
- Treat the adviser thesis and landmark AI papers as structural inspiration, not as scientific evidence for PIKAN claims.
- Keep the contribution framed as a focalization mechanism evaluated on a PIKAN substrate, not as a broad architecture benchmark.

---

### Task 1: Rewrite the preliminary methodology note

**Files:**
- Modify: `var/agent_runs/obsidian-bootstrap/Tesis-Studio/01_Secciones/70_Metodologia_y_Plan_de_Trabajo.md`
- Inspect: `var/agent_runs/obsidian-bootstrap/Tesis-Studio/99_Bibliografia.md`
- Inspect: `src/harness_loop/experiment_manifest.yaml`

**Interfaces:**
- Consumes: existing frontmatter, LaTeX label comments, Obsidian block IDs, protected citation markers, decision notes `DEC-001` through `DEC-004`, and the existing plan-of-work figure block.
- Produces: one editable methodology note that the existing stage command can convert into `j-capitulos.tex` without modifying the active source.

- [ ] **Step 1: Preserve the note contract and change its status**

  Keep all frontmatter fields unchanged except `status`, which becomes `editing`. Preserve every existing top-level chapter boundary and the protected Gantt figure block verbatim.

- [ ] **Step 2: Add an editorial-status callout**

  Add a short Obsidian callout after `# Metodología` stating that this is a preliminary protocol: completed design decisions are narrated in past tense, pending runs remain in future tense, and the final thesis will convert executed procedures to past tense after evidence exists.

- [ ] **Step 3: Map DSRM to observable project outputs**

  Describe the six DSRM activities and map each one to an artifact already used by this project: research problem, mechanism requirements, implementation, domain demonstration, metric-based evaluation, and thesis/MLflow communication. Cite only the existing Peffers key.

- [ ] **Step 4: Define the common experimental protocol**

  Add subsections for the unit of analysis, controlled comparison, reproducibility contract, reference-solution generation, sampling, training, independent evaluation, failure handling, and run registration. Use TODO callouts for unresolved numeric values and do not invent them.

- [ ] **Step 5: Define the baseline and focalization intervention**

  Specify the PIKAN control, the focalized candidate, preserved variables, allowable intervention surface, signal-transform-weight pipeline, reduction to the baseline, and planned ablations. Link the decision notes rather than silently resolving pending scientific gates.

- [ ] **Step 6: Define repeated procedures for the three domain levels**

  For canonical, oscillatory/multiscale, and coupled piezoelectric problems, repeat this fixed order: problem definition, reference solution, data/collocation generation, architecture/configuration, loss and training, evaluation, and advancement criterion. Keep unknown concrete problem choices and numerical settings in explicit TODO callouts.

- [ ] **Step 7: Operationalize metrics and statistical reporting**

  Include the relative $L_2$ error formula, global and critical-region residuals, stability, wall time/cost, paired-seed comparison, aggregation and uncertainty reporting. State that thresholds and the final run count require human approval before execution.

- [ ] **Step 8: Preserve and tighten the plan-of-work chapter**

  Keep the existing Gantt block unchanged and revise only its introductory prose so each phase corresponds to a methodological output and approval gate.

- [ ] **Step 9: Inspect the complete note**

  Confirm that headings follow the existing hierarchy, labels and block IDs are unique, citations use existing keys, unresolved choices are visible, and no result is claimed.

### Task 2: Verify Obsidian and staged-LaTeX compatibility

**Files:**
- Verify: `var/agent_runs/obsidian-bootstrap/Tesis-Studio/01_Secciones/70_Metodologia_y_Plan_de_Trabajo.md`
- Generate: `var/agent_runs/latex-drafts/methodology-preliminary/` through the existing stage command

**Interfaces:**
- Consumes: the edited Markdown note and the existing bootstrap manifest.
- Produces: test evidence, a staged `j-capitulos.tex`, and a diff confined to `var/agent_runs/latex-drafts/`.

- [ ] **Step 1: Run converter and workspace tests**

  Run:

  ```powershell
  py -m unittest tests.test_markdown_to_latex tests.test_thesis_workspace_stage -v
  ```

  Expected: all tests pass.

- [ ] **Step 2: Stage the edited vault without promoting it**

  Run:

  ```powershell
  py -m src.thesis_workspace.cli stage --root . --vault var/agent_runs/obsidian-bootstrap/Tesis-Studio --out var/agent_runs/latex-drafts/methodology-preliminary
  ```

  Expected: the command writes a staged `j-capitulos.tex`, its diff, and a manifest under the declared output directory; active `j-capitulos.tex` remains unchanged.

- [ ] **Step 3: Validate scientific and structural markers**

  Run:

  ```powershell
  rg -n "TBD|TODO|resultados obtenidos|se demostró|se comprobó" var/agent_runs/obsidian-bootstrap/Tesis-Studio/01_Secciones/70_Metodologia_y_Plan_de_Trabajo.md
  rg -n "sec:metodologia|sec:plan-trabajo|fig:cronograma-gantt" var/agent_runs/latex-drafts/methodology-preliminary/j-capitulos.tex
  ```

  Expected: unresolved scientific choices appear only as intentional Obsidian TODO callouts; no unsupported result claim appears; all three required LaTeX labels are present.

- [ ] **Step 4: Inspect scope**

  Run:

  ```powershell
  git status --short
  git diff --check
  ```

  Expected: no active thesis source, canonical vault note, policy file, workflow file, or bibliography authority was modified by this task.

