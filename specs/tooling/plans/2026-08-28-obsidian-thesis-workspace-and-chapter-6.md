# Obsidian Thesis Workspace and Chapter 6 Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe external Obsidian drafting workflow that preserves the active LaTeX sources and execute an evidence-backed revision of Chapter 6 from the integrated GAB feedback.

**Architecture:** A Python standard-library bridge generates a one-time Markdown bootstrap under `var/agent_runs/`, which the researcher copies to `C:/Users/rodri/Documents/Tesis-Studio`. Later imports read that external vault and generate reviewable LaTeX drafts and unified diffs under `var/agent_runs/latex-drafts/`; active `.tex` files are never overwritten. The Chapter 6 attack plan is represented as revision, decision, and evidence notes alongside the one-to-one section notes.

**Tech Stack:** Python 3.13 standard library (`argparse`, `dataclasses`, `hashlib`, `difflib`, `pathlib`, `tomllib`, `unittest`), Obsidian Markdown, LaTeX/MiKTeX, PowerShell, Git diff.

**Spec:** `docs/superpowers/specs/2026-08-28-obsidian-thesis-workspace-and-chapter-6-design.md`

## Estado de esta corrida aprobada

- Implementado: mapa estable de 14 módulos LaTeX a notas Markdown.
- Implementado: conversión conservadora, bootstrap interno, hashes y límites
  de escritura.
- Implementado: consenso, attack plan, ledger de 29 anotaciones, cuatro gates
  de decisión y backlog de citas/figuras.
- Implementado: staging Markdown → LaTeX con drafts, diffs y conflicto de
  hashes; los archivos .tex activos no se sobrescriben.
- Deliberadamente pendiente: la redacción científica del investigador, las
  decisiones del asesor, la verificación de fuentes primarias, la compilación
  final y cualquier promoción a los archivos .tex activos.
- Acción manual pendiente: copiar el payload a
  C:/Users/rodri/Documents/Tesis-Studio y abrir esa carpeta como bóveda de
  Obsidian. Esta escritura externa no la ejecuta la automatización del repo.

## Global Constraints

- Preserve every active root `.tex` file byte-for-byte during bootstrap and staging.
- Do not write outside `C:/Users/rodri/Documents/ProyectoGrad`; prepare the external-vault payload under `var/agent_runs/` for the researcher to copy.
- Read `C:/Users/rodri/Documents/Tesis-Studio` only after announcing the exact path.
- Never edit `context/`, `workflow/`, `AGENTS.md`, or `.senpi/extensions/` as part of this plan.
- Keep `PIKAN-Vault/PIKAN-Vault` as canonical literature; the new vault is a drafting workspace.
- Preserve `\cite`, `\label`, `\ref`, `\eqref`, equations, and unsupported LaTeX blocks losslessly; expose narrative citations and cross-references as reversible Obsidian links.
- All Markdown-to-LaTeX output goes to `var/agent_runs/latex-drafts/<run-id>/`.
- Scientific claims and active theory-text promotion remain human-gated.

---

## Planned file structure

```text
src/thesis_workspace/
├── __init__.py                 # public package surface
├── __main__.py                 # `py -m src.thesis_workspace`
├── cli.py                      # bootstrap and stage commands
├── model.py                    # SectionSpec and report dataclasses
├── sections.toml               # exact TeX-to-note map
├── latex_to_markdown.py        # conservative bootstrap conversion
├── markdown_to_latex.py        # inverse conversion for supported constructs
├── bootstrap.py                # internal bootstrap payload generation
├── stage.py                    # external read and internal draft/diff staging
└── templates/                  # dashboard, revision, decision, evidence notes

tests/
├── test_thesis_workspace_manifest.py
├── test_latex_to_markdown.py
├── test_thesis_workspace_bootstrap.py
├── test_chapter6_revision_notes.py
├── test_markdown_to_latex.py
└── test_thesis_workspace_stage.py
```

The converter modules own syntax preservation; `bootstrap.py` and `stage.py`
own filesystem boundaries; templates own prose and review-state structure.

---

### Task 1: Lock the section manifest and immutable-source contract

**Files:**
- Create: `src/thesis_workspace/__init__.py`
- Create: `src/thesis_workspace/sections.toml`
- Create: `src/thesis_workspace/model.py`
- Test: `tests/test_thesis_workspace_manifest.py`

**Interfaces:**
- Consumes: the active `z-main.tex` input graph and the design spec.
- Produces: `SectionSpec`, `load_section_specs(root: Path) -> list[SectionSpec]`, and the exact 14-file TeX-to-Markdown map.

- [ ] **Step 1: Write the failing manifest test** asserting the exact mapping from `a-prefacio.tex` through `o-glosario.tex`, excluding `z-main.tex`, configuration modules, and `m-bibliografia.bib`.
- [ ] **Step 2: Run the test** with `py -m unittest tests.test_thesis_workspace_manifest -v`; expect failure because `src.thesis_workspace` does not exist.
- [ ] **Step 3: Implement `SectionSpec`** as a frozen dataclass with `source_tex`, `note_path`, `chapter_title`, and `contains_own_chapter`.
- [ ] **Step 4: Add `sections.toml`** with these exact outputs: `00_Prefacio.md`, `01_Resumen.md`, `02_Abstract.md`, `10_Introduccion.md`, `20_Antecedentes.md`, `30_Justificacion.md`, `40_Objetivos.md`, `50_Alcance.md`, `60_Marco_Teorico.md`, `70_Metodologia_y_Plan_de_Trabajo.md`, `80_Conclusiones.md`, `90_Recomendaciones.md`, `95_Anexos.md`, and `99_Glosario.md`.
- [ ] **Step 5: Implement manifest loading** with `tomllib`, reject duplicate source/note paths, missing source files, absolute note paths, and mappings outside `01_Secciones/`.
- [ ] **Step 6: Run the manifest test** and expect all assertions to pass.
- [ ] **Step 7: Commit** with `git add src/thesis_workspace tests/test_thesis_workspace_manifest.py && git commit -m "feat: define thesis workspace section manifest"`.

### Task 2: Implement conservative LaTeX-to-Markdown conversion

**Files:**
- Create: `src/thesis_workspace/latex_to_markdown.py`
- Test: `tests/test_latex_to_markdown.py`

**Interfaces:**
- Consumes: `SectionSpec` and UTF-8 LaTeX text.
- Produces: `convert_latex_to_markdown(spec: SectionSpec, latex: str, source_hash: str) -> str`.

- [ ] **Step 1: Write failing tests** for chapter insertion, heading conversion, equation preservation, citation/reference preservation, label protection, TODO-callout conversion, lists, and raw `table`/`figure`/`landscape` blocks.
- [ ] **Step 2: Add a round-trip safety fixture** containing `\cite{raissi2019}`, `\eqref{eq:residual}`, `\label{eq:residual}`, `\pendiente{...}`, a display equation, and a table.
- [ ] **Step 3: Run** `py -m unittest tests.test_latex_to_markdown -v`; expect import or assertion failures.
- [ ] **Step 4: Implement a line-oriented protected-block scanner** for `equation`, `align`, `table`, `figure`, and `landscape`, rejecting unbalanced `\begin`/`\end` pairs.
- [ ] **Step 5: Implement safe inline transforms** only for headings, `\textit`, `\emph`, `\textbf`, and list items; leave all unknown commands unchanged.
- [ ] **Step 6: Emit frontmatter** with `kind`, absolute `source_tex`, basename `target_tex`, fixed `sync_direction`, computed `source_sha256`, and `status: imported`.
- [ ] **Step 7: Run tests** and confirm citation keys, references, and raw blocks are preserved after the reversible Markdown mapping.
- [ ] **Step 8: Commit** with `git add src/thesis_workspace/latex_to_markdown.py tests/test_latex_to_markdown.py && git commit -m "feat: convert thesis latex to safe markdown"`.

### Task 3: Generate the Obsidian bootstrap without touching external storage

**Files:**
- Create: `src/thesis_workspace/bootstrap.py`
- Create: `src/thesis_workspace/cli.py`
- Create: `src/thesis_workspace/__main__.py`
- Test: `tests/test_thesis_workspace_bootstrap.py`
- Generate ignored: `var/agent_runs/obsidian-bootstrap/Tesis-Studio/**`

**Interfaces:**
- Consumes: `load_section_specs()` and `convert_latex_to_markdown()`.
- Produces: `bootstrap_workspace(root: Path, output: Path) -> BootstrapReport` and CLI command `py -m src.thesis_workspace bootstrap --root . --out var/agent_runs/obsidian-bootstrap/Tesis-Studio`.

- [ ] **Step 1: Write a failing integration test** using temporary `.tex` fixtures and asserting 14 section notes, `00_Dashboard.md`, revision/decision/evidence directories, and `bootstrap-manifest.json`.
- [ ] **Step 2: Assert immutability** by hashing all source fixtures before and after bootstrap and requiring identical hashes.
- [ ] **Step 3: Run** `py -m unittest tests.test_thesis_workspace_bootstrap -v`; expect failure.
- [ ] **Step 4: Implement bootstrap generation** with refusal to write anywhere outside `var/agent_runs/obsidian-bootstrap/` when invoked from this repository.
- [ ] **Step 5: Generate `00_Dashboard.md`** linking all section, revision, decision, and evidence notes and showing status values.
- [ ] **Step 6: Generate `bootstrap-manifest.json`** containing source/note paths, SHA-256 hashes, conversion version, and UTC timestamp.
- [ ] **Step 7: Run the integration test** and confirm the source hashes remain unchanged.
- [ ] **Step 8: Run the real bootstrap command** and compare the generated note headings against `z-main.tex` plus the module headings.
- [ ] **Step 9: Have the researcher copy** `var/agent_runs/obsidian-bootstrap/Tesis-Studio` to `C:/Users/rodri/Documents/Tesis-Studio`, open that exact folder as an Obsidian vault, and confirm the dashboard and 14 section notes render.
- [ ] **Step 10: Commit** with `git add src/thesis_workspace tests/test_thesis_workspace_bootstrap.py && git commit -m "feat: generate obsidian thesis workspace bootstrap"`.

### Task 4: Add the Chapter 6 consensus, feedback ledger, and decision packets

**Files:**
- Create: `src/thesis_workspace/templates/02_Revision/Capitulo_6_Consenso.md`
- Create: `src/thesis_workspace/templates/02_Revision/Capitulo_6_Attack_Plan.md`
- Create: `src/thesis_workspace/templates/02_Revision/GAB_29_Anotaciones.md`
- Create: `src/thesis_workspace/templates/03_Decisiones/DEC-001_Punto_de_inyeccion.md`
- Create: `src/thesis_workspace/templates/03_Decisiones/DEC-002_Mecanismo_vs_aplicacion.md`
- Create: `src/thesis_workspace/templates/03_Decisiones/DEC-003_Senal_y_residual.md`
- Create: `src/thesis_workspace/templates/03_Decisiones/DEC-004_Notacion_del_residual.md`
- Test: `tests/test_chapter6_revision_notes.py`

**Interfaces:**
- Consumes: the three external review documents listed in the design evidence section.
- Produces: durable Markdown notes that distinguish consensus, open decisions, adviser questions, and execution order.

- [ ] **Step 1: Write a failing test** requiring all 29 GAB comments to appear exactly once with status `open`, `addressed`, `superseded`, or `needs_adviser`.
- [ ] **Step 2: Require the consensus note** to state mechanism thesis, pedagogical/argumentative complementarity, subsection entry/exit, restored classical adaptivity, 6.3 redistribution axis, 6.4/6.7 separation, and PIKAN's two roles.
- [ ] **Step 3: Require DEC-001** to compare A residual weighting, B sampling modulation, and C grid density, with B marked baseline/fallback and an explicit adviser decision field.
- [ ] **Step 4: Require DEC-002** to keep piezoelectricity as the culminating stress test unless the adviser explicitly changes the thesis identity.
- [ ] **Step 5: Require DEC-003** to distinguish problem property from observable signal and to test incremental information beyond residual collinearity.
- [ ] **Step 6: Require DEC-004** to recommend `r(x;theta) = N[u_theta](x) - f(x)` and list all equations/sections affected.
- [ ] **Step 7: Implement the notes** from the integrated wording in the spec; do not introduce new scientific claims.
- [ ] **Step 8: Run** `py -m unittest tests.test_chapter6_revision_notes -v` and expect pass.
- [ ] **Step 9: Commit** with `git add src/thesis_workspace tests/test_chapter6_revision_notes.py && git commit -m "docs: encode chapter 6 consensus and decisions"`.

### Task 5: Build the evidence and figure backlog before prose revision

**Files:**
- Create: `src/thesis_workspace/templates/04_Evidencia/Mapa_de_citas_Capitulo_6.md`
- Create: `src/thesis_workspace/templates/04_Evidencia/Figuras_requeridas.md`
- Verify: `m-bibliografia.bib`

**Interfaces:**
- Consumes: Chapter 6 section contracts and existing BibTeX keys.
- Produces: a claim-to-source matrix and five figure briefs; it does not modify the bibliography.

- [ ] **Step 1: Inventory every theoretical claim** in Sections 6.1–6.7 and classify it as definition, established claim, design choice, hypothesis, or pending adviser decision.
- [ ] **Step 2: Map each established claim** to an existing BibTeX key or mark it `source_needed`; no claim may be marked addressed by a secondary summary when a cited primary source exists.
- [ ] **Step 3: Verify the KAN primary source task**: determine whether grid extension exists, whether it is uniform, and how it differs from signal-guided local refinement before DEC-001 can select C.
- [ ] **Step 4: Populate the 6.3 redistribution matrix** for RAR, importance sampling, `hp`-VPINN, causal training, KAN, and PIKAN with one or more citations per row.
- [ ] **Step 5: Write figure briefs** for adaptive FEM mesh, MLP-global versus spline-local support, PINN loss, `s -> T -> w`, and the absent/static/mobile/anchored/between-fields test ladder.
- [ ] **Step 6: Review against GAB comments 2, 21, and 27** and require zero unreferenced theoretical paragraphs.
- [ ] **Step 7: Commit** with `git add src/thesis_workspace && git commit -m "docs: map chapter 6 evidence and figures"`.

### Task 6: Close the upstream scientific gates with the adviser

**Files:**
- Update in external vault by researcher: `03_Decisiones/DEC-001_Punto_de_inyeccion.md`
- Update in external vault by researcher: `03_Decisiones/DEC-002_Mecanismo_vs_aplicacion.md`
- Update in external vault by researcher: `03_Decisiones/DEC-003_Senal_y_residual.md`

**Interfaces:**
- Consumes: the A/B/C comparison, grid-extension verification, and test-bank role analysis.
- Produces: signed decisions that unblock Sections 6.2, 6.5, 6.6, and 6.7.

- [ ] **Step 1: Present A versus C as thesis choices** and B as baseline/fallback, including novelty, feasibility, and wording implications.
- [ ] **Step 2: Ask the adviser to confirm** that the work remains a mechanism thesis and piezoelectricity remains the culminating stress test.
- [ ] **Step 3: Ask for explicit review of printed pages 12–14**, because absence of annotations is not approval.
- [ ] **Step 4: Record decision owner, date, rationale, selected option, and rejected alternatives** in each decision note.
- [ ] **Step 5: Keep “learning effort” in all draft prose until DEC-001 explicitly selects internal grid density; only then permit “local capacity”.**

### Task 7: Restructure Chapter 6 in Markdown using section contracts

**Files:**
- Modify externally by researcher: `01_Secciones/60_Marco_Teorico.md`
- Update externally by researcher: `02_Revision/GAB_29_Anotaciones.md`
- Update externally by researcher: `02_Revision/Capitulo_6_Attack_Plan.md`

**Interfaces:**
- Consumes: closed DEC-001/002/003, evidence matrix, and figure briefs.
- Produces: a review-ready Markdown Chapter 6 preserving citation and cross-reference keys.

- [ ] **Step 1: Replace only the heading skeleton first** with 6.1–6.7 from the spec; move existing paragraphs under new headings without rewriting them.
- [ ] **Step 2: Add a one-paragraph entry and one-sentence handoff** to every section and subsection, satisfying GAB comments 1, 24, and 28.
- [ ] **Step 3: Rewrite 6.1** to define `x`, domain, operator, forcing, boundary/initial operators, closed-form versus numerical approximation, and the object of study.
- [ ] **Step 4: Rewrite 6.2** to define why mesh matters, then add FEM, AMR, `hp` refinement, and a-posteriori estimators; close with local allocation.
- [ ] **Step 5: Rewrite 6.3 once on the redistribution axis**; define PINN, MLP, KAN, and PIKAN at first use; populate the table rather than repeating SOTA in 6.5.
- [ ] **Step 6: Write 6.4 as a property space** with no Poisson, Burgers, Allen–Cahn, Navier–Stokes, or piezoelectric benchmark names.
- [ ] **Step 7: Constrain 6.5 to the empty row and research question**; target approximately one page.
- [ ] **Step 8: Complete 6.6** with introductory paragraph, `s -> T -> w`, types, selected injection, signal rationale, reduction cases, cost, mechanism name/acronym, and PIKAN-as-instrument closure.
- [ ] **Step 9: Rewrite 6.7** with the four-move template and explicit predictions: negative-control Poisson; concentrate/maintain Burgers; concentrate/release Allen–Cahn; boundary-layer versus wake Navier–Stokes; between-field piezoelectric focalization.
- [ ] **Step 10: Update all 29 GAB statuses** and require an explanatory note for any `superseded` comment.

### Task 8: Remove Chapter 1 duplication and enforce the page budget

**Files:**
- Modify externally by researcher: `01_Secciones/10_Introduccion.md`
- Verify externally by researcher: `01_Secciones/60_Marco_Teorico.md`

**Interfaces:**
- Consumes: the revised formal ramp in Sections 6.1–6.3.
- Produces: a narrative Chapter 1 that motivates the work without re-teaching PINNs/KANs.

- [ ] **Step 1: Mark paragraphs in Chapter 1** as motivation, formal definition, duplicated SOTA, or thesis positioning.
- [ ] **Step 2: Compress formal PINN and KAN explanations** to one sentence each and retain motivation, context, and relevance.
- [ ] **Step 3: Target approximately 40% reduction** in duplicated explanatory prose, not 40% of the entire chapter.
- [ ] **Step 4: Measure Chapter 6 block shares** and enforce: 6.1–6.3 <= 35%, 6.4–6.5 about 20%, 6.6 about 20%, 6.7 about 25%.
- [ ] **Step 5: Replace unnecessary AI/ML/DL history prose** with first-use definitions, tables, or the approved diagrams.

### Task 9: Implement Markdown-to-staged-LaTeX export and conflict guards

**Files:**
- Create: `src/thesis_workspace/markdown_to_latex.py`
- Create: `src/thesis_workspace/stage.py`
- Modify: `src/thesis_workspace/cli.py`
- Test: `tests/test_markdown_to_latex.py`
- Test: `tests/test_thesis_workspace_stage.py`
- Generate ignored: `var/agent_runs/latex-drafts/<run-id>/**`

**Interfaces:**
- Consumes: approved section notes and bootstrap manifest.
- Produces: `convert_markdown_to_latex(markdown: str) -> str`, staged `.tex`, `manifest.json`, and unified `.diff` files.

- [ ] **Step 1: Write failing round-trip tests** proving headings, equations, citations, labels, references, TODOs, and raw environments survive TeX → Markdown → TeX semantically.
- [ ] **Step 2: Write a failing conflict test** where the active `.tex` hash differs from `source_sha256`; require staging refusal with the changed path in the error.
- [ ] **Step 3: Run** `py -m unittest tests.test_markdown_to_latex tests.test_thesis_workspace_stage -v`; expect failure.
- [ ] **Step 4: Implement Markdown parsing** for the exact bootstrap constructs and pass fenced `latex` blocks through unchanged.
- [ ] **Step 5: Implement `stage --vault`** to read the announced external path and write only under `var/agent_runs/latex-drafts/<UTC-run-id>/`.
- [ ] **Step 6: Generate unified diffs** against active root `.tex` files using `difflib.unified_diff` and include changed citation/label inventories in `manifest.json`.
- [ ] **Step 7: Add hard refusal** for output paths resolving outside `var/agent_runs/latex-drafts/` or targeting an active `.tex` file.
- [ ] **Step 8: Run all workspace tests** and expect pass.
- [ ] **Step 9: Commit** with `git add src/thesis_workspace tests && git commit -m "feat: stage reviewed markdown as latex drafts"`.

### Task 10: Audit, compile, and promote only after approval

**Files:**
- Verify: `var/agent_runs/latex-drafts/<run-id>/*.tex`
- Verify: `var/agent_runs/latex-drafts/<run-id>/*.diff`
- Promote only with approval: `d-introduccion.tex`, `i-marco_teorico.tex`, and any other approved targets
- Generate ignored: `artifacts/latex/**`

**Interfaces:**
- Consumes: staged LaTeX, diffs, decision notes, and GAB ledger.
- Produces: an approved active revision and compiled PDF evidence.

- [ ] **Step 1: Compare citation keys** between each active and staged file; reject accidental removals not explained in the revision ledger.
- [ ] **Step 2: Compare labels/references** and reject missing targets or duplicated labels.
- [ ] **Step 3: Confirm all 29 GAB comments** are addressed, deliberately superseded, or explicitly awaiting adviser input.
- [ ] **Step 4: Confirm DEC-001 and DEC-002 are closed** before promoting Sections 6.5–6.7.
- [ ] **Step 5: Obtain explicit human approval** for each active `.tex` target and then apply only its reviewed diff.
- [ ] **Step 6: Compile** with `pdflatex -> biber -> pdflatex -> pdflatex` into `artifacts/latex/`.
- [ ] **Step 7: Search the log** for errors, undefined citations, undefined references, and duplicate labels.
- [ ] **Step 8: Render and visually review** Chapter 1 and Chapter 6, including every required table and figure.
- [ ] **Step 9: Run** `py -m unittest discover -s tests -v` and `py -3.12 -m src.graph_engineering validate --root .` when Python 3.12 is available; otherwise record the interpreter mismatch and run the supported project environment.
- [ ] **Step 10: Commit the approved academic revision** separately from bridge tooling so content and infrastructure can be reviewed independently.

## Recommended execution order

1. Tasks 1–5: create the safe drafting system and evidence/decision packet.
2. Task 6: stop for adviser decisions; this is a scientific gate, not an implementation delay.
3. Tasks 7–8: revise Chapter 6 and remove Chapter 1 duplication in Obsidian.
4. Task 9: stage the reviewed Markdown as LaTeX and generate diffs.
5. Task 10: approve, promote, compile, and visually review.
