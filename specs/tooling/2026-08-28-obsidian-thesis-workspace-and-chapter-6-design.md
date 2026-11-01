# Obsidian Thesis Workspace and Chapter 6 Revision — Design

**Date:** 2026-08-28  
**Repository:** `C:/Users/rodri/Documents/ProyectoGrad`  
**External workspace:** `C:/Users/rodri/Documents/Tesis-Studio`

## Goal

Create a human-owned Obsidian drafting workspace outside `ProyectoGrad`, copy
the current thesis sections from LaTeX to editable Markdown without modifying
the original `.tex` files, and establish a reviewed one-way route from edited
Markdown back to staged LaTeX drafts. Use the workspace to execute the Chapter
6 revision agreed from GAB's annotations and the two integration documents.

## Evidence inputs

- `C:/Users/rodri/.codex/attachments/ddc84932-50b9-4781-8c94-8127068f25a9/pasted-text.txt` — integrated Chapter 6 justification v2.
- `C:/Users/rodri/.codex/attachments/62e9822a-474b-46a4-a1a1-ee908ece61bf/pasted-text.txt` — convergence analysis between GAB and the structural review.
- `C:/Users/rodri/.codex/attachments/70be24e7-a321-43b3-a866-1ef81eeb39aa/pasted-text.txt` — transcription of GAB's 29 annotations.
- `C:/Users/rodri/Documents/ProyectoGrad/i-marco_teorico.tex` — current Chapter 6 source.
- `C:/Users/rodri/Documents/ProyectoGrad/d-introduccion.tex` — current Chapter 1 source affected by deduplication.
- `C:/Users/rodri/Documents/ProyectoGrad/m-bibliografia.bib` — citation-key authority.

## Source-of-truth boundaries

- `ProyectoGrad/*.tex` remains the compilable source until a human explicitly
  approves a staged Markdown-to-LaTeX promotion.
- `C:/Users/rodri/Documents/Tesis-Studio` is a drafting workspace, not a
  scientific source of truth and not a Git checkout.
- `PIKAN-Vault/PIKAN-Vault` remains the canonical literature and concept vault.
- `m-bibliografia.bib` remains the citation-key authority.
- Generated bootstrap files and LaTeX drafts live under `var/agent_runs/`.
- Automation never overwrites active `.tex` files.
- Because project policy forbids external writes, automation prepares the
  vault bootstrap inside `var/agent_runs/`; the researcher copies it to the
  external path. Subsequent import reads the external vault and writes only to
  `var/agent_runs/`.

## Workspace topology

```text
Tesis-Studio/
├── 00_Dashboard.md
├── 01_Secciones/
│   ├── 00_Prefacio.md
│   ├── 01_Resumen.md
│   ├── 02_Abstract.md
│   ├── 10_Introduccion.md
│   ├── 20_Antecedentes.md
│   ├── 30_Justificacion.md
│   ├── 40_Objetivos.md
│   ├── 50_Alcance.md
│   ├── 60_Marco_Teorico.md
│   ├── 70_Metodologia_y_Plan_de_Trabajo.md
│   ├── 80_Conclusiones.md
│   ├── 90_Recomendaciones.md
│   ├── 95_Anexos.md
│   └── 99_Glosario.md
├── 02_Revision/
│   ├── Capitulo_6_Consenso.md
│   ├── Capitulo_6_Attack_Plan.md
│   └── GAB_29_Anotaciones.md
├── 03_Decisiones/
│   ├── DEC-001_Punto_de_inyeccion.md
│   ├── DEC-002_Mecanismo_vs_aplicacion.md
│   ├── DEC-003_Senal_y_residual.md
│   └── DEC-004_Notacion_del_residual.md
└── 04_Evidencia/
    ├── Mapa_de_citas_Capitulo_6.md
    └── Figuras_requeridas.md
```

The section notes preserve the current one-file-per-module structure. In
particular, `j-capitulos.tex` remains one Markdown note containing both
`Metodología` and `Plan de trabajo`; the bootstrap does not redesign chapter
boundaries.

## Section metadata contract

Every file under `01_Secciones/` has frontmatter with these exact keys:

```yaml
---
kind: thesis_section
source_tex: C:/Users/rodri/Documents/ProyectoGrad/i-marco_teorico.tex
target_tex: i-marco_teorico.tex
chapter_title: Marco teórico
contains_own_chapter: false
sync_direction: reviewed_markdown_to_staged_latex
source_sha256: generated-at-bootstrap
status: imported
---
```

`source_sha256` is generated from the original file and is used only as a
conflict guard. `chapter_title` and `contains_own_chapter` let the inverse
converter distinguish a synthetic chapter heading from a real one.
`bootstrap-manifest.json` also records `note_sha256`; an unchanged imported
note stages as a byte-identical no-op. `status` may be `imported`,
`editing`, `ready_for_review`, or `approved_for_staging`.

## Conservative conversion rules

- `\chapter`, `\section`, and `\subsection` become Markdown headings.
- Chapter headings supplied by `z-main.tex` are inserted into modules that do
  not carry their own `\chapter` command.
- Display equations remain LaTeX math delimited by `$$`.
- `m-bibliografia.bib` remains the citation-key authority and generates a
  root-level `99_Bibliografia.md` index with one stable block per key. In
  narrative drafting text, `\cite...{...}` becomes a protected source marker
  plus one Obsidian wikilink per key; the reverse converter restores the
  original citation command and key order. Citations inside protected raw
  LaTeX environments remain unchanged inside the fence; a separate visible
  link summary is added outside the fence so those environments stay lossless
  while their references remain navigable.
- `\label{...}` is emitted as protected label metadata plus a deterministic
  Obsidian block ID, and is restored inside the corresponding LaTeX
  environment on export.
- Leading LaTeX comments and `\newcommand` definitions are kept in a hidden
  protected preamble block and restored on export.
- Simple emphasis and itemized/enumerated lists become Markdown.
- `\pendiente{...}` becomes an Obsidian TODO callout with a protected source
  marker so it can be reconstructed.
- Tables, figures, landscape blocks, and unknown/custom commands remain fenced
  `latex` blocks and round-trip byte-for-byte.
- Conversion fails closed on unbalanced environments or unsupported commands
  outside protected raw blocks.

Pandoc is not installed on the current host. The bridge therefore uses a
small Python standard-library converter tailored to the active thesis command
surface instead of adding a system dependency.

## Reviewed export flow

```text
active .tex --bootstrap--> var/agent_runs/obsidian-bootstrap/Tesis-Studio
                                      |
                                      | human copy
                                      v
                         C:/Users/rodri/Documents/Tesis-Studio
                                      |
                                      | read-only import
                                      v
             var/agent_runs/latex-drafts/<run-id>/*.tex + unified diffs
                                      |
                                      | scientific/human approval
                                      v
                              active .tex promotion
```

Import refuses to stage a note when its original `.tex` hash has changed since
bootstrap. Reconciliation must happen explicitly; there is no two-way merge.

## Chapter 6 consensus

1. The work is a **mechanism thesis**, not an application thesis.
2. The object is not PDEs themselves; it is how approximation effort or
   capacity is distributed over a domain.
3. GAB's feedback and the structural review address different layers and are
   compatible: pedagogy versus argumentation.
4. Each subsection must be expositorily self-contained while remaining
   argumentatively dependent. Each gets an entry paragraph and a closing
   handoff sentence.
5. The classical adaptive precedent must remain: FEM mesh, AMR, `hp`
   refinement, and a-posteriori estimators establish that local allocation is
   not an arbitrary invention.
6. Section 6.3 presents the state of the art once, on the fixed axis “what does
   each strategy redistribute?” Section 6.5 only identifies the empty row.
7. Section 6.4 defines mathematical properties without naming benchmarks;
   Section 6.7 instantiates those properties in the test bank.
8. PIKAN has two distinct roles: state-of-the-art object in 6.3 and measurement
   substrate, justified by local support, at the end of 6.6.
9. No annotations on printed pages 12–14 are not evidence of approval; the
   injection point, cost, later stress tests, and piezoelectric case require an
   explicit adviser review.

## Upstream scientific gate

The prose of Sections 6.2, 6.5, and 6.6 cannot be finalized until the injection
point is selected:

- **A — residual weighting:** defensible claim is redistribution of learning
  effort; lowest execution risk.
- **B — sampling modulation:** RAR on PIKAN; useful as baseline/fallback, but
  narrows the novelty claim.
- **C — internal grid density:** literal redistribution of local capacity;
  strongest AMR analogy and highest implementation risk.

The decision packet should compare A and C as thesis candidates and retain B
as a control/fallback. Until the gate closes, draft language uses “learning
effort”, not “capacity”. Before selecting C, verify in the primary KAN source
whether grid extension exists and whether it is uniform/programmed.

## Revised Chapter 6 contract

| Section | Contract | Acceptance condition |
|---|---|---|
| 6.1 | PDEs as approximation problems | Defines `x`, domain, operator, forcing, conditions, analytical/numerical solution, and the object of study |
| 6.2 | Classical numerical approximation and adaptivity | Defines why mesh matters; covers FEM, AMR, `hp`, and a-posteriori error; closes on local allocation |
| 6.3 | Physics-informed SOTA | Defines PINN/MLP/KAN/PIKAN at first use; populates the redistribution-axis table with citations |
| 6.4 | Mathematical signature of difficulty | Defines oscillation, localized gradient, curvature, nonlinearity, moving interface, and coupling without benchmarks |
| 6.5 | Local allocation gap | Points to the empty table row, asks the research question, and stays near one page |
| 6.6 | Adaptive focalization mechanism | Has an intro before 6.6.1; defines `s -> T -> w`, types, injection, reductions, cost, name/acronym, and PIKAN-as-instrument |
| 6.7 | Stress-test bank | Uses equation → solution behavior → 6.4 signature → experimental role, then states falsifiable predictions |

## Page and media budget

- Sections 6.1–6.3: at most 35% of Chapter 6.
- Sections 6.4–6.5: approximately 20%.
- Section 6.6: approximately 20%.
- Section 6.7: approximately 25%.
- Use concise first-use definitions, not separate AI/ML/DL history sections.
- Required visuals: adaptive FEM mesh; MLP global support versus B-spline local
  support; PINN loss decomposition; `s -> T -> w`; stress-test ladder.

## Promotion gates

- A staged draft cannot modify an active `.tex` file.
- A scientific-claim review is required before changing the mechanism/gap text.
- The adviser must review DEC-001 and DEC-002 before Sections 6.5–6.7 are marked
  `ready_for_review`.
- Promotion checks citation keys, labels, references, acronym first use, page
  budget, LaTeX compilation, and a visual PDF review.
