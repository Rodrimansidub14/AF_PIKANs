"""Generate a safe, reviewable Markdown bootstrap for the thesis workspace."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .bibliography import BibEntry, parse_bibtex, render_bibliography_note
from .latex_to_markdown import convert_latex_to_markdown
from .model import SectionSpec, load_section_specs


CONVERSION_VERSION = "latex-to-markdown-v2"
STAGING_ROOT_PARTS = ("var", "agent_runs", "obsidian-bootstrap")


@dataclass(frozen=True)
class BootstrapReport:
    """Summary of generated staging artifacts."""

    output: Path
    note_count: int
    note_paths: tuple[Path, ...]
    manifest_path: Path


def _staging_root(root: Path) -> Path:
    return (root.resolve() / Path(*STAGING_ROOT_PARTS)).resolve()


def _ensure_staging_output(root: Path, output: Path) -> Path:
    staging_root = _staging_root(root)
    resolved_output = output.resolve()
    try:
        resolved_output.relative_to(staging_root)
    except ValueError as exc:
        raise ValueError(
            "Bootstrap output must stay inside "
            f"{staging_root.as_posix()}"
        ) from exc
    return resolved_output


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_workspace_readme(path: Path, title: str, purpose: str) -> None:
    path.write_text(
        f"# {title}\n\n"
        f"{purpose}\n\n"
        "Este espacio es de trabajo humano. Las notas no promueven cambios "
        "automáticamente al proyecto LaTeX.\n",
        encoding="utf-8",
    )


def _copy_templates(output: Path) -> None:
    template_root = Path(__file__).with_name("templates")
    if not template_root.is_dir():
        return
    for template in sorted(template_root.rglob("*.md")):
        relative = template.relative_to(template_root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")


def _dashboard(
    specs: list[SectionSpec],
    *,
    include_bibliography: bool = False,
) -> str:
    rows = [
        "| Nota | Fuente LaTeX | Estado |",
        "| --- | --- | --- |",
    ]
    for spec in specs:
        note_link = spec.note_path.with_suffix("").as_posix()
        rows.append(
            f"| [[{note_link}]] | {spec.source_tex.name} | imported |"
        )
    return "\n".join(
        [
            "# Tesis Studio",
            "",
            "> Bóveda externa de trabajo para revisar y ajustar las secciones "
            "antes de producir un borrador LaTeX staged.",
            "",
            "## Estado",
            "",
            f"- **{len(specs)} notas** importadas desde los módulos activos.",
            "- Las fuentes .tex originales permanecen en ProyectoGrad.",
            "- Dirección permitida: Markdown revisado → LaTeX staged.",
            "- La promoción al proyecto requiere revisión humana explícita.",
            "",
            "## Secciones",
            "",
            *rows,
            "",
            "## Espacios de trabajo",
            "",
            *(
                ["- [[99_Bibliografia]] — índice interactivo de citaciones.", ""]
                if include_bibliography
                else []
            ),
            "- [[02_Revision/README]] — ledger y revisiones.",
            "- [[03_Decisiones/README]] — decisiones pendientes y aprobaciones.",
            "- [[04_Evidencia/README]] — citas, evidencia y figuras.",
            "",
            "## Flujo",
            "",
            "1. Editar una nota en Obsidian.",
            "2. Revisar cambios y decisiones en esta bóveda.",
            "3. Generar un borrador LaTeX en staging.",
            "4. Comparar el diff y aprobar manualmente cualquier promoción.",
            "",
        ],
    )


def _manifest(
    root: Path,
    specs: list[SectionSpec],
    hashes: dict[Path, str],
    note_hashes: dict[Path, str],
    bibliography: dict[str, object] | None = None,
) -> dict[str, object]:
    sections = []
    for spec in specs:
        relative_source = spec.source_tex.relative_to(root).as_posix()
        sections.append(
            {
                "source_tex": relative_source,
                "note_path": spec.note_path.as_posix(),
                "target_tex": spec.source_tex.name,
                "chapter_title": spec.chapter_title,
                "source_sha256": hashes[spec.source_tex],
                "note_sha256": note_hashes[spec.note_path],
                "status": "imported",
            }
        )
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest: dict[str, object] = {
        "kind": "thesis_workspace_bootstrap",
        "conversion_version": CONVERSION_VERSION,
        "created_at_utc": timestamp.replace("+00:00", "Z"),
        "source_root": root.as_posix(),
        "sections": sections,
    }
    if bibliography is not None:
        manifest["bibliography"] = bibliography
    return manifest


def bootstrap_workspace(root: Path, output: Path) -> BootstrapReport:
    """Generate the external-vault payload under the repository staging area.

    The function never writes to the external Obsidian path. The output must
    resolve below root/var/agent_runs/obsidian-bootstrap.
    """

    root = root.resolve()
    output = _ensure_staging_output(root, output)
    specs = load_section_specs(root)
    bib_path = root / "m-bibliografia.bib"
    bibliography_entries: list[BibEntry] = []
    if bib_path.is_file():
        bibliography_entries = parse_bibtex(bib_path.read_text(encoding="utf-8"))
    bibliography_by_key = {
        entry.key: entry for entry in bibliography_entries
    }
    output.mkdir(parents=True, exist_ok=True)
    if bib_path.is_file():
        (output / bib_path.name).write_bytes(bib_path.read_bytes())

    for directory in ("01_Secciones", "02_Revision", "03_Decisiones", "04_Evidencia"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    _write_workspace_readme(
        output / "02_Revision" / "README.md",
        "Revisión",
        "Registra observaciones, consenso y cambios pendientes por sección.",
    )
    _write_workspace_readme(
        output / "03_Decisiones" / "README.md",
        "Decisiones",
        "Registra decisiones que requieren aprobación humana antes de promover.",
    )
    _write_workspace_readme(
        output / "04_Evidencia" / "README.md",
        "Evidencia",
        "Registra fuentes, citas, figuras y pruebas que respaldan cada afirmación.",
    )
    _copy_templates(output)

    hashes: dict[Path, str] = {}
    note_hashes: dict[Path, str] = {}
    note_paths: list[Path] = []
    for spec in specs:
        source_bytes = spec.source_tex.read_bytes()
        source_hash = _sha256(source_bytes)
        hashes[spec.source_tex] = source_hash
        note_path = output / spec.note_path
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note = convert_latex_to_markdown(
            spec,
            source_bytes.decode("utf-8"),
            source_hash,
            bibliography=bibliography_by_key,
        )
        note_path.write_text(note, encoding="utf-8")
        note_hashes[spec.note_path] = _sha256(note_path.read_bytes())
        note_paths.append(note_path)

    bibliography_metadata: dict[str, object] | None = None
    if bibliography_entries:
        bibliography_note = output / "99_Bibliografia.md"
        bibliography_note.write_text(
            render_bibliography_note(
                bibliography_entries,
                bib_path.relative_to(root).as_posix(),
            ),
            encoding="utf-8",
        )
        bibliography_metadata = {
            "source_bib": bib_path.relative_to(root).as_posix(),
            "note_path": bibliography_note.relative_to(output).as_posix(),
            "source_sha256": _sha256(bib_path.read_bytes()),
            "entry_count": len(bibliography_entries),
        }

    (output / "00_Dashboard.md").write_text(
        _dashboard(specs, include_bibliography=bool(bibliography_entries)),
        encoding="utf-8",
    )
    manifest_path = output / "bootstrap-manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                root,
                specs,
                hashes,
                note_hashes,
                bibliography=bibliography_metadata,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return BootstrapReport(
        output=output,
        note_count=len(note_paths),
        note_paths=tuple(note_paths),
        manifest_path=manifest_path,
    )
