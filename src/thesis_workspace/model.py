"""Configuration model for the thesis Markdown workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SectionSpec:
    source_tex: Path
    note_path: Path
    chapter_title: str
    contains_own_chapter: bool


def _validate_section(raw: dict[str, object], root: Path) -> SectionSpec:
    required = {
        "source_tex",
        "note_path",
        "chapter_title",
        "contains_own_chapter",
    }
    missing = required.difference(raw)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Section manifest entry is missing: {names}")

    source_rel = Path(str(raw["source_tex"]))
    note_rel = Path(str(raw["note_path"]))
    if source_rel.is_absolute():
        raise ValueError(f"Source path must be relative: {source_rel}")
    if note_rel.is_absolute() or ".." in note_rel.parts:
        raise ValueError(f"Note path must stay inside the workspace: {note_rel}")
    if note_rel.parts[:1] != ("01_Secciones",):
        raise ValueError(f"Note path must start with 01_Secciones/: {note_rel}")

    source = root / source_rel
    if not source.is_file():
        raise FileNotFoundError(source)

    return SectionSpec(
        source_tex=source,
        note_path=note_rel,
        chapter_title=str(raw["chapter_title"]),
        contains_own_chapter=bool(raw["contains_own_chapter"]),
    )


def load_section_specs(root: Path) -> list[SectionSpec]:
    """Load and validate the stable mapping from active TeX modules to notes."""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    manifest_path = Path(__file__).with_name("sections.toml")
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    raw_sections = data.get("section")
    if not isinstance(raw_sections, list):
        raise ValueError("Section manifest must contain [[section]] entries")

    specs = [_validate_section(dict(raw), root) for raw in raw_sections]
    sources = [spec.source_tex for spec in specs]
    notes = [spec.note_path for spec in specs]
    if len(set(sources)) != len(sources):
        raise ValueError("Section manifest contains duplicate source files")
    if len(set(notes)) != len(notes):
        raise ValueError("Section manifest contains duplicate note paths")
    return specs
