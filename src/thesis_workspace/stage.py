"""Stage reviewed Markdown as LaTeX drafts and unified diffs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re

from .markdown_to_latex import convert_markdown_to_latex
from .model import SectionSpec, load_section_specs


STAGE_ROOT_PARTS = ("var", "agent_runs", "latex-drafts")


class SourceConflictError(RuntimeError):
    """Raised when an active TeX source changed since bootstrap."""


@dataclass(frozen=True)
class StageReport:
    output: Path
    section_count: int
    draft_paths: tuple[Path, ...]
    manifest_path: Path


def _stage_root(root: Path) -> Path:
    return (root.resolve() / Path(*STAGE_ROOT_PARTS)).resolve()


def _ensure_stage_output(root: Path, output: Path) -> Path:
    stage_root = _stage_root(root)
    resolved_output = output.resolve()
    try:
        resolved_output.relative_to(stage_root)
    except ValueError as exc:
        raise ValueError(
            f"Stage output must stay inside {stage_root.as_posix()}"
        ) from exc
    return resolved_output


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inventory(pattern: str, text: str) -> list[str]:
    values: set[str] = set()
    for match in re.finditer(pattern, text):
        values.update(value.strip() for value in match.group(1).split(",") if value.strip())
    return sorted(values)


def _citations(text: str) -> list[str]:
    return _inventory(r"\\cite[a-zA-Z]*\{([^{}]+)\}", text)


def _labels(text: str) -> list[str]:
    return _inventory(r"\\label\{([^{}]+)\}", text)


def _references(text: str) -> list[str]:
    return _inventory(r"\\(?:ref|eqref)\{([^{}]+)\}", text)


def _frontmatter_value(markdown: str, key: str) -> str | None:
    in_frontmatter = False
    for line in markdown.splitlines():
        if line.strip() == "---":
            if in_frontmatter:
                return None
            in_frontmatter = True
            continue
        if in_frontmatter and line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip('"')
    return None


def _manifest_by_note(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    sections = manifest.get("sections")
    if not isinstance(sections, list):
        raise ValueError("Bootstrap manifest must contain a sections list")
    result: dict[str, dict[str, object]] = {}
    for raw in sections:
        if not isinstance(raw, dict) or "note_path" not in raw:
            raise ValueError("Invalid section entry in bootstrap manifest")
        note_path = str(raw["note_path"])
        if note_path in result:
            raise ValueError(f"Duplicate note in bootstrap manifest: {note_path}")
        result[note_path] = dict(raw)
    return result


def _safe_output_file(output: Path, name: str) -> Path:
    target = (output / name).resolve()
    try:
        target.relative_to(output)
    except ValueError as exc:
        raise ValueError(f"Stage target escapes output directory: {name}") from exc
    if target.exists() and target.is_file() and target.resolve() != output / name:
        raise ValueError(f"Stage target is a symlink: {target}")
    return target


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stage_workspace(
    root: Path,
    vault: Path,
    output: Path | None = None,
) -> StageReport:
    """Read a vault and create reviewable LaTeX drafts under repository staging."""

    root = root.resolve()
    vault = vault.resolve()
    if not vault.is_dir():
        raise FileNotFoundError(vault)
    specs = load_section_specs(root)
    manifest_path = vault / "bootstrap-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_note = _manifest_by_note(manifest)

    prepared: list[tuple[SectionSpec, str, str, str, str]] = []
    for spec in specs:
        note_key = spec.note_path.as_posix()
        entry = by_note.get(note_key)
        if entry is None:
            raise ValueError(f"Section is missing from bootstrap manifest: {note_key}")
        expected_source = str(entry.get("source_tex", ""))
        actual_source = spec.source_tex.relative_to(root).as_posix()
        if expected_source != actual_source:
            raise ValueError(
                f"Manifest source mismatch for {note_key}: "
                f"{expected_source!r} != {actual_source!r}"
            )
        expected_hash = str(entry.get("source_sha256", ""))
        source_bytes = spec.source_tex.read_bytes()
        actual_hash = _sha256(source_bytes)
        if actual_hash != expected_hash:
            raise SourceConflictError(
                f"Active source changed since bootstrap: "
                f"{spec.source_tex.name}"
            )
        note_path = vault / spec.note_path
        if not note_path.is_file():
            raise FileNotFoundError(note_path)
        note_bytes = note_path.read_bytes()
        markdown = note_bytes.decode("utf-8")
        note_hash = _frontmatter_value(markdown, "source_sha256")
        if note_hash != expected_hash:
            raise SourceConflictError(
                f"Note metadata does not match bootstrap hash: "
                f"{spec.note_path.as_posix()}"
            )
        note_digest = _sha256(note_bytes)
        if note_digest == str(entry.get("note_sha256", "")):
            staged = source_bytes.decode("utf-8")
        else:
            staged = convert_markdown_to_latex(markdown)
        prepared.append(
            (
                spec,
                source_bytes.decode("utf-8"),
                staged,
                expected_hash,
                note_key,
            )
        )

    if output is None:
        output = _stage_root(root) / _run_id()
    output = _ensure_stage_output(root, output)
    output.mkdir(parents=True, exist_ok=True)

    draft_paths: list[Path] = []
    manifest_sections: list[dict[str, object]] = []
    for spec, active_text, staged_text, source_hash, note_key in prepared:
        draft_path = _safe_output_file(output, spec.source_tex.name)
        diff_path = _safe_output_file(output, f"{spec.source_tex.stem}.diff")
        diff = "".join(
            difflib.unified_diff(
                active_text.splitlines(keepends=True),
                staged_text.splitlines(keepends=True),
                fromfile=spec.source_tex.as_posix(),
                tofile=draft_path.as_posix(),
            )
        )
        draft_path.write_text(staged_text, encoding="utf-8")
        diff_path.write_text(diff, encoding="utf-8")
        draft_paths.append(draft_path)

        active_citations = _citations(active_text)
        staged_citations = _citations(staged_text)
        active_labels = _labels(active_text)
        staged_labels = _labels(staged_text)
        manifest_sections.append(
            {
                "source_tex": spec.source_tex.relative_to(root).as_posix(),
                "note_path": note_key,
                "staged_tex": draft_path.name,
                "diff": diff_path.name,
                "source_sha256": source_hash,
                "citations_active": active_citations,
                "citations_staged": staged_citations,
                "citations_added": sorted(set(staged_citations) - set(active_citations)),
                "citations_removed": sorted(set(active_citations) - set(staged_citations)),
                "labels_active": active_labels,
                "labels_staged": staged_labels,
                "labels_added": sorted(set(staged_labels) - set(active_labels)),
                "labels_removed": sorted(set(active_labels) - set(staged_labels)),
                "references_active": _references(active_text),
                "references_staged": _references(staged_text),
            }
        )

    staged_manifest = {
        "kind": "thesis_latex_stage",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_root": root.as_posix(),
        "vault": vault.as_posix(),
        "sections": manifest_sections,
    }
    staged_manifest_path = _safe_output_file(output, "manifest.json")
    staged_manifest_path.write_text(
        json.dumps(staged_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return StageReport(
        output=output,
        section_count=len(draft_paths),
        draft_paths=tuple(draft_paths),
        manifest_path=staged_manifest_path,
    )
