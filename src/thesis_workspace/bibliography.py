"""Small, dependency-free BibTeX index for the thesis Obsidian workspace."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


_ENTRY_START = re.compile(r"@(?P<kind>[A-Za-z]+)\s*(?P<opening>[({])")
_FIELD_START = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*=")
_IGNORED_ENTRIES = {"comment", "preamble", "string"}


@dataclass(frozen=True)
class BibEntry:
    """The citation fields needed to render a navigable bibliography note."""

    entry_type: str
    key: str
    fields: dict[str, str]

    @property
    def citation_label(self) -> str:
        return format_citation_label(self)


def _matching_delimiter(text: str, opening: int, opener: str) -> int:
    closer = "}" if opener == "{" else ")"
    depth = 1
    quoted = False
    escaped = False
    for index in range(opening + 1, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced BibTeX entry")


def _top_level_comma(text: str) -> int:
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            return index
    return -1


def _read_value(text: str, start: int) -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        raise ValueError("BibTeX field is missing its value")

    if text[start] == "{":
        end = _matching_delimiter(text, start, "{")
        return text[start + 1 : end], end + 1
    if text[start] == '"':
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return text[start + 1 : index], index + 1
        raise ValueError("Unbalanced quoted BibTeX value")

    end = start
    while end < len(text) and text[end] != ",":
        end += 1
    return text[start:end].strip(), end


def _parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    index = 0
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] == ","):
            index += 1
        if index >= len(text):
            break
        match = _FIELD_START.match(text, index)
        if not match:
            raise ValueError(f"Invalid BibTeX field near: {text[index:index + 30]!r}")
        name = match.group(1).lower()
        value, index = _read_value(text, match.end())
        if name in fields:
            raise ValueError(f"Duplicate BibTeX field: {name}")
        fields[name] = value.strip()
    return fields


def parse_bibtex(text: str) -> list[BibEntry]:
    """Parse ordinary BibTeX entries without requiring a third-party package."""

    entries: list[BibEntry] = []
    seen_keys: set[str] = set()
    cursor = 0
    while True:
        match = _ENTRY_START.search(text, cursor)
        if match is None:
            break
        opening = match.group("opening")
        closing = _matching_delimiter(text, match.end() - 1, opening)
        body = text[match.end() : closing]
        comma = _top_level_comma(body)
        if comma < 0:
            raise ValueError(f"BibTeX entry has no citation key: {match.group(0)}")
        key = body[:comma].strip()
        if not key:
            raise ValueError("BibTeX entry has an empty citation key")
        kind = match.group("kind").lower()
        if kind not in _IGNORED_ENTRIES:
            if key in seen_keys:
                raise ValueError(f"Duplicate BibTeX citation key: {key}")
            seen_keys.add(key)
            entries.append(
                BibEntry(
                    entry_type=kind,
                    key=key,
                    fields=_parse_fields(body[comma + 1 :]),
                )
            )
        cursor = closing + 1
    return entries


def _clean_value(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip()
    while len(cleaned) >= 2 and (
        (cleaned.startswith("{") and cleaned.endswith("}"))
        or (cleaned.startswith('"') and cleaned.endswith('"'))
    ):
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"[{}]", "", cleaned)
    return " ".join(cleaned.split())


def _year(entry: BibEntry) -> str:
    value = _clean_value(entry.fields.get("date") or entry.fields.get("year"))
    match = re.search(r"\b(?:18|19|20|21)\d{2}\b", value)
    return match.group(0) if match else value


def _surname(author: str) -> str:
    author = _clean_value(author)
    if "," in author:
        return author.split(",", 1)[0].strip()
    words = author.split()
    return words[-1] if words else author


def format_citation_label(entry: BibEntry) -> str:
    """Return a compact author-year label for an Obsidian citation link."""

    raw_author = entry.fields.get("author", "")
    authors = [part.strip() for part in raw_author.split(" and ") if part.strip()]
    names = [_surname(author) for author in authors]
    if not names:
        author_label = entry.key
    elif len(names) >= 3 or any(name.lower() == "others" for name in names):
        author_label = f"{names[0]} et al."
    elif len(names) == 2:
        author_label = f"{names[0]} & {names[1]}"
    else:
        author_label = names[0]
    year = _year(entry)
    return f"{author_label} ({year})" if year else author_label


def citation_block_id(key: str) -> str:
    """Map a BibTeX key to a stable Obsidian block ID."""

    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", key).strip("-")
    if not safe:
        raise ValueError(f"BibTeX key cannot produce a block ID: {key!r}")
    return f"cite-{safe}"


def render_bibliography_note(
    entries: list[BibEntry],
    source_bib: str,
) -> str:
    """Render a human-readable, block-addressable bibliography index."""

    lines = [
        "---",
        "kind: thesis_bibliography",
        f"source_bib: {json.dumps(source_bib, ensure_ascii=False)}",
        f"entry_count: {len(entries)}",
        "sync_direction: generated_from_bibtex",
        "---",
        "",
        "# Bibliografía",
        "",
        "> Índice interactivo generado desde `m-bibliografia.bib`. Las claves "
        "BibTeX son la autoridad y no deben renombrarse.",
        "",
    ]
    for entry in entries:
        title = _clean_value(entry.fields.get("title")) or entry.key
        lines.extend(
            [
                f"## {entry.citation_label} — {title}",
                "",
                f"^{citation_block_id(entry.key)}",
                "",
                f"- **Clave:** `{entry.key}`",
                f"- **Tipo:** `{entry.entry_type}`",
            ]
        )
        author = _clean_value(entry.fields.get("author"))
        year = _year(entry)
        doi = _clean_value(entry.fields.get("doi"))
        url = _clean_value(entry.fields.get("url"))
        note = _clean_value(entry.fields.get("note"))
        if author:
            lines.append(f"- **Autoría:** {author}")
        if year:
            lines.append(f"- **Año:** {year}")
        if doi:
            lines.append(f"- **DOI:** [{doi}](https://doi.org/{doi})")
        if url:
            lines.append(f"- **URL:** [{url}]({url})")
        if note:
            lines.append(f"- **Nota:** {note}")
        lines.extend(["", "---", ""])
    return "\n".join(lines)
