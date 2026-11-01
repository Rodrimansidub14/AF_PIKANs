"""Conservative conversion of the active thesis modules to Markdown."""

from __future__ import annotations

import re
import json
from collections.abc import Mapping

from .bibliography import BibEntry, citation_block_id
from .model import SectionSpec


RAW_ENVIRONMENTS = {
    "align",
    "align*",
    "figure",
    "landscape",
    "table",
    "tabular",
    "tabularx",
}
EQUATION_ENVIRONMENTS = {"equation", "equation*"}
PROTECTED_ENVIRONMENTS = RAW_ENVIRONMENTS | EQUATION_ENVIRONMENTS
_LABEL_COMMAND = re.compile(r"\\label\{([^{}]+)\}")
_REFERENCE_COMMAND = re.compile(r"\\(eqref|ref)\{([^{}]+)\}")
_CITATION_COMMAND = re.compile(
    r"\\(?P<command>cite[a-zA-Z]*)(?P<options>(?:\[[^\]\r\n]*\])*)"
    r"\{(?P<keys>[^{}]+)\}"
)
_BIBLIOGRAPHY_NOTE = "99_Bibliografia"


def _find_braced_end(text: str, opening: int) -> int:
    if opening >= len(text) or text[opening] != "{":
        raise ValueError("Expected an opening brace")
    depth = 0
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced braces in LaTeX command")


def _replace_braced_command(text: str, command: str, replacement) -> str:
    needle = f"\\{command}{{"
    cursor = 0
    pieces: list[str] = []
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            pieces.append(text[cursor:])
            return "".join(pieces)
        opening = start + len(needle) - 1
        closing = _find_braced_end(text, opening)
        pieces.append(text[cursor:start])
        pieces.append(replacement(text[opening + 1 : closing]))
        cursor = closing + 1


def _citation_link(key: str, bibliography: Mapping[str, BibEntry] | None) -> str:
    entry = bibliography.get(key) if bibliography is not None else None
    label = entry.citation_label if entry is not None else key
    return f"[[{_BIBLIOGRAPHY_NOTE}#^{citation_block_id(key)}|{label}]]"


def _convert_citations(
    text: str,
    bibliography: Mapping[str, BibEntry] | None = None,
) -> str:
    def replacement(match: re.Match[str]) -> str:
        source = match.group(0)
        keys = [key.strip() for key in match.group("keys").split(",") if key.strip()]
        links = ", ".join(_citation_link(key, bibliography) for key in keys)
        return f"<!-- latex-citation: {source} -->{links}"

    return _CITATION_COMMAND.sub(replacement, text)


def _convert_inline(
    text: str,
    bibliography: Mapping[str, BibEntry] | None = None,
) -> str:
    text = _replace_braced_command(text, "textbf", lambda value: f"**{value}**")
    text = _replace_braced_command(text, "textit", lambda value: f"*{value}*")
    text = _replace_braced_command(text, "emph", lambda value: f"*{value}*")
    text = _convert_references(text)
    return _convert_citations(text, bibliography)


def _convert_pending(
    text: str,
    bibliography: Mapping[str, BibEntry] | None = None,
) -> str:
    def replacement(value: str) -> str:
        normalized = " ".join(value.split())
        visible = _convert_citations(normalized, bibliography)
        original = f"\\pendiente{{{normalized}}}"
        return (
            "\n> [!todo] Pendiente\n"
            f"> {visible}\n"
            f"<!-- latex-pendiente: {original} -->\n"
        )

    converted = _replace_braced_command(text, "pendiente", replacement)
    parts = converted.splitlines()
    if parts:
        parts[0] = parts[0].rstrip()
        parts[-1] = parts[-1].lstrip()
    return "\n".join(parts)


def _label_to_block_id(label: str) -> str:
    prefix, separator, suffix = label.partition(":")
    if not separator:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")
    else:
        safe_suffix = re.sub(r"[^A-Za-z0-9_-]+", "-", suffix).strip("-")
        safe = f"{prefix}-{safe_suffix}"
    return f"^{safe}"


def _convert_references(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        command, label = match.groups()
        return f"[[#{_label_to_block_id(label)}]]"

    return _REFERENCE_COMMAND.sub(replacement, text)


def _convert_labels(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        label = match.group(1)
        return f"<!-- latex-label: {label} -->\n{_label_to_block_id(label)}"

    return _LABEL_COMMAND.sub(replacement, text)


def _collect_multiline_command(
    lines: list[str],
    start: int,
    command: str,
) -> tuple[int, str]:
    """Collect a braced command that may continue on subsequent lines."""

    marker = f"\\{command}{{"
    opening_offset = lines[start].find(marker)
    if opening_offset < 0:
        return start, lines[start]
    opening = opening_offset + len(marker) - 1
    end = start
    combined = lines[start]
    while True:
        try:
            _find_braced_end(combined, opening)
            return end, combined
        except ValueError:
            end += 1
            if end >= len(lines):
                raise
            combined += "\n" + lines[end]


def _environment_name(line: str, keyword: str) -> str | None:
    match = re.match(rf"^\s*\\{keyword}\{{([^}}]+)\}}", line)
    return match.group(1) if match else None


def _find_environment_end(lines: list[str], start: int, environment: str) -> int:
    end_pattern = re.compile(rf"^\s*\\end\{{{re.escape(environment)}\}}\s*$")
    for index in range(start + 1, len(lines)):
        if end_pattern.match(lines[index]):
            return index
    raise ValueError(f"Unbalanced LaTeX environment: {environment}")


def _extract_preamble(lines: list[str]) -> tuple[list[str], int]:
    """Separate leading comments/definitions from the document body."""

    preamble: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("%"):
            preamble.append(lines[index])
            index += 1
            continue
        if stripped.startswith("\\label{"):
            break
        if re.match(r"^\\(?:chapter|section|subsection|subsubsection|begin|end)\b", stripped):
            break
        if stripped.startswith("\\"):
            preamble.append(lines[index])
            index += 1
            continue
        break

    while preamble and not preamble[-1].strip():
        preamble.pop()
    return preamble, index


def _render_preamble(preamble: list[str]) -> list[str]:
    return [
        "<!-- latex-preamble",
        *preamble,
        "latex-preamble-end -->",
        "",
    ]


def _render_equation_environment(environment: str, block: list[str]) -> list[str]:
    labels: list[str] = []
    body: list[str] = []

    def remove_label(match: re.Match[str]) -> str:
        labels.append(match.group(1))
        return ""

    for line in block[1:-1]:
        clean_line = _LABEL_COMMAND.sub(remove_label, line)
        if clean_line.strip() or not line.strip():
            body.append(clean_line.rstrip() if clean_line.strip() else clean_line)

    rendered = [
        f"<!-- latex-env: {environment} -->",
        "$$",
        *body,
        "$$",
    ]
    for label in labels:
        rendered.extend([_label_to_block_id(label), f"<!-- latex-label: {label} -->"])
    if labels:
        rendered.append("")
    rendered.extend(["<!-- /latex-env -->", ""])
    return rendered


def _render_protected_environment(
    environment: str,
    block: list[str],
    bibliography: Mapping[str, BibEntry] | None = None,
) -> list[str]:
    if environment in EQUATION_ENVIRONMENTS:
        return _render_equation_environment(environment, block)

    labels = _LABEL_COMMAND.findall("\n".join(block))
    citation_sources = [
        match.group(0)
        for match in _CITATION_COMMAND.finditer("\n".join(block))
    ]
    rendered = [
        f"<!-- latex-env: {environment} -->",
        "```latex",
        *block,
        "```",
    ]
    for label in labels:
        rendered.append(_label_to_block_id(label))
    if labels:
        rendered.append("")
    if citation_sources:
        rendered.append("<!-- latex-raw-citations -->")
        rendered.extend(
            f"- {_convert_citations(source, bibliography)}"
            for source in citation_sources
        )
        rendered.extend(["<!-- /latex-raw-citations -->", ""])
    rendered.extend(["<!-- /latex-env -->", ""])
    return rendered


def _convert_body(
    spec: SectionSpec,
    latex: str,
    bibliography: Mapping[str, BibEntry] | None = None,
) -> str:
    lines = latex.splitlines()
    output: list[str] = []
    if not spec.contains_own_chapter:
        output.extend([f"# {spec.chapter_title}", ""])

    preamble, body_start = _extract_preamble(lines)
    if preamble:
        output.extend(_render_preamble(preamble))
    lines = lines[body_start:]

    list_mode: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        environment = _environment_name(line, "begin")
        if environment in PROTECTED_ENVIRONMENTS:
            end = _find_environment_end(lines, index, environment)
            output.extend(
                _render_protected_environment(
                    environment,
                    lines[index : end + 1],
                    bibliography,
                )
            )
            index = end + 1
            continue
        if _environment_name(line, "end") in PROTECTED_ENVIRONMENTS:
            raise ValueError(f"Unexpected LaTeX environment end: {line.strip()}")

        if re.match(r"^\s*\\begin\{(itemize|enumerate)\}\s*$", line):
            list_mode = re.search(r"\\begin\{(itemize|enumerate)\}", line).group(1)
            index += 1
            continue
        if re.match(r"^\s*\\end\{(itemize|enumerate)\}\s*$", line):
            list_mode = None
            output.append("")
            index += 1
            continue

        heading = re.match(
            r"^\s*\\(chapter|section|subsection|subsubsection)\*?\{(.*)\}\s*$",
            line,
        )
        if heading:
            level = {
                "chapter": 1,
                "section": 2,
                "subsection": 3,
                "subsubsection": 4,
            }[heading.group(1)]
            output.append(
                f"{'#' * level} {_convert_inline(heading.group(2), bibliography)}"
            )
            index += 1
            continue

        item = re.match(r"^\s*\\item\s*(.*)$", line)
        if item and list_mode:
            marker = "-" if list_mode == "itemize" else "1."
            output.append(
                f"{marker} {_convert_inline(item.group(1), bibliography)}".rstrip()
            )
            index += 1
            continue

        if "\\pendiente{" in line:
            pending_end, pending_line = _collect_multiline_command(
                lines,
                index,
                "pendiente",
            )
            converted = _convert_pending(
                _convert_labels(pending_line),
                bibliography,
            )
            index = pending_end + 1
        else:
            converted = _convert_labels(_convert_inline(line, bibliography))
            index += 1
        output.extend(converted.strip("\n").splitlines() or [""])

    while output and output[-1] == "":
        output.pop()
    return "\n".join(output) + "\n"


def convert_latex_to_markdown(
    spec: SectionSpec,
    latex: str,
    source_hash: str,
    bibliography: Mapping[str, BibEntry] | None = None,
) -> str:
    """Convert one active LaTeX module to an interactive, reversible note."""

    source_path = spec.source_tex.as_posix()
    frontmatter = "\n".join(
        [
            "---",
            "kind: thesis_section",
            f"source_tex: {json.dumps(source_path, ensure_ascii=False)}",
            f"target_tex: {json.dumps(spec.source_tex.name, ensure_ascii=False)}",
            f"chapter_title: {json.dumps(spec.chapter_title, ensure_ascii=False)}",
            f"contains_own_chapter: {'true' if spec.contains_own_chapter else 'false'}",
            "sync_direction: reviewed_markdown_to_staged_latex",
            f"source_sha256: {source_hash}",
            "status: imported",
            "---",
            "",
        ],
    )
    return frontmatter + _convert_body(spec, latex, bibliography)
