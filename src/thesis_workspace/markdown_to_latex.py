"""Inverse conversion for the exact Markdown constructs emitted by bootstrap."""

from __future__ import annotations

import re


_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_RAW_MARKER = re.compile(r"^\s*<!-- latex-env: ([^>]+?) -->\s*$")
_CLOSING_MARKER = re.compile(r"^\s*<!-- /latex-env -->\s*$")
_RAW_CITATIONS_START = re.compile(r"^\s*<!-- latex-raw-citations -->\s*$")
_RAW_CITATIONS_END = re.compile(r"^\s*<!-- /latex-raw-citations -->\s*$")
_PENDING_MARKER = re.compile(r"^\s*<!-- latex-pendiente: (.*?) -->\s*$")
_PREAMBLE_START = re.compile(r"^\s*<!-- latex-preamble\s*$")
_PREAMBLE_END = re.compile(r"^\s*latex-preamble-end -->\s*$")
_AGENT_PROMPT_START = re.compile(r"^\s*<!-- latex-agent-prompt\s*$")
_AGENT_PROMPT_END = re.compile(r"^\s*latex-agent-prompt-end -->\s*$")
_LABEL_MARKER = re.compile(r"^\s*<!--\s*latex-label:\s*([^>]+?)\s*-->\s*$")
_BLOCK_ID = re.compile(r"^\s*\^([A-Za-z][A-Za-z0-9_-]*)\s*$")
_WIKILINK_REFERENCE = re.compile(
    r"\[\[(?P<target>[^#\]|]*)#\^(?P<block>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?:\|[^\]]+)?\]\]"
)
_CITATION_SOURCE = r"\\cite[a-zA-Z]*(?:\[[^\]\r\n]*\])*\{[^{}]+\}"
_CITATION_LINK_PATTERN = (
    r"\[\[99_Bibliografia#\^cite-[A-Za-z][A-Za-z0-9_-]*"
    r"(?:\|[^\]]+)?\]\]"
)
_CITATION_MARKER = re.compile(
    rf"<!--\s*latex-citation:\s*(?P<source>{_CITATION_SOURCE})\s*-->"
    rf"(?:\s*{_CITATION_LINK_PATTERN}"
    rf"(?:\s*,\s*{_CITATION_LINK_PATTERN})*)?"
)
_CITATION_LINK = re.compile(
    r"\[\[99_Bibliografia#\^cite-(?P<key>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?:\|[^\]]+)?\]\]"
)
_TODO_CALLOUT = re.compile(
    r"> \[!todo\] Pendiente\r?\n"
    r"(?:> [^\r\n]*(?:\r?\n|$))*"
    r"<!-- latex-pendiente: (.*?) -->",
    re.DOTALL,
)
_EQUATION_ENVS = {"equation", "equation*"}
_FENCE = chr(96) * 3


def _parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(markdown)
    if not match:
        return {}, markdown
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"')
    return values, markdown[match.end() :]


def _restore_labels(text: str) -> str:
    return re.sub(
        r"<!--\s*latex-label:\s*([^>]+?)\s*-->",
        r"\\label{\1}",
        text,
    )


def _restore_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    return re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\\textit{\1}", text)


def _block_id_to_label(block_id: str) -> str | None:
    prefix, separator, suffix = block_id.partition("-")
    if not separator or prefix not in {"eq", "tab", "fig", "sec", "subsec", "subsubsec"}:
        return None
    return f"{prefix}:{suffix}"


def _restore_references(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        block_id = match.group("block")
        label = _block_id_to_label(block_id)
        if label is None:
            return match.group(0)
        command = "eqref" if block_id.startswith("eq-") else "ref"
        return f"\\{command}{{{label}}}"

    return _WIKILINK_REFERENCE.sub(replacement, text)


def _restore_citations(text: str) -> str:
    text = _CITATION_MARKER.sub(
        lambda match: match.group("source"),
        text,
    )
    return _CITATION_LINK.sub(
        lambda match: f"\\cite{{{match.group('key')}}}",
        text,
    )


def _restore_line(text: str) -> str:
    text = _restore_inline(text)
    text = _restore_labels(text)
    text = _restore_references(text)
    return _restore_citations(text)


def _restore_preamble(
    lines: list[str],
    start: int,
) -> tuple[int, list[str]]:
    index = start + 1
    body: list[str] = []
    while index < len(lines) and not _PREAMBLE_END.match(lines[index]):
        body.append(lines[index])
        index += 1
    if index >= len(lines):
        raise ValueError("Unbalanced LaTeX preamble block")
    return index + 1, body


def _skip_agent_prompt(lines: list[str], start: int) -> int:
    index = start + 1
    while index < len(lines) and not _AGENT_PROMPT_END.match(lines[index]):
        index += 1
    if index >= len(lines):
        raise ValueError("Unbalanced LaTeX agent prompt block")
    return index + 1


def _restore_protected(
    lines: list[str],
    start: int,
    environment: str,
) -> tuple[int, list[str]]:
    index = start + 1
    if index >= len(lines):
        raise ValueError(f"Incomplete protected block: {environment}")

    if environment in _EQUATION_ENVS:
        if lines[index].strip() != "$$":
            raise ValueError(f"Expected equation fence for {environment}")
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index].strip() != "$$":
            body.append(_restore_labels(lines[index]))
            index += 1
        if index >= len(lines):
            raise ValueError(f"Unbalanced equation block: {environment}")
        index += 1
        block_ids: list[str] = []
        labels: list[str] = []
        while index < len(lines) and not _CLOSING_MARKER.match(lines[index]):
            block_match = _BLOCK_ID.match(lines[index])
            label_match = _LABEL_MARKER.match(lines[index])
            if block_match:
                block_ids.append(block_match.group(1))
            elif label_match:
                labels.append(label_match.group(1).strip())
            elif lines[index].strip():
                raise ValueError(f"Unexpected equation metadata: {lines[index].strip()}")
            index += 1
        if index >= len(lines):
            raise ValueError(f"Missing protected-block terminator: {environment}")
        for label in labels:
            label_line = f"\\label{{{label}}}"
            if not any(existing.strip() == label_line for existing in body):
                body.append(label_line)
        if not labels:
            for block_id in block_ids:
                label = _block_id_to_label(block_id)
                if label is not None:
                    label_line = f"\\label{{{label}}}"
                    if not any(existing.strip() == label_line for existing in body):
                        body.append(label_line)
        return index + 1, [
            f"\\begin{{{environment}}}",
            *body,
            f"\\end{{{environment}}}",
        ]

    if lines[index].strip() != _FENCE + "latex":
        raise ValueError(f"Expected raw-LaTeX fence for {environment}")
    index += 1
    body = []
    while index < len(lines) and lines[index].strip() != _FENCE:
        body.append(lines[index])
        index += 1
    if index >= len(lines):
        raise ValueError(f"Unbalanced raw-LaTeX block: {environment}")
    index += 1
    while index < len(lines) and not _CLOSING_MARKER.match(lines[index]):
        if _RAW_CITATIONS_START.match(lines[index]):
            index += 1
            while index < len(lines) and not _RAW_CITATIONS_END.match(lines[index]):
                index += 1
            if index >= len(lines):
                raise ValueError("Unbalanced raw citation summary")
            index += 1
            continue
        if not _BLOCK_ID.match(lines[index]) and not _LABEL_MARKER.match(lines[index]) and lines[index].strip():
            raise ValueError(f"Unexpected raw-block metadata: {lines[index].strip()}")
        index += 1
    if index >= len(lines):
        raise ValueError(f"Missing protected-block terminator: {environment}")
    return index + 1, body


def _restore_pending(lines: list[str], start: int) -> tuple[int, list[str]]:
    index = start + 1
    while index < len(lines):
        match = _PENDING_MARKER.match(lines[index])
        if match:
            return index + 1, [match.group(1)]
        index += 1
    raise ValueError("TODO callout is missing its protected LaTeX marker")


def _restore_todo_callouts(body: str) -> str:
    if "> [!todo] Pendiente" not in body:
        return body
    restored, _ = _TODO_CALLOUT.subn(lambda match: match.group(1), body)
    if "> [!todo] Pendiente" in restored:
        raise ValueError("TODO callout is missing its protected LaTeX marker")
    return restored


def _heading_to_latex(
    level: int,
    title: str,
    *,
    contains_own_chapter: bool,
    chapter_title: str,
) -> str | None:
    title = _restore_line(title.strip())
    if level == 1:
        if not contains_own_chapter and title == chapter_title:
            return None
        return f"\\chapter{{{title}}}"
    if level == 2:
        return f"\\section{{{title}}}"
    if level == 3:
        return f"\\subsection{{{title}}}"
    if level == 4:
        return f"\\subsubsection{{{title}}}"
    return f"{'#' * level} {title}"


def convert_markdown_to_latex(markdown: str) -> str:
    """Restore Markdown produced by the thesis bootstrap to LaTeX text."""

    metadata, body = _parse_frontmatter(markdown)
    body = _restore_todo_callouts(body)
    contains_own_chapter = metadata.get("contains_own_chapter", "false").lower() == "true"
    chapter_title = metadata.get("chapter_title", "")
    lines = body.splitlines()
    output: list[str] = []
    list_kind: str | None = None

    def close_list() -> None:
        nonlocal list_kind
        if list_kind is not None:
            output.append(f"\\end{{{list_kind}}}")
            list_kind = None

    index = 0
    while index < len(lines):
        line = lines[index]
        if _AGENT_PROMPT_START.match(line):
            index = _skip_agent_prompt(lines, index)
            continue
        if _PREAMBLE_START.match(line):
            index, restored = _restore_preamble(lines, index)
            output.extend(restored)
            continue
        marker = _RAW_MARKER.match(line)
        if marker:
            close_list()
            index, restored = _restore_protected(lines, index, marker.group(1))
            output.extend(restored)
            continue
        if _BLOCK_ID.match(line):
            index += 1
            continue
        if line.lstrip().startswith("> [!todo] Pendiente"):
            close_list()
            index, restored = _restore_pending(lines, index)
            output.extend(restored)
            continue

        heading = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            close_list()
            restored = _heading_to_latex(
                len(heading.group(1)),
                heading.group(2),
                contains_own_chapter=contains_own_chapter,
                chapter_title=chapter_title,
            )
            if restored is not None:
                output.append(restored)
            index += 1
            continue

        item = re.match(r"^\s*(-|\d+\.)\s+(.*)$", line)
        if item:
            wanted_kind = "itemize" if item.group(1) == "-" else "enumerate"
            if list_kind != wanted_kind:
                close_list()
                output.append(f"\\begin{{{wanted_kind}}}")
                list_kind = wanted_kind
            output.append(f"\\item {_restore_line(item.group(2))}".rstrip())
            index += 1
            continue

        close_list()
        output.append(_restore_line(line))
        index += 1

    close_list()
    while output and output[-1] == "":
        output.pop()
    return "\n".join(output) + ("\n" if output else "")
