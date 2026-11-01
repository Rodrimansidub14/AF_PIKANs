"""Lectura y resolución honesta del grafo Markdown de PIKAN-Vault."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


WIKILINK_PATTERN = re.compile(r"\[\[([^\]\n]+)\]\]")
HEADING_PATTERN = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
TAG_PATTERN = re.compile(r"#([^\s#]+)")


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    if value.startswith(("\"", "'")) and value.endswith(value[0]):
        try:
            return json.loads(value) if value[0] == '"' else value[1:-1]
        except json.JSONDecodeError:
            return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parsea el bloque YAML simple usado por las notas del vault.

    El parser deliberadamente soporta solo pares `key: value`, listas simples y
    tags. No intenta convertirse en un parser YAML general.
    """

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if end is None:
        return {}

    result: dict[str, Any] = {}
    for line in lines[1:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        key = key.strip()
        if not key:
            continue
        if key == "tags" and not raw.strip().startswith("["):
            result[key] = TAG_PATTERN.findall(raw)
        else:
            result[key] = _parse_scalar(raw)
    return result


def extract_wikilinks(text: str) -> list[str]:
    """Extrae destinos de wikilinks, quitando alias y encabezados."""

    result: list[str] = []
    seen: set[str] = set()
    for match in WIKILINK_PATTERN.finditer(text):
        target = match.group(1).split("|", 1)[0].split("#", 1)[0].strip()
        if target and target not in seen:
            result.append(target)
            seen.add(target)
    return result


def _first_heading(text: str) -> str | None:
    match = HEADING_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _node_id(relative_path: str) -> str:
    return f"vault:{relative_path.removesuffix('.md')}"


def collect_vault(vault_root: Path) -> list[dict[str, Any]]:
    """Lee notas Markdown del vault, excluyendo plantillas."""

    if not vault_root.exists():
        return []

    nodes: list[dict[str, Any]] = []
    for path in sorted(vault_root.rglob("*.md")):
        relative = path.relative_to(vault_root).as_posix()
        if any(part.startswith("_") for part in Path(relative).parts):
            continue

        text = path.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(text)
        title = str(frontmatter.get("title") or _first_heading(text) or path.stem)
        directory_type = {
            "papers": "paper",
            "concepts": "concept",
            "domains": "domain",
            "frontiers": "frontier",
        }.get(Path(relative).parts[0], "unclassified")
        node_type = str(frontmatter.get("type") or directory_type)
        nodes.append(
            {
                "id": _node_id(relative),
                "kind": "vault_note",
                "type": node_type,
                "path": relative,
                "title": title,
                "tags": frontmatter.get("tags", []),
                "frontmatter": frontmatter,
                "links": extract_wikilinks(text),
            },
        )
    return nodes


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\\", "/").strip()
    normalized = normalized.removesuffix(".md")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.casefold()


def _index_nodes(nodes: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for node in nodes:
        value = node[field]
        key = _normalize(str(value))
        index.setdefault(key, []).append(str(node["id"]))
    return index


def resolve_links(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Resuelve enlaces y conserva los faltantes o ambiguos como diagnósticos."""

    by_path = _index_nodes(nodes, "path")
    by_stem: dict[str, list[str]] = {}
    by_title = _index_nodes(nodes, "title")
    for node in nodes:
        stem_key = _normalize(Path(str(node["path"])).stem)
        by_stem.setdefault(stem_key, []).append(str(node["id"]))

    edges: list[dict[str, str]] = []
    missing_links: list[dict[str, str]] = []
    ambiguous_links: list[dict[str, Any]] = []

    for node in nodes:
        for target in node.get("links", []):
            normalized_target = _normalize(str(target))
            candidates = by_path.get(normalized_target, [])
            if not candidates:
                candidates = by_stem.get(normalized_target, [])
            if not candidates:
                candidates = by_title.get(normalized_target, [])

            if len(candidates) == 1:
                edges.append(
                    {
                        "source": str(node["id"]),
                        "target": candidates[0],
                        "relation": "links_to",
                        "label": str(target),
                    },
                )
            elif not candidates:
                missing_links.append({"source": str(node["id"]), "target": str(target)})
            else:
                ambiguous_links.append(
                    {
                        "source": str(node["id"]),
                        "target": str(target),
                        "candidates": candidates,
                    },
                )

    return {
        "edges": edges,
        "missing_links": missing_links,
        "ambiguous_links": ambiguous_links,
    }
