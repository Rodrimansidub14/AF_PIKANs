"""Construcción de artefactos derivados para el grafo de investigación."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import audit_notebooks, load_contracts, validate_contracts
from .vault import collect_vault, resolve_links


def _notebook_nodes(audit: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for item in audit.get("items", []):
        relative = str(item["path"])
        nodes.append(
            {
                "id": f"notebook:{relative}",
                "kind": "notebook",
                "type": "notebook",
                "path": f"notebooks/{relative}",
                "title": Path(relative).stem,
                "notebook_id": item.get("notebook_id"),
                "role": item.get("role"),
                "status": item.get("status"),
                "missing_fields": item.get("missing_fields", []),
            },
        )
    return nodes


def _manifest_node(root: Path, ontology: dict[str, Any]) -> dict[str, Any] | None:
    relative = ontology.get("paths", {}).get("experiment_manifest")
    if not relative:
        return None
    path = root / relative
    if not path.exists():
        return None
    return {
        "id": f"experiment:{relative}",
        "kind": "manifest",
        "type": "experiment",
        "path": relative,
        "title": path.stem,
    }


def build_graph(root: Path) -> dict[str, Any]:
    """Construye un grafo sin modificar el vault ni los notebooks."""

    contracts = load_contracts(root)
    ontology = contracts["ontology"]
    validation = validate_contracts(root)
    vault_root = root / ontology["paths"]["vault"]
    notebooks_root = root / ontology["paths"]["notebooks"]

    vault_nodes = collect_vault(vault_root)
    link_graph = resolve_links(vault_nodes)
    notebook_audit = audit_notebooks(notebooks_root, contracts["notebook"])
    nodes = list(vault_nodes) + _notebook_nodes(notebook_audit)
    manifest_node = _manifest_node(root, ontology)
    if manifest_node:
        nodes.append(manifest_node)

    node_ids = {str(node["id"]) for node in nodes}
    connected: set[str] = set()
    edge_endpoints: set[str] = set()
    for edge in link_graph["edges"]:
        connected.add(str(edge["source"]))
        connected.add(str(edge["target"]))
        edge_endpoints.update((str(edge["source"]), str(edge["target"])))
    orphan_nodes = [
        {"id": str(node["id"]), "path": str(node.get("path", ""))}
        for node in nodes
        if str(node["id"]) not in connected
    ]

    return {
        "schema_version": 1,
        "ontology": ontology.get("meta", {}).get("namespace"),
        "source_roots": {
            "vault": ontology["paths"]["vault"],
            "notebooks": ontology["paths"]["notebooks"],
            "experiment_manifest": ontology["paths"]["experiment_manifest"],
        },
        "nodes": nodes,
        "edges": link_graph["edges"],
        "diagnostics": {
            "missing_links": link_graph["missing_links"],
            "ambiguous_links": link_graph["ambiguous_links"],
            "orphan_nodes": orphan_nodes,
            "unknown_node_ids": sorted(edge_endpoints - node_ids),
            "validation": validation,
        },
        "notebook_audit": notebook_audit,
    }


def render_report(graph: dict[str, Any], contracts: dict[str, Any]) -> str:
    """Renderiza un reporte estable y legible del grafo generado."""

    nodes = graph.get("nodes", [])
    type_counts = Counter(str(node.get("type", "unclassified")) for node in nodes)
    diagnostics = graph.get("diagnostics", {})
    validation = diagnostics.get("validation", {})
    notebook_audit = graph.get("notebook_audit", {})
    lines = [
        "# Graph Engineering Report",
        "",
        "## Resumen",
        "",
        f"- Nodos: {len(nodes)}",
        f"- Relaciones explícitas: {len(graph.get('edges', []))}",
        f"- Enlaces faltantes: {len(diagnostics.get('missing_links', []))}",
        f"- Enlaces ambiguos: {len(diagnostics.get('ambiguous_links', []))}",
        f"- Nodos huérfanos: {len(diagnostics.get('orphan_nodes', []))}",
        "",
        "## Nodos por tipo",
        "",
    ]
    for node_type, count in sorted(type_counts.items()):
        lines.append(f"- `{node_type}`: {count}")

    lines.extend(
        [
            "",
            "## Validación de contratos",
            "",
            f"- Errores: {len(validation.get('errors', []))}",
            f"- Advertencias: {len(validation.get('warnings', []))}",
            "",
            "## Auditoría de notebooks",
            "",
            f"- Total: {notebook_audit.get('count', 0)}",
        ],
    )
    for status, count in sorted(notebook_audit.get("status_counts", {}).items()):
        lines.append(f"- `{status}`: {count}")

    lines.extend(["", "## Política de agentes", ""])
    for role in contracts.get("agent", {}).get("role", []):
        lines.append(f"- `{role.get('id')}`: {role.get('purpose', '')}")

    lines.extend(["", "## Enlaces faltantes", ""])
    missing = diagnostics.get("missing_links", [])
    if missing:
        lines.extend(f"- `{item['source']}` → `{item['target']}`" for item in missing[:50])
        if len(missing) > 50:
            lines.append(f"- ... y {len(missing) - 50} más")
    else:
        lines.append("- Ninguno.")

    lines.extend(["", "## Enlaces ambiguos", ""])
    ambiguous = diagnostics.get("ambiguous_links", [])
    if ambiguous:
        for item in ambiguous[:50]:
            candidates = ", ".join(f"`{candidate}`" for candidate in item["candidates"])
            lines.append(f"- `{item['source']}` → `{item['target']}`: {candidates}")
    else:
        lines.append("- Ninguno.")

    lines.extend(["", "## Nodos huérfanos", ""])
    orphans = diagnostics.get("orphan_nodes", [])
    if orphans:
        lines.extend(f"- `{item['id']}` — `{item['path']}`" for item in orphans[:100])
        if len(orphans) > 100:
            lines.append(f"- ... y {len(orphans) - 100} más")
    else:
        lines.append("- Ninguno.")

    return "\n".join(lines) + "\n"
