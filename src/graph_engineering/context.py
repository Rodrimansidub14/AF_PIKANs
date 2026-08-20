"""Proyección declarativa del contexto que consume el adaptador de Senpi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import CONTRACT_FILES
from .workflows import project_workflow_registry


def _role_projection(role: dict[str, Any]) -> dict[str, Any]:
    """Conserva únicamente campos declarativos seguros para el contexto."""

    return {
        key: role.get(key, [])
        for key in (
            "id",
            "purpose",
            "allowed_actions",
            "read_paths",
            "write_paths",
            "approval_required",
        )
    }


def _gate_projection(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: gate.get(key, [])
        for key in ("id", "description", "required_before", "decision_owner")
    }


def build_context_pack(
    root: Path,
    graph: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Construye un paquete relativo y estable para la extensión de Senpi.

    ``root`` se recibe para mantener una interfaz explícita de proyecto, pero
    no se serializa: el paquete debe ser portable y no filtrar rutas locales.
    """

    ontology = contracts["ontology"]
    notebook = contracts["notebook"]
    agent = contracts["agent"]
    diagnostics = graph.get("diagnostics", {})
    notebook_audit = graph.get("notebook_audit", {})

    return {
        "schema_version": 1,
        "ontology": ontology.get("meta", {}).get("namespace"),
        "autonomy_level": agent.get("meta", {}).get("autonomy_level"),
        "human_authority": agent.get("meta", {}).get("human_authority"),
        "contracts": dict(CONTRACT_FILES),
        "paths": dict(ontology.get("paths", {})),
        "notebook": {
            "metadata_key": notebook.get("meta", {}).get("metadata_key", "research"),
            "allowed_roles": list(notebook.get("metadata", {}).get("allowed_roles", [])),
            "allowed_source_of_truth": list(
                notebook.get("metadata", {}).get("allowed_source_of_truth", []),
            ),
        },
        "roles": [_role_projection(role) for role in agent.get("role", [])],
        "gates": [_gate_projection(gate) for gate in agent.get("gate", [])],
        "graph": {
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            "missing_links": len(diagnostics.get("missing_links", [])),
            "ambiguous_links": len(diagnostics.get("ambiguous_links", [])),
            "orphan_nodes": len(diagnostics.get("orphan_nodes", [])),
            "validation_errors": len(diagnostics.get("validation", {}).get("errors", [])),
            "validation_warnings": len(diagnostics.get("validation", {}).get("warnings", [])),
        },
        "notebooks": {
            "count": notebook_audit.get("count", 0),
            "status_counts": dict(notebook_audit.get("status_counts", {})),
        },
        "workflow": project_workflow_registry(root, agent),
    }
