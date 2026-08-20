"""Carga y validación de contratos declarativos de research automation."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from .workflows import validate_workflow_registry


CONTRACT_FILES = {
    "ontology": "context/research-ontology.toml",
    "notebook": "context/notebook-contract.toml",
    "agent": "context/agent-policy.toml",
}


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_contracts(root: Path) -> dict[str, dict[str, Any]]:
    """Carga los tres contratos TOML desde la raíz del repositorio."""

    return {name: _load_toml(root / relative) for name, relative in CONTRACT_FILES.items()}


def _unique_ids(items: list[dict[str, Any]], label: str) -> list[str]:
    ids = [str(item.get("id", "")) for item in items]
    duplicates = [item_id for item_id, count in Counter(ids).items() if item_id and count > 1]
    return [f"{label} duplicado: '{item_id}'" for item_id in sorted(duplicates)]


def validate_contracts(root: Path) -> dict[str, list[str]]:
    """Valida estructura, referencias y vocabularios de los contratos."""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        contracts = load_contracts(root)
    except (FileNotFoundError, tomllib.TOMLDecodeError) as exc:
        return {"errors": [f"No se pudieron cargar los contratos: {exc}"], "warnings": []}

    ontology = contracts["ontology"]
    notebook = contracts["notebook"]
    agent = contracts["agent"]

    if ontology.get("meta", {}).get("schema_version") != 1:
        errors.append("research-ontology.toml debe declarar meta.schema_version = 1")
    if ontology.get("meta", {}).get("namespace") != "pikan-research":
        errors.append("research-ontology.toml debe usar el namespace 'pikan-research'")

    node_types = ontology.get("node_type", [])
    relation_types = ontology.get("relation_type", [])
    errors.extend(_unique_ids(node_types, "node_type"))
    errors.extend(_unique_ids(relation_types, "relation_type"))
    node_ids = {str(item.get("id")) for item in node_types}
    required_node_ids = {
        "paper",
        "concept",
        "domain",
        "frontier",
        "problem",
        "model",
        "hypothesis",
        "experiment",
        "run",
        "metric",
        "artifact",
        "evidence",
        "decision",
        "notebook",
    }
    for missing in sorted(required_node_ids - node_ids):
        errors.append(f"node_type requerido ausente: '{missing}'")

    for vocab_name, spec in ontology.get("vocab", {}).items():
        values = spec.get("values", [])
        if not values:
            errors.append(f"vocabulario vacio: '{vocab_name}'")
        if len(values) != len(set(values)):
            errors.append(f"vocabulario duplicado: '{vocab_name}'")

    paths = ontology.get("paths", {})
    for key in ("vault", "notebooks", "experiment_manifest", "generated_output"):
        relative = paths.get(key)
        if not relative:
            errors.append(f"research-ontology.toml no declara paths.{key}")
        elif key != "generated_output" and not (root / relative).exists():
            errors.append(f"ruta declarada inexistente en paths.{key}: '{relative}'")

    notebook_meta = notebook.get("metadata", {})
    allowed_roles = set(notebook_meta.get("allowed_roles", []))
    role_items = notebook.get("role", [])
    errors.extend(_unique_ids(role_items, "notebook role"))
    declared_roles = {str(item.get("id")) for item in role_items}
    if allowed_roles != declared_roles:
        errors.append("notebook-contract.toml debe declarar exactamente sus roles permitidos")
    if notebook.get("meta", {}).get("ontology") != ontology.get("meta", {}).get("namespace"):
        errors.append("notebook-contract.toml referencia una ontologia distinta")

    agent_roles = agent.get("role", [])
    errors.extend(_unique_ids(agent_roles, "agent role"))
    for role in agent_roles:
        if not role.get("approval_required"):
            errors.append(f"agent role sin approval_required: '{role.get('id')}'")
    if agent.get("meta", {}).get("autonomy_level") != "L2_human_verified_ai_execution":
        warnings.append("La politica de agentes usa un nivel distinto de L2")

    workflow_validation = validate_workflow_registry(root, agent)
    errors.extend(workflow_validation["errors"])
    warnings.extend(workflow_validation["warnings"])

    return {"errors": errors, "warnings": warnings}


def _role_by_id(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("id")): item for item in contract.get("role", [])}


def validate_notebook_metadata(metadata: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    """Devuelve errores de metadatos sin cambiar el notebook."""

    errors: list[str] = []
    metadata_spec = contract.get("metadata", {})
    required = list(metadata_spec.get("required_for_all", []))
    role = metadata.get("role")
    for field in required:
        if not metadata.get(field):
            errors.append(f"metadata.research.{field} es requerido")

    allowed_roles = set(metadata_spec.get("allowed_roles", []))
    if role and role not in allowed_roles:
        errors.append(f"role invalido: '{role}'")

    role_spec = _role_by_id(contract).get(str(role))
    if role_spec:
        for field in role_spec.get("required_metadata", []):
            if not metadata.get(field):
                errors.append(f"metadata.research.{field} es requerido para role '{role}'")

    source_of_truth = metadata.get("source_of_truth")
    allowed_sources = set(metadata_spec.get("allowed_source_of_truth", []))
    if source_of_truth and source_of_truth not in allowed_sources:
        errors.append(f"source_of_truth invalido: '{source_of_truth}'")
    return errors


def audit_notebooks(notebooks_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Audita notebooks existentes sin ejecutar ni escribir sus celdas."""

    metadata_key = str(contract.get("meta", {}).get("metadata_key", "research"))
    required_all = list(contract.get("metadata", {}).get("required_for_all", []))
    items: list[dict[str, Any]] = []

    if not notebooks_root.exists():
        return {"count": 0, "items": [], "status_counts": {}}

    for path in sorted(notebooks_root.rglob("*.ipynb")):
        relative = path.relative_to(notebooks_root).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            items.append(
                {
                    "path": relative,
                    "notebook_id": f"notebook:{relative}",
                    "role": None,
                    "missing_fields": [],
                    "errors": [f"JSON invalido: {exc}"],
                    "status": "invalid",
                },
            )
            continue

        metadata = payload.get("metadata", {}).get(metadata_key, {})
        if not isinstance(metadata, dict):
            metadata = {}
        missing_fields = [field for field in required_all if not metadata.get(field)]
        errors = validate_notebook_metadata(metadata, contract) if metadata else []

        if not metadata:
            status = "unclassified"
        elif errors:
            status = "invalid"
        else:
            status = "ready"

        items.append(
            {
                "path": relative,
                "notebook_id": metadata.get("notebook_id", f"notebook:{relative}"),
                "role": metadata.get("role"),
                "metadata": metadata,
                "missing_fields": missing_fields,
                "errors": errors,
                "status": status,
            },
        )

    return {
        "count": len(items),
        "items": items,
        "status_counts": dict(Counter(item["status"] for item in items)),
    }
