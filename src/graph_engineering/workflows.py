"""Carga, valida y proyecta el registro declarativo de workflows."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


WORKFLOW_REGISTRY = Path("workflow/index.toml")
REQUIRED_META = {"schema_version", "default", "ambiguity_policy"}
REQUIRED_ITEM_FIELDS = {
    "id",
    "role",
    "priority",
    "triggers",
    "excludes",
    "stages",
    "gate_ids",
}
SAFE_ITEM_FIELDS = (
    "id",
    "role",
    "priority",
    "triggers",
    "excludes",
    "stages",
    "gate_ids",
)


def load_workflow_registry(root: Path) -> dict[str, Any]:
    """Carga el registro TOML desde una ruta relativa al proyecto."""

    path = root / WORKFLOW_REGISTRY
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _as_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        return None
    return [item.strip() for item in value]


def validate_workflow_registry(
    root: Path,
    agent_contract: dict[str, Any],
) -> dict[str, list[str]]:
    """Valida metadatos, referencias y vocabulario del registro."""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        registry = load_workflow_registry(root)
    except (FileNotFoundError, tomllib.TOMLDecodeError) as exc:
        return {"errors": [f"No se pudo cargar workflow/index.toml: {exc}"], "warnings": []}

    meta = registry.get("meta", {})
    missing_meta = REQUIRED_META - set(meta)
    errors.extend(f"workflow meta requerido ausente: '{field}'" for field in sorted(missing_meta))
    if meta.get("schema_version") != 1:
        errors.append("workflow/index.toml debe declarar meta.schema_version = 1")
    if meta.get("ambiguity_policy") != "conservative":
        errors.append("workflow/index.toml debe usar ambiguity_policy = 'conservative'")

    items = registry.get("workflow", [])
    if not isinstance(items, list) or not items:
        errors.append("workflow/index.toml debe declarar al menos un [[workflow]]")
        items = []

    role_ids = {str(item.get("id")) for item in agent_contract.get("role", [])}
    gate_ids = {str(item.get("id")) for item in agent_contract.get("gate", [])}
    workflow_ids: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            errors.append("cada [[workflow]] debe ser una tabla")
            continue
        item_id = str(item.get("id", ""))
        workflow_ids.append(item_id)
        for field in sorted(REQUIRED_ITEM_FIELDS - set(item)):
            errors.append(f"workflow '{item_id}' requiere '{field}'")
        if not item_id:
            errors.append("workflow sin id")
        if str(item.get("role", "")) not in role_ids:
            errors.append(f"workflow '{item_id}' referencia rol inexistente: '{item.get('role')}'")
        if not isinstance(item.get("priority"), int):
            errors.append(f"workflow '{item_id}' debe tener priority entero")
        for field in ("triggers", "excludes", "stages", "gate_ids"):
            values = _as_string_list(item.get(field))
            if values is None:
                errors.append(f"workflow '{item_id}' debe declarar {field} como lista de strings no vacía")
                continue
            if field == "triggers" and len(values) != len({value.casefold() for value in values}):
                errors.append(f"workflow '{item_id}' tiene triggers duplicados")
            if field == "gate_ids":
                for gate_id in values:
                    if gate_id not in gate_ids:
                        errors.append(f"workflow '{item_id}' referencia gate inexistente: '{gate_id}'")

    duplicates = sorted({item_id for item_id in workflow_ids if workflow_ids.count(item_id) > 1 and item_id})
    errors.extend(f"workflow duplicado: '{item_id}'" for item_id in duplicates)
    default = str(meta.get("default", ""))
    if default not in workflow_ids:
        errors.append(f"workflow default inexistente: '{default}'")

    return {"errors": errors, "warnings": warnings}


def project_workflow_registry(
    root: Path,
    agent_contract: dict[str, Any],
) -> dict[str, Any]:
    """Proyecta únicamente el contrato seguro que consume la extensión."""

    validation = validate_workflow_registry(root, agent_contract)
    if validation["errors"]:
        raise ValueError("; ".join(validation["errors"]))
    registry = load_workflow_registry(root)
    meta = registry["meta"]
    items = [
        {field: item[field] for field in SAFE_ITEM_FIELDS}
        for item in registry["workflow"]
    ]
    return {
        "schema_version": meta["schema_version"],
        "default": meta["default"],
        "ambiguity_policy": meta["ambiguity_policy"],
        "items": items,
    }
