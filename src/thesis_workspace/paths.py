"""Resolución de la raíz del repositorio de escritura.

Desde la separación del 2026-09-22 el código vive en `AF_PIKANs/` y la
escritura (LaTeX, figuras, plantilla, bibliografía) vive en un repositorio
privado aparte. Este módulo es el único lugar que sabe dónde está.

Orden de resolución:

1. la variable de entorno `THESIS_ROOT`, si apunta a un directorio válido;
2. `../ProyectoGrad` relativo a la raíz de este repositorio;
3. nada.

Devolver `None` en vez de fallar es deliberado: el repositorio de código debe
poder clonarse, testearse y correr en CI sin el repositorio privado de
escritura. Las pruebas que necesitan los `.tex` se saltan, no se rompen.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIBLING = "ProyectoGrad"
ENV_VAR = "THESIS_ROOT"

# Archivo centinela: si no está, no es la raíz de la escritura.
SENTINEL = "z-main.tex"


def thesis_root() -> Path | None:
    """Devuelve la raíz del repositorio de escritura, o None si no está."""
    override = os.environ.get(ENV_VAR)
    if override:
        candidate = Path(override).expanduser().resolve()
        return candidate if _is_thesis_root(candidate) else None

    sibling = (CODE_ROOT.parent / DEFAULT_SIBLING).resolve()
    return sibling if _is_thesis_root(sibling) else None


def require_thesis_root() -> Path:
    """Como `thesis_root`, pero falla con un mensaje accionable."""
    root = thesis_root()
    if root is None:
        raise FileNotFoundError(
            "No se encontró el repositorio de escritura. "
            f"Se buscó ${ENV_VAR} y '{(CODE_ROOT.parent / DEFAULT_SIBLING)}'. "
            f"Debe contener '{SENTINEL}'."
        )
    return root


def _is_thesis_root(candidate: Path) -> bool:
    return candidate.is_dir() and (candidate / SENTINEL).is_file()
