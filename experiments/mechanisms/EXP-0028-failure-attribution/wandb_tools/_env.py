"""Lectura minima del .env del repo. La clave solo se devuelve en memoria."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def read_env(path=None):
    out = {}
    p = Path(path) if path else REPO / ".env"
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def public_cfg(env):
    """Solo valores no secretos."""
    return {
        "entity": env.get("WANDB_ENTITY"),
        "project": env.get("WANDB_PROJECT") or "AF_PIKANS",
    }
