"""Sincroniza carpetas offline-run* de W&B hacia entity/project del .env.

Por defecto es --dry-run (solo lista). Con --apply ejecuta `wandb sync`.
La clave WANDB_API_KEY se lee del .env dentro de este proceso y se entrega
solo por el entorno del subproceso: nunca se imprime, registra ni pasa en argv.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import read_env, public_cfg  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(HERE / "offline"),
                    help="ruta bajo la cual buscar offline-run*")
    ap.add_argument("--apply", action="store_true", help="sincroniza de verdad")
    ap.add_argument("--entity")
    ap.add_argument("--project")
    ap.add_argument("--env-file")
    ap.add_argument("--limit", type=int, help="maximo de runs a sincronizar")
    ap.add_argument("--log", default=str(HERE / "sync_log.json"))
    a = ap.parse_args()

    env = read_env(a.env_file)
    cfg = public_cfg(env)
    entity = a.entity or cfg["entity"]
    project = a.project or cfg["project"]
    key = env.get("WANDB_API_KEY")
    runs = sorted(p for p in Path(a.path).rglob("offline-run*") if p.is_dir())
    pending = [r for r in runs if not Path(str(r) + ".synced").exists()
               and not any(r.glob("*.wandb.synced"))]
    if a.limit:
        pending = pending[: a.limit]
    print(f"{len(runs)} carpetas offline-run*, {len(pending)} a sincronizar; "
          f"destino {entity}/{project}; modo {'APPLY' if a.apply else 'dry-run'}")
    if not a.apply:
        for r in pending:
            print("  [dry-run]", r.name)
        return 0
    if not key:
        print("ERROR: no hay WANDB_API_KEY en el .env; no se sincroniza.")
        return 2
    child_env = dict(os.environ)
    child_env["WANDB_API_KEY"] = key
    ok = fail = 0
    entries = []
    for r in pending:
        cmd = [sys.executable, "-m", "wandb", "sync", "--project", project]
        if entity:
            cmd += ["--entity", entity]
        cmd.append(str(r))
        for intento in range(3):  # reintenta solo timeouts de red
            p = subprocess.run(cmd, env=child_env, capture_output=True, text=True)
            if p.returncode == 0 or "TimeoutError" not in ((p.stdout or "") + (p.stderr or "")):
                break
        out = ((p.stdout or "") + (p.stderr or "")).replace(key, "***")
        tail = [l for l in out.splitlines() if l.strip()][-4:]
        good = p.returncode == 0
        ok += good
        fail += not good
        entries.append({"run_dir": r.name, "returncode": p.returncode, "ok": good, "tail": tail})
        print("  OK  " if good else "  FAIL", r.name)
        if not good:
            print("      ", " | ".join(tail)[:300])
            if any(w in out.lower() for w in ("401", "invalid api key", "not logged in", "authentication failed", "permission denied")):
                print("Fallo de autenticacion: se detiene. No se prueban otras vias.")
                break
    log = Path(a.log)
    prev = []
    if log.exists():
        try:
            prev = json.loads(log.read_text(encoding="utf-8")).get("envios", [])
        except Exception:
            prev = []
    prev.append({"fecha": datetime.now(timezone.utc).isoformat(), "entity": entity,
                 "project": project, "ok": ok, "fallidos": fail, "runs": entries})
    log.write_text(json.dumps({"envios": prev}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"subidos OK={ok} fallidos={fail}; registro en {log.name}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
