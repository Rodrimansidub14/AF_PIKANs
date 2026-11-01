"""Adjunta en LOCAL los npz de la rebanada (EXP-0028) como wandb.Artifact a sus corridas ya sincronizadas. En Kaggle (sin internet) cada corrida registra su artefacto solo como REFERENCIA (URI + digest);
los bytes viajan en la salida del kernel. Despues de `wandb sync` (wandb_tools/sync_wandb.py), este script reanuda cada run por id (= etiqueta `<problema>-<brazo>-seed<semilla>`) y sube el npz como artefacto.
Por defecto es DRY-RUN (lista). --apply sube. NO lee .env ni WANDB_API_KEY: usa la sesion de W&B del entorno (la misma con que sincronizo el investigador); si no hay sesion, falla sin intentar otra via.
Uso: python attach_artifacts_local.py DIR_DE_SALIDA [--apply] [--entity man22611-uvg] [--project AF_PIKANS]"""
import argparse
import re
import sys
from pathlib import Path

NAME = re.compile(r"(?P<tag>(?P<p>[a-z0-9]+)-(?P<a>[A-Z]+)-seed(?P<s>\d+))-artifacts\.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--entity", default="man22611-uvg")
    ap.add_argument("--project", default="AF_PIKANS")
    a = ap.parse_args()
    files = sorted(f for f in Path(a.path).rglob("*-artifacts.npz") if NAME.fullmatch(f.name))
    print(f"{len(files)} npz; destino {a.entity}/{a.project}; modo {'APPLY' if a.apply else 'dry-run'}")
    if not a.apply:
        for f in files:
            print("  [dry-run]", f.name, f"{f.stat().st_size / 2 ** 20:.1f} MB")
        return 0
    import wandb
    ok = fail = 0
    for f in files:
        m = NAME.fullmatch(f.name)
        try:
            run = wandb.init(entity=a.entity, project=a.project, id=m["tag"].replace("/", "_"), resume="allow", job_type=m["p"], settings=wandb.Settings(console="off", silent=True))
            art = wandb.Artifact(f"{m['tag']}-artifacts-file", type="run-state", metadata=dict(file=f.name, problem=m["p"], arm=m["a"], seed=int(m["s"])))
            art.add_file(str(f))
            run.log_artifact(art)
            run.finish()
            ok += 1
            print("  OK  ", f.name)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print("  FAIL", f.name, type(e).__name__, str(e)[:200])
            if "auth" in str(e).lower() or "api key" in str(e).lower():
                print("Sin sesion de W&B: se detiene.")
                break
    print(f"adjuntados {ok}, fallidos {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
