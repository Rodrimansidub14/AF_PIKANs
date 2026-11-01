"""Create the reviewed, self-contained EXP-0003 Kaggle notebook bundle."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.research_pipeline.remote.bundle import prepare, verify_transport  # noqa: E402


EXPERIMENT = ROOT / "experiments/EXP-0003-allen-cahn-rbf-sobol"
WORK = ROOT / "var/agent_runs/allen-cahn-exp003"
MANIFEST_SOURCE = EXPERIMENT / "manifest.json"
MANIFEST_FOR_BUNDLE = WORK / "manifest-bundle.json"
OUTPUT = WORK / "bundle-v1"
NOTEBOOK_COPY = ROOT / "notebooks/experiments/allen-cahn-rbf-sobol-exploratory.ipynb"

FILES = [
    "src/research_pipeline/sciml/__init__.py",
    "src/research_pipeline/sciml/allen_cahn_evaluation.py",
    "src/research_pipeline/sciml/campaign.py",
    "src/research_pipeline/sciml/model.py",
    "src/research_pipeline/sciml/sampling.py",
    "src/research_pipeline/sciml/train.py",
    "src/research_pipeline/remote/__init__.py",
    "src/research_pipeline/remote/reporting.py",
    "evals/__init__.py",
    "evals/graders/__init__.py",
    "evals/graders/core.py",
    "evals/graders/exploratory.py",
]


def main() -> None:
    manifest = json.loads(MANIFEST_SOURCE.read_text(encoding="utf-8"))
    decision_path = ROOT / manifest["parameter_decisions_file"]
    manifest["parameter_decisions_markdown"] = decision_path.read_text(encoding="utf-8")
    MANIFEST_FOR_BUNDLE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    spec = prepare(
        MANIFEST_FOR_BUNDLE,
        OUTPUT,
        kernel="rodrigomansilla/af-pikans-allen-cahn-exp003",
        files=FILES,
        timeout=manifest["remote"]["run_timeout_sec"],
        evidence_kind="research",
        gpu=True,
    )
    verified = verify_transport(OUTPUT)
    if verified["bundle_id"] != spec["bundle_id"]:
        raise RuntimeError("Bundle hash changed during transport verification")

    NOTEBOOK_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUT / "experiment.ipynb", NOTEBOOK_COPY)
    request = f"""# Aprobación pendiente: EXP-0003 Allen–Cahn

Este archivo describe la revisión requerida; **no es una aprobación** y no
autoriza por sí mismo ningún despacho. El bundle fue verificado localmente y
su identificador es `{spec['bundle_id']}`.

Revisar el notebook versionado en `notebooks/experiments/allen-cahn-rbf-sobol-exploratory.ipynb`
y las decisiones de `experiments/EXP-0003-allen-cahn-rbf-sobol/parameter-decisions.md`.
La corrida está limitada a tres semillas, dos T4 en paralelo, 20.000 pasos o
1.200 s de entrenamiento por semilla. W&B queda en modo offline. El error de
campo y cualquier aceptación científica siguen bloqueados.

Bundle transport path: `{OUTPUT.relative_to(ROOT).as_posix()}`
Notebook SHA-256: `{verified['files']['experiment.ipynb'] if 'experiment.ipynb' in verified['files'] else 'see transport-hashes.json'}`

Para autorizar el envío remoto, crear `approval.json` **fuera del bundle** con
el hash exacto después de revisar alcance, hipótesis y superficie de código:

```json
{{
  "bundle_id": "{spec['bundle_id']}",
  "approved_by": "<nombre del investigador>",
  "scope": "approved",
  "hypothesis": "approved",
  "code_surface": "approved",
  "remote_execution": true
}}
```

Luego el usuario puede autorizar la llamada remota usando el CLI del repositorio.
No se incluyeron credenciales en el bundle ni se leyó `.env`.
"""
    (WORK / "approval-request.md").write_text(request, encoding="utf-8")
    print(json.dumps({
        "bundle_id": spec["bundle_id"],
        "bundle": str(OUTPUT),
        "notebook": str(NOTEBOOK_COPY),
        "approval_request": str(WORK / "approval-request.md"),
        "source_count": len(spec["files"]),
        "transport_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
