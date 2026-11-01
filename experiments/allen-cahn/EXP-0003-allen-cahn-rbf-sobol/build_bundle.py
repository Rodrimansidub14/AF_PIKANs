"""Build the reviewable, self-contained EXP-0003 Kaggle notebook bundle.

This script creates only a local bundle and approval request. It never reads
credentials, writes Kaggle state, or submits a kernel.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.research_pipeline.remote.bundle import prepare, verify_transport  # noqa: E402


EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_WORK = ROOT / "var/agent_runs/allen-cahn-exp003"
FILES = [
    "src/sciml/__init__.py",
    "src/sciml/allen_cahn_evaluation.py",
    "src/sciml/campaign.py",
    "src/sciml/model.py",
    "src/sciml/sampling.py",
    "src/sciml/train.py",
    "src/research_pipeline/remote/__init__.py",
    "src/research_pipeline/remote/reporting.py",
    "evals/__init__.py",
    "evals/graders/__init__.py",
    "evals/graders/core.py",
    "evals/graders/exploratory.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_WORK / "bundle-final"))
    parser.add_argument("--kernel", default="rodrigomansilla/af-pikans-allen-cahn-exp003")
    args = parser.parse_args()

    source_manifest = json.loads((EXPERIMENT / "manifest.json").read_text(encoding="utf-8"))
    decisions_file = ROOT / source_manifest["parameter_decisions_file"]
    source_manifest["parameter_decisions_markdown"] = decisions_file.read_text(encoding="utf-8")
    work = DEFAULT_WORK
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest-bundle.json"
    manifest_path.write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    output = Path(args.out).resolve()
    spec = prepare(
        manifest_path,
        output,
        kernel=args.kernel,
        files=FILES,
        timeout=source_manifest["remote"]["run_timeout_sec"],
        evidence_kind="research",
        gpu=True,
    )
    verified = verify_transport(output)
    if verified["bundle_id"] != spec["bundle_id"]:
        raise RuntimeError("Bundle hash changed during transport verification")

    notebook = ROOT / "notebooks/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/allen-cahn-rbf-sobol-exploratory.ipynb"
    notebook.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "experiment.ipynb", notebook)
    transport_hashes = json.loads((output / "transport-hashes.json").read_text(encoding="utf-8"))
    approval_request = work / "approval-request.md"
    approval_request.write_text(f"""# Aprobación pendiente: EXP-0003 Allen–Cahn

Este archivo describe la revisión requerida; **no es una aprobación** ni
autoriza por sí mismo un despacho. Bundle verificado localmente:
`{spec['bundle_id']}`.

Revisar `notebooks/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/allen-cahn-rbf-sobol-exploratory.ipynb` y
`experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/parameter-decisions.md`. La corrida
está limitada a tres semillas, dos T4 en paralelo, 20.000 pasos o 1.200 s de
entrenamiento por semilla. W&B queda offline. Error de campo y aceptación
científica continúan bloqueados.

Bundle: `{output.relative_to(ROOT).as_posix()}`  
Notebook SHA-256: `{transport_hashes['experiment.ipynb']}`

Si decides autorizar el despacho remoto de este hash, el registro debe incluir
estos campos y guardarse fuera del bundle como `approval.json`:

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

No se incluyeron credenciales en el bundle ni se leyó `.env`.
""", encoding="utf-8")

    print(json.dumps({
        "bundle_id": spec["bundle_id"],
        "bundle": str(output),
        "notebook": str(notebook),
        "notebook_sha256": transport_hashes["experiment.ipynb"],
        "approval_request": str(approval_request),
        "source_count": len(spec["files"]),
        "transport_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
