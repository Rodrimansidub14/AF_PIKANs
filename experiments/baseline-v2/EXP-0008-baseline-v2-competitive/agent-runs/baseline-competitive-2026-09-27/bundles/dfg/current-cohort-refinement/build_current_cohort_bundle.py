from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = (
    HERE.parents[2]
    / "kaggle-outputs"
    / "dfg-r4-error-fetch-20260928"
    / "pikan-c349281ede3181b8"
    / "bundle"
    / "src"
    / "research_pipeline"
    / "sciml"
)
MODULE_FILES = {
    "src/research_pipeline/sciml/dfg.py": SOURCE_ROOT / "dfg.py",
    "src/research_pipeline/sciml/dfg_campaign.py": SOURCE_ROOT / "dfg_campaign.py",
}
RUNTIME_TEMPLATE = HERE / "notebook_runtime.py"
WORKER = HERE / "dfg_rbf_lbfgs_worker.py"
NOTEBOOK = HERE / "dfg-rbf-current-cohort-refinement.ipynb"
METADATA = HERE / "kernel-metadata.json"
SLUG = "rodrimandub/af-pikans-dfg-rbf-refinement-20260928"
DATASET_ID = "rodrimandub/af-pikans-dfg-rbf-inputs-20260928"
DATASET_ARCHIVE = "dfg-rbf-cohort-inputs.zip"
SOURCE_KERNELS = [
    "rodrimandub/af-pikans-dfg-re20-baseline-20260927-r3",
    "rodrimandub/af-pikans-dfg-re20-baseline-20260927-r4",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def main() -> None:
    module_payloads = {}
    module_hashes = {}
    for relative_path, source_path in MODULE_FILES.items():
        payload = source_path.read_bytes()
        module_payloads[relative_path] = base64.b64encode(payload).decode("ascii")
        module_hashes[relative_path] = sha256_bytes(payload)

    runtime = RUNTIME_TEMPLATE.read_text(encoding="utf-8")
    runtime = runtime.replace("__EMBEDDED_MODULES__", repr(module_payloads))
    runtime = runtime.replace("__SOURCE_MODULE_HASHES__", repr(module_hashes))
    runtime = runtime.replace("__WORKER_BASE64__", base64.b64encode(WORKER.read_bytes()).decode("ascii"))

    markdown = """# DFG Re=20: L-BFGS continuation for the current RBF cohort

This private, internet-disabled T4 kernel consumes the private input dataset `rodrimandub/af-pikans-dfg-rbf-inputs-20260928`, where Kaggle has expanded the source files. It requires exactly one dataset mount, then validates the exact routes `rodrimandub/af-pikans-dfg-re20-baseline-20260927-r3` and `rodrimandub/af-pikans-dfg-re20-baseline-20260927-r4` directly inside that mount. Seeds 301–302 must come from the r3 route and seeds 303–305 from the r4 route; the loader checks each per-seed checkpoint and pool, the `run.json` and `provenance.json` seed ids, and the recorded pool SHA-256 before starting.

Each seed receives up to 540 seconds of deterministic full-pool L-BFGS. Two seeds run at once on the two Tesla T4 devices; the five seeds use waves 301/302, 303/304, then 305. The closure preserves the r4 Navier–Stokes residual and inlet, no-slip wall/cylinder, and natural outlet traction losses. Adam is not rerun. The training loss never uses a FEATFLOW error.

Before and after scores report FEATFLOW drag, lift, and front-minus-rear pressure at 512 and 1,024 cylinder points, plus held-out residual regions. The runtime writes source hashes, device assignments, elapsed times, optimizer states, and per-seed outputs to `/kaggle/working/dfg-current-cohort-refinement/`. `field_relative_l2` remains null and `scientific_accepted` remains false because no converged FEM field is packaged. This is an RBF-only cohort; it has no MLP arm.

The notebook has a 3,500-second work deadline, a 6,800-second watchdog, and a 7,200-second declared session cap. It requires exactly two visible Tesla T4 GPUs and fails if either device or either source route is missing or ambiguous.
"""
    notebook_payload = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": source_lines(markdown)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(runtime)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook_payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    metadata = {
        "id": SLUG,
        "title": "AF PIKANs DFG RBF Refinement 20260928",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    source_hash_file = HERE / "SOURCE_MODULES.sha256"
    source_hash_file.write_text(
        "".join(f"{digest}  {relative_path}\n" for relative_path, digest in sorted(module_hashes.items())),
        encoding="utf-8",
    )

    files_for_manifest = sorted(
        path for path in HERE.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum_text = "".join(f"{sha256_bytes(path.read_bytes())}  {path.name}\n" for path in files_for_manifest)
    (HERE / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")
    print(f"Generated {NOTEBOOK.name} ({NOTEBOOK.stat().st_size} bytes) and {METADATA.name}")


if __name__ == "__main__":
    main()
