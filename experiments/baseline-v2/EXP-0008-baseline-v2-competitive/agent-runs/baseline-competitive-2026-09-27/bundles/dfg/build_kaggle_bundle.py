"""Offline assembler for the self-contained Kaggle notebook; never launches a runner."""
from __future__ import annotations

import base64
import json
import textwrap
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
RUNNER = BUNDLE / "run_campaign.py"
NOTEBOOK = BUNDLE / "dfg-re20-t4x2.ipynb"
METADATA = BUNDLE / "kernel-metadata.json"


def build_notebook() -> None:
    runner_payload = base64.b64encode(RUNNER.read_bytes()).decode("ascii")
    code = textwrap.dedent(r'''
        # Self-contained Kaggle entry point. No pip, internet, CPU solver or fallback.
        import time
        session_started = time.monotonic()
        import base64, hashlib, json, os, pathlib, signal, subprocess, sys, zipfile
        import numpy as np
        import scipy
        import torch

        SESSION_CAP_SECONDS = 7200.0
        EXECUTION_CAP_SECONDS = 6900.0  # leave 300 seconds for Kaggle setup/closeout
        work = pathlib.Path("/kaggle/working/dfg-re20-t4x2")
        work.mkdir(parents=True, exist_ok=True)
        run_root = work / "campaign"
        log_path = work / "runner.log"
        runner_path = work / "run_campaign.py"
        runner_payload = @@RUNNER_PAYLOAD@@
        runner_bytes = base64.b64decode(runner_payload)
        runner_path.write_bytes(runner_bytes)
        runner_sha256 = hashlib.sha256(runner_bytes).hexdigest()

        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("Expected exactly two visible T4 GPUs; CPU fallback is disabled.")
        gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
        if any("T4" not in name.upper() for name in gpu_names):
            raise RuntimeError(f"Expected two Tesla T4 devices, found {gpu_names!r}.")
        print(json.dumps({"gpus": gpu_names, "torch": torch.__version__, "numpy": np.__version__,
                          "scipy": scipy.__version__, "runner_sha256": runner_sha256,
                          "quota_cap_seconds": SESSION_CAP_SECONDS,
                          "execution_cap_seconds": EXECUTION_CAP_SECONDS}, indent=2))

        timed_out = False
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(runner_path), "--output", str(run_root)],
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            deadline = session_started + EXECUTION_CAP_SECONDS
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(1.0)
            return_code = process.wait()

        elapsed = time.monotonic() - session_started
        summary = {"status": "timeout_stopped" if timed_out else ("complete" if return_code == 0 else "runner_failed"),
                   "runner_return_code": return_code, "runner_sha256": runner_sha256,
                   "visible_gpus": gpu_names, "notebook_cell_wall_seconds": elapsed,
                   "session_quota_cap_seconds": SESSION_CAP_SECONDS,
                   "execution_cap_seconds": EXECUTION_CAP_SECONDS,
                   "output_directory": str(run_root), "runner_log": str(log_path),
                   "featflow_qois": {"Cd": 5.57953523384, "Cl": 0.010618948146, "delta_p": 0.11752016697},
                   "field_relative_l2": None,
                   "field_reference_status": "missing_converged_FEM_field",
                   "claim_limit": "No directly matching published DFG Re=20 PINN SOTA was identified; QoI comparison only."}
        (work / "session_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        archive_path = work / "dfg-artifacts.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file() and path != archive_path:
                    archive.write(path, path.relative_to(work).as_posix())
        print(json.dumps({**summary, "downloadable_archive": str(archive_path)}, indent=2))
    ''').replace("@@RUNNER_PAYLOAD@@", json.dumps(runner_payload))
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# DFG Re=20 paired campaign — Kaggle T4x2\n\n",
            "**DRAFT_UNVERIFIED.** This self-contained notebook embeds `run_campaign.py`; it uses no pip install, internet access, CPU solver/fallback, or external dataset. Running it starts the GPU campaign. The watchdog allows at most 6,900 seconds from the first code cell, leaving 300 seconds within the 7,200-second session quota.\n\n",
            "The RBF-PIKAN and tanh MLP train concurrently on separate T4 processes. FEATFLOW QoIs remain the official anchors (`Cd=5.57953523384`, `Cl=0.010618948146`, `delta_p=0.11752016697`). `field_relative_l2` remains null because a converged FEM field is missing; this is not a published-SOTA PINN comparison. Metrics and logs are zipped under `/kaggle/working/dfg-re20-t4x2/`.\n"
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": code.splitlines(keepends=True)},
    ]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3"}},
                "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def build_metadata() -> None:
    metadata = {
        "id": "rodrimandub/af-pikans-dfg-re20-competitive-pilot-t4x2",
        "title": "AF PIKANs DFG Re20 Competitive Pilot T4x2",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_notebook()
    build_metadata()
    print(f"Wrote {NOTEBOOK.name} and {METADATA.name}; no Kaggle runner was executed.")
