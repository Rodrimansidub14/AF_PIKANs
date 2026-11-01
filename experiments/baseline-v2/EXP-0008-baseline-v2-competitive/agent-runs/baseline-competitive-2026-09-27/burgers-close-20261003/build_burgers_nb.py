"""Offline assembler for self-contained Kaggle notebooks (bench / ac / ns). Never launches anything.

Usage: py -3.12 build_notebooks.py <name> [extra campaign args...]
  name = bench | ac | ns ; writes <name>/{notebook.ipynb, kernel-metadata.json, approval.json, SHA256SUMS.txt}
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "pikan_campaign.py"
BU_REF = HERE / "reference" / "burgers_coleHopf_1025x129.npz"
OWNER = "rodrimandub"

SEEDS = ["--seeds", "301", "302", "303", "304", "305"]
SPECS = {
    "burgers": {"slug": "af-pikans-burgers-close-20261003", "title": "AF PIKANs Burgers Close 20261003", "cap": 0.0, "mode": "campaign",
                "args": ["campaign", "--problem", "burgers", "--ref", "@BU", "--out", "@OUT/burgers", "--arms", "mlp", "rbf@16", "rbf@48", *SEEDS,
                         "--adam-steps", "20000", "--lbfgs-seconds", "600", "--wall-cap", "10000"]},
    "burgers64": {"slug": "af-pikans-burgers-f64lbfgs-20261003", "title": "AF PIKANs Burgers F64 LBFGS 20261003", "cap": 0.0, "mode": "campaign",
                  "args": ["campaign", "--problem", "burgers", "--ref", "@BU", "--out", "@OUT/burgers64", "--arms", "mlp", "rbf@48", *SEEDS,
                           "--adam-steps", "20000", "--lbfgs-dtype", "float64", "--lbfgs-seconds", "900", "--wall-cap", "9000"]},
    "burgersbig": {"slug": "af-pikans-burgers-big-20261003", "title": "AF PIKANs Burgers Big 20261003", "cap": 0.0, "mode": "campaign",
                   "args": ["campaign", "--problem", "burgers", "--ref", "@BU", "--out", "@OUT/burgersbig", "--arms", "mlp", "rbf@48", *SEEDS,
                            "--adam-steps", "24000", "--n-col", "16384", "--lbfgs-dtype", "float64", "--lbfgs-seconds", "1800", "--wall-cap", "15000"]},
}


def b64(path: Path, gz: bool = False) -> str:
    data = path.read_bytes()
    return base64.b64encode(gzip.compress(data, 9) if gz else data).decode("ascii")


def main() -> None:
    name = sys.argv[1]
    extra = sys.argv[2:]
    spec = SPECS[name]
    cap = float(spec["cap"])
    cli = list(spec["args"]) + extra
    if "--session-cap" in cli:  # builder-only flag: notebook-level hard cap in seconds
        i = cli.index("--session-cap")
        cap = float(cli[i + 1])
        del cli[i:i + 2]
    assert cap > 0, "a session cap is required (use --session-cap SECONDS)"
    out = HERE / name
    out.mkdir(exist_ok=True)
    code = textwrap.dedent(r'''
        import base64, gzip, hashlib, json, os, pathlib, signal, subprocess, sys, time
        session_started = time.monotonic()
        import torch
        CAP = @@CAP@@
        work = pathlib.Path("/kaggle/working/@@NAME@@"); work.mkdir(parents=True, exist_ok=True)
        payload = {"pikan_campaign.py": (@@RUNNER@@, False), "burgers_coleHopf_1025x129.npz": (@@BUREF@@, True)}
        hashes = {}
        for fname, (blob, gz) in payload.items():
            raw = base64.b64decode(blob)
            raw = gzip.decompress(raw) if gz else raw
            (work / fname).write_bytes(raw)
            hashes[fname] = hashlib.sha256(raw).hexdigest()
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(json.dumps({"gpus": gpus, "torch": torch.__version__, "hashes": hashes, "cap": CAP}, indent=1), flush=True)
        if not gpus or any("T4" not in g.upper() for g in gpus):
            raise RuntimeError(f"Expected T4 GPUs, got {gpus!r}; CPU fallback disabled.")
        subs = {"@BU": str(work / "burgers_coleHopf_1025x129.npz"), "@OUT": str(work / "results")}
        cli = @@CLI@@
        for k, v in subs.items():
            cli = [c.replace(k, v) if c.startswith(k) or k in c else c for c in cli]
        log = (work / "runner.log").open("w")
        proc = subprocess.Popen([sys.executable, str(work / "pikan_campaign.py"), *cli], stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True, cwd=str(work))
        timed_out = False
        while proc.poll() is None:
            if time.monotonic() - session_started > CAP:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(10)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(2)
        rc = proc.wait()
        summary = {"status": "timeout_stopped" if timed_out else ("complete" if rc == 0 else "runner_failed"),
                   "return_code": rc, "wall_seconds": time.monotonic() - session_started, "cap_seconds": CAP,
                   "gpus": gpus, "hashes": hashes, "scientific_accepted": False, "reference_accepted": False}
        (work / "session_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, ))
        print((work / "runner.log").read_text()[-3000:])
    ''')
    code = (code.replace("@@CAP@@", repr(cap)).replace("@@NAME@@", name)
            .replace("@@RUNNER@@", repr(b64(RUNNER))).replace("@@BUREF@@", repr(b64(BU_REF, True))).replace("@@CLI@@", repr(cli)))
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "notebook.ipynb").write_text(json.dumps(nb, ))
    meta = {"id": f"{OWNER}/{spec['slug']}", "title": spec["title"], "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
            "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (RUNNER, out / "notebook.ipynb", out / "kernel-metadata.json")}
    (out / "SHA256SUMS.txt").write_text("\n".join(f"{v}  {k}" for k, v in sums.items()) + "\n")
    approval = {"bundle": name, "slug": spec["slug"], "cli": cli, "session_cap_seconds": cap,
                "approved_by": "researcher, chat 2026-10-03: cierra el baseline; quota renewed (30 h)",
                "scope": "approved", "hypothesis": "exploratory_unsigned_preregistration", "code_surface": "var/agent_runs only",
                "remote_execution": True, "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False,
                "scientific_accepted": False, "reference_accepted": False, "component_sha256": sums}
    (out / "approval.json").write_text(json.dumps(approval, indent=2))
    print(json.dumps({"name": name, "cap": cap, "cli": cli, "sha256": sums}, ))


if __name__ == "__main__":
    main()
