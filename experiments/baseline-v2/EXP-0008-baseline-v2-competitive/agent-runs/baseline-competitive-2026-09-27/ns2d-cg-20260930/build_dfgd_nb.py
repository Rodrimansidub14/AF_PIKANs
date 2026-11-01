"""Assemble the Kaggle notebook for the DFG campaign under Daniel's thesis configuration. Never launches anything."""
import base64
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = {"pikan_campaign.py": HERE / "pikan_campaign.py", "dfg_daniel.py": HERE / "dfg_daniel.py",
         "dfg_L3_fields.npz": HERE / "reference" / "dfg_L3_fields.npz"}
blob = {k: base64.b64encode(v.read_bytes()).decode() for k, v in files.items()}
hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}
code = textwrap.dedent(r'''
    import base64, json, os, pathlib, signal, subprocess, sys, time, torch
    T0 = time.monotonic(); CAP = 3600.0; START_CUTOFF = 2700.0
    w = pathlib.Path("/kaggle/working"); p = w / "payload"; p.mkdir(exist_ok=True)
    BLOB = @@BLOB@@
    for k, v in BLOB.items():
        (p / k).write_bytes(base64.b64decode(v))
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(json.dumps({"gpus": names, "torch": torch.__version__, "hashes": @@HASH@@}), flush=True)
    assert names and all("T4" in x for x in names), names
    cmd = [sys.executable, str(p / "dfg_daniel.py"), "campaign", "--fem", str(p / "dfg_L3_fields.npz"),
           "--out", str(w / "results"), "--wall-cap", str(START_CUTOFF), "--width", "16",
           "--seeds", "801", "802", "803", "804", "805"]
    proc = subprocess.Popen(cmd, stdout=open(w / "runner.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    while proc.poll() is None:
        if time.monotonic() - T0 > CAP:
            timed_out = True; os.killpg(proc.pid, signal.SIGTERM); time.sleep(10)
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            break
        time.sleep(3)
    summ = {"status": "timeout_stopped" if timed_out else "complete", "rc": proc.wait(), "wall": time.monotonic() - T0,
            "cap": CAP, "gpus": names, "scientific_accepted": False, "reference_accepted": False}
    (w / "session_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))
''').replace("@@BLOB@@", repr(blob)).replace("@@HASH@@", repr(hashes))
out = HERE / "dfgd"
out.mkdir(exist_ok=True)
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(out / "notebook.ipynb").write_text(json.dumps(nb))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "rodrimandub/af-pikans-dfg-daniel-cfg-20260930", "title": "AF PIKANs DFG Daniel Cfg 20260930",
    "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook", "is_private": "true",
    "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
    "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
(out / "approval.json").write_text(json.dumps({
    "approved_by": "researcher, chat 2026-09-30: 'usa tambien la configuracion de daniel para dfg' + standing 12 h approval, >= 6 h reserve",
    "session_cap_seconds": 3600, "scientific_accepted": False, "reference_accepted": False, "component_sha256": hashes}, indent=2))
print("built", len(json.dumps(nb)) // 1024, "KB", hashes)
