"""Assemble the Kaggle notebook for the Heat L-BFGS redo. Never launches anything."""
import base64
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
S2 = HERE.parent / "kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2"
files = {"heat_lbfgs_gpu.py": HERE / "heat_lbfgs_gpu.py", "af_sciml_model.py": S2 / "payload/af_sciml_model.py"}
for s in range(901, 906):
    files[f"seed-{s}-adam-final.pt"] = S2 / f"runs/tier2/seed-{s}/rbf/checkpoint-adam-final.pt"
    files[f"heat-seed-{s}.npz"] = S2 / f"pools/heat-seed-{s}.npz"
blob = {k: base64.b64encode(v.read_bytes()).decode() for k, v in files.items()}
hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}
code = textwrap.dedent(r'''
    import base64, json, os, pathlib, subprocess, sys, time, torch
    T0 = time.monotonic(); CAP = 2700.0; PER_SEED = 700.0
    w = pathlib.Path("/kaggle/working"); p = w / "payload"; p.mkdir(exist_ok=True)
    BLOB = @@BLOB@@
    for k, v in BLOB.items():
        (p / k).write_bytes(base64.b64decode(v))
    n = torch.cuda.device_count(); names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(json.dumps({"gpus": names, "hashes": @@HASH@@}), flush=True)
    assert n >= 1 and all("T4" in x for x in names)
    queue = [901, 902, 903, 904, 905]; running = {}; done = []
    while queue or running:
        for g in list(running):
            pr, s, st = running[g]
            if pr.poll() is not None:
                done.append({"seed": s, "rc": pr.returncode, "wall": time.monotonic() - st}); del running[g]
        if time.monotonic() - T0 > CAP:
            queue = []
        for g in range(min(n, 2)):
            if g not in running and queue:
                s = queue.pop(0); env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                lg = open(w / f"seed-{s}.log", "w")
                cmd = [sys.executable, str(p / "heat_lbfgs_gpu.py"), str(s), "--seconds", str(PER_SEED),
                       "--dir", str(p), "--out", str(w / "results")]
                running[g] = (subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env), s, time.monotonic())
        time.sleep(3)
    summ = {"done": done, "wall": time.monotonic() - T0, "cap": CAP, "scientific_accepted": False}
    (w / "session_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))
''').replace("@@BLOB@@", repr(blob)).replace("@@HASH@@", repr(hashes))
out = HERE / "nb"
out.mkdir(exist_ok=True)
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(out / "notebook.ipynb").write_text(json.dumps(nb))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "rodrimandub/af-pikans-heat-lbfgs-redo-20260930", "title": "AF PIKANs Heat LBFGS Redo 20260930",
    "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook", "is_private": "true",
    "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
    "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
(out / "approval.json").write_text(json.dumps({
    "approved_by": "researcher, chat 2026-09-30: 'Adam->L-BFGS ... ok. Repite.'; GPU within the standing 12 h approval, >= 6 h reserve",
    "session_cap_seconds": 2700, "per_seed_lbfgs_seconds": 700, "scientific_accepted": False,
    "component_sha256": hashes}, indent=2))
print("built", len(json.dumps(nb)) // 1024, "KB")
