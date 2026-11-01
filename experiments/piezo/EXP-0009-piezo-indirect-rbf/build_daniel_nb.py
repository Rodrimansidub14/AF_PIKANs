"""Kaggle notebook for the Daniel-recipe arms (EXP-0009). usage: build_daniel_nb.py <name> <cap_s> <jobs_json>; jobs=[{"arm","seed","epochs","width","tag"}]"""
import base64, hashlib, io, json, sys, textwrap, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
name, cap, jobs = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])
UP = HERE.parents[2] / "var/agent_runs/piezo-ref-20260930/daniel-ref-20261003"
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("src", "scripts"):
        for f in (UP / sub).rglob("*.py"):
            z.write(f, f.relative_to(UP).as_posix())
    z.write(UP / "pyproject.toml", "pyproject.toml")
files = {"upstream.zip": buf.getvalue(), "daniel_arm.py": (HERE / "daniel_arm.py").read_bytes(), "fem.npz": (HERE / "results/reference/fem-grid-indirect-polingminus.npz").read_bytes()}
blob = {k: base64.b64encode(v).decode() for k, v in files.items()}
hashes = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
code = textwrap.dedent(r'''
    import base64, json, os, pathlib, subprocess, sys, time, zipfile, torch
    T0 = time.monotonic(); CAP = @@CAP@@
    w = pathlib.Path("/kaggle/working"); p = w / "payload"; p.mkdir(exist_ok=True)
    BLOB = @@BLOB@@
    for k, v in BLOB.items(): (p / k).write_bytes(base64.b64decode(v))
    zipfile.ZipFile(p / "upstream.zip").extractall(p / "daniel")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(json.dumps({"gpus": names, "torch": torch.__version__, "hashes": @@HASH@@}), flush=True)
    assert names and all("T4" in x for x in names), names
    JOBS = @@JOBS@@
    running, ledger, queue = {}, [], list(JOBS)
    while queue or running:
        for g in list(running):
            pr, tag, st = running[g]
            if pr.poll() is not None: ledger.append({"job": tag, "rc": pr.returncode, "wall": time.monotonic() - st}); del running[g]
        if time.monotonic() - T0 > CAP:
            for g, (pr, tag, st) in running.items(): pr.kill()
            ledger.append({"note": "cap reached, killed", "running": [t for _, t, _ in running.values()]}); break
        for g in range(len(names)):
            if g not in running and queue:
                j = queue.pop(0); tag = j["tag"]
                cmd = [sys.executable, str(p / "daniel_arm.py"), "--daniel-root", str(p / "daniel"), "--arm", j["arm"], "--seed", str(j["seed"]),
                       "--epochs", str(j["epochs"]), "--width", str(j.get("width", 16)), "--device", "cuda", "--fem", str(p / "fem.npz"), "--out", str(w / "results" / tag)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                running[g] = (subprocess.Popen(cmd, stdout=open(w / f"{tag}.log", "w"), stderr=subprocess.STDOUT, env=env), tag, time.monotonic())
        time.sleep(3)
    summ = {"wall": time.monotonic() - T0, "cap": CAP, "ledger": ledger, "gpus": names, "scientific_accepted": False, "reference_accepted": False}
    (w / "session_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))
''').replace("@@CAP@@", repr(cap)).replace("@@BLOB@@", repr(blob)).replace("@@HASH@@", repr(hashes)).replace("@@JOBS@@", repr(jobs))
out = HERE / "kaggle" / name; out.mkdir(parents=True, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(out / "notebook.ipynb").write_text(json.dumps(nb))
title = "AF PIKANs " + name.replace("-", " ").title().replace("Pikans", "PIKANs")
(out / "kernel-metadata.json").write_text(json.dumps({"id": "rodrimandub/" + "-".join(title.lower().split()), "title": title, "code_file": "notebook.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
(out / "approval.json").write_text(json.dumps({"approved_by": "researcher, chat 2026-10-03: usar la PINN de Daniel como base con RBF; cuota 30 h", "session_cap_seconds": cap,
    "upstream": "Daniel14gonc/PINNs_piezoelectricity e7856f1 (read-only reuse; licence file absent, MIT declared)", "scientific_accepted": False,
    "reference_accepted": False, "component_sha256": hashes, "jobs": jobs}, indent=2))
print("built", out, len(json.dumps(nb)) // 1024, "KB", json.loads((out / "kernel-metadata.json").read_text())["id"])
