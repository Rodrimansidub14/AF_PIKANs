"""Assemble a self-contained Kaggle notebook for EXP-0009 (piezo). Never launches anything.
usage: py -3.13 build_piezo_nb.py <name> <cap_s> <start_cutoff_s> <jobs_json>  ; jobs = [{"arm","dtype","seed","adam","lbfgs_s","width"}...]"""
import base64, hashlib, json, sys, textwrap
from pathlib import Path
HERE = Path(__file__).resolve().parent
name, cap, cutoff, jobs = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), json.loads(sys.argv[4])
files = {"piezo_pinn.py": HERE / "piezo_pinn.py", "piezo_layered.py": HERE / "piezo_layered.py", "fem.npz": HERE / "results/reference/fem-grid-indirect-polingminus.npz"}
blob = {k: base64.b64encode(v.read_bytes()).decode() for k, v in files.items()}
hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}
code = textwrap.dedent(r'''
    import base64, json, os, pathlib, signal, subprocess, sys, time, torch
    T0 = time.monotonic(); CAP = @@CAP@@; CUT = @@CUT@@
    w = pathlib.Path("/kaggle/working"); p = w / "payload"; p.mkdir(exist_ok=True)
    BLOB = @@BLOB@@
    for k, v in BLOB.items(): (p / k).write_bytes(base64.b64decode(v))
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(json.dumps({"gpus": names, "torch": torch.__version__, "hashes": @@HASH@@}), flush=True)
    assert names and all("T4" in x for x in names), names
    JOBS = @@JOBS@@
    running, ledger, queue = {}, [], list(JOBS)
    while queue or running:
        for g in list(running):
            pr, tag, st = running[g]
            if pr.poll() is not None:
                ledger.append({"job": tag, "rc": pr.returncode, "wall": time.monotonic() - st}); del running[g]
        if time.monotonic() - T0 > CUT: queue = []
        if time.monotonic() - T0 > CAP:
            for g, (pr, tag, st) in running.items(): pr.kill()
            ledger.append({"note": "cap reached, killed", "running": [t for _, t, _ in running.values()]}); break
        for g in range(len(names)):
            if g not in running and queue:
                j = queue.pop(0); tag = j.get('tag') or f"{j['arm']}-{j['dtype']}-vs{j.get('vsplit', 0.0)}-seed-{j['seed']}"
                if j.get("script") == "layered":
                    cmd = [sys.executable, str(p / "piezo_layered.py"), "run", "--arm", j["arm"], "--dtype", j["dtype"], "--seed", str(j["seed"]),
                           "--width", str(j.get("width", 16)), "--device", f"cuda:{g}", "--fem", str(p / "fem.npz"), "--out", str(w / "results" / tag),
                           "--vsplit", str(j.get("vsplit", 0.0)), "--balance", str(j.get("balance", 0)), "--lbfgs-weights", j.get("lbfgs_weights", "unit"),
                           "--adam-steps", str(j["adam"]), "--lbfgs-seconds", str(j["lbfgs_s"])]
                else:
                    cmd = [sys.executable, str(p / "piezo_pinn.py"), "run", "--arm", j["arm"], "--dtype", j["dtype"], "--seed", str(j["seed"]),
                           "--width", str(j.get("width", 16)), "--device", f"cuda:{g}", "--fem", str(p / "fem.npz"), "--out", str(w / "results" / tag),
                           "--vsplit", str(j.get("vsplit", 0.0)), "--lr", str(j.get("lr", 1e-3)), "--balance", str(j.get("balance", 0)), "--lbfgs-weights", j.get("lbfgs_weights", "frozen"), "--w-tau", str(j.get("w_tau", 1.0)), "--w-sz", str(j.get("w_sz", 1.0)), "--adam-steps", str(j["adam"]), "--lbfgs-seconds", str(j["lbfgs_s"])]

                running[g] = (subprocess.Popen(cmd, stdout=open(w / f"{tag}.log", "w"), stderr=subprocess.STDOUT), tag, time.monotonic())
        time.sleep(2)
    summ = {"wall": time.monotonic() - T0, "cap": CAP, "ledger": ledger, "gpus": names, "scientific_accepted": False, "reference_accepted": False}
    (w / "session_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))
''').replace("@@CAP@@", repr(cap)).replace("@@CUT@@", repr(cutoff)).replace("@@BLOB@@", repr(blob)).replace("@@HASH@@", repr(hashes)).replace("@@JOBS@@", repr(jobs))
out = HERE / "kaggle" / name; out.mkdir(parents=True, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(out / "notebook.ipynb").write_text(json.dumps(nb))
(out / "kernel-metadata.json").write_text(json.dumps({"id": f"rodrimandub/af-pikans-{name}", "title": f"AF PIKANs {name}", "code_file": "notebook.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
(out / "approval.json").write_text(json.dumps({"approved_by": "researcher, chat 2026-10-01: 'podes gastarlo, se reinicia manana'", "session_cap_seconds": cap,
    "scientific_accepted": False, "reference_accepted": False, "component_sha256": hashes, "jobs": jobs}, indent=2))
print("built", out, len(json.dumps(nb)) // 1024, "KB")
