"""Kaggle notebook for EXP-0011 (RAD on the Heat RBF control). Never launches anything."""
import base64, hashlib, json, textwrap
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
S2 = REPO / "var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2"
files = {"heat_rad_runner.py": HERE / "heat_rad_runner.py", "af_sciml_model.py": S2 / "payload/af_sciml_model.py", "rad.py": REPO / "src/mechanisms/rad.py"}
for s in range(901, 906):
    files[f"heat-seed-{s}.npz"] = S2 / f"pools/heat-seed-{s}.npz"
blob = {k: base64.b64encode(v.read_bytes()).decode() for k, v in files.items()}
hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}
code = textwrap.dedent(r'''
    import base64, json, os, pathlib, subprocess, sys, time, torch
    T0 = time.monotonic(); CAP = 5400.0; CUT = 4300.0
    w = pathlib.Path("/kaggle/working"); p = w / "payload"; p.mkdir(exist_ok=True)
    BLOB = @@BLOB@@
    for k, v in BLOB.items(): (p / k).write_bytes(base64.b64decode(v))
    n = torch.cuda.device_count(); names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(json.dumps({"gpus": names, "hashes": @@HASH@@}), flush=True)
    assert n >= 1 and all("T4" in x for x in names)
    # 5 RAD jobs first (the pre-registered runs), then 5 control replicates (refresh 0, no L-BFGS) as a parity check of this runner vs EXP-0008
    queue = [("rad", s, ["--refresh", "1000", "--lbfgs-seconds", "700"]) for s in range(901, 906)] + \
            [("ctl", s, ["--refresh", "0", "--lbfgs-seconds", "0"]) for s in range(901, 906)]
    running, done = {}, []
    while queue or running:
        for g in list(running):
            pr, tag, st = running[g]
            if pr.poll() is not None: done.append({"job": tag, "rc": pr.returncode, "wall": time.monotonic() - st}); del running[g]
        if time.monotonic() - T0 > CUT: queue = []
        if time.monotonic() - T0 > CAP:
            for g, (pr, tag, st) in running.items(): pr.kill()
            done.append({"note": "cap reached, killed", "running": [t for _, t, _ in running.values()]}); break
        for g in range(min(n, 2)):
            if g not in running and queue:
                kind, s, extra = queue.pop(0); tag = f"{kind}-{s}"; env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                cmd = [sys.executable, str(p / "heat_rad_runner.py"), str(s), "--dir", str(p), "--out", str(w / "results" / kind), "--device", "cuda:0", *extra]
                running[g] = (subprocess.Popen(cmd, stdout=open(w / f"{tag}.log", "w"), stderr=subprocess.STDOUT, env=env), tag, time.monotonic())
        time.sleep(3)
    summ = {"done": done, "wall": time.monotonic() - T0, "cap": CAP, "scientific_accepted": False, "reference_accepted": False}
    (w / "session_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))
''').replace("@@BLOB@@", repr(blob)).replace("@@HASH@@", repr(hashes))
out = HERE / "kaggle"; out.mkdir(exist_ok=True)
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(out / "notebook.ipynb").write_text(json.dumps(nb))
(out / "kernel-metadata.json").write_text(json.dumps({"id": "rodrimandub/af-pikans-heat-rad-exp0011-20261004", "title": "AF PIKANs Heat RAD EXP0011 20261004", "code_file": "notebook.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
(out / "approval.json").write_text(json.dumps({"approved_by": "researcher: EXP-0011 signed 2026-10-03 (chat: 'firmo EXP-0011'); GPU quota 30 h renewed", "preregistration": "experiments/heat/EXP-0011-heat-rad-adam/spec.md",
    "session_cap_seconds": 5400, "scientific_accepted": False, "reference_accepted": False, "component_sha256": hashes,
    "note": "5 control replicates (refresh 0, no L-BFGS) added after the 5 pre-registered RAD runs as a runner-parity check; they do not change the pre-registered criterion"}, indent=2))
print("built", len(json.dumps(nb)) // 1024, "KB")
