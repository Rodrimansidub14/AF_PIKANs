"""EXP-0018, prueba 7 en T4: la rama B del ejecutor reproduce bit a bit el L-BFGS de la etapa A del EXP-0016 (k = 1.00) desde el mismo estado final de Adam.

Dos trabajos como los de produccion (un proceso por semilla, con su calentamiento, una GPU cada uno, en paralelo): Burgers 301 y Allen-Cahn 601, solo la rama B.
Se compara el hash del estado final (y el error global) con los registrados en la etapa A (`stageA_reference_hashes.json`). Criterio fijado antes de correr: los dos hashes iguales.
La rama B no usa A* ni el foco: es el control. No puntua ninguna hipotesis.

Variables de entorno (solo para probar en CPU): PRECHECK_TINY=1 (red minima; se compara B contra B en dos procesos), BUNDLE_DIR, OUT_DIR.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TINY = os.environ.get("PRECHECK_TINY") == "1"
bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("run_ceiling.py")).parent)
out = Path(os.environ.get("OUT_DIR", "/kaggle/working/precheck18"))
out.mkdir(parents=True, exist_ok=True)
ref = json.loads((bundle / "stageA_reference_hashes.json").read_text())

JOBS = [("burgers", 301, "burgers_coleHopf_1025x129.npz", 48, 20000, 600), ("ac", 601, "Allen_Cahn.mat", 16, 40000, 300)]


def command(job, tag):
    problem, seed, refname, width, adam, cap = job
    cmd = [sys.executable, str(bundle / "run_ceiling.py"), "--problem", problem, "--ref", str(bundle / refname), "--campaign-py", str(bundle / "pikan_campaign.py"),
           "--seeds", str(seed), "--out", str(out / tag), "--state-dir", str(bundle), "--width", str(width), "--adam-steps", str(adam if not TINY else 40),
           "--lbfgs-seconds", str(cap), "--arms", "B", "--device", "cpu" if TINY else "cuda:0"]
    if TINY:
        cmd += ["--lbfgs-max-iter", "40", "--lbfgs-chunk", "10", "--warmup-steps", "10"] + (["--n-col", "256"] if problem == "burgers" else [])
    return cmd


def launch(job, tag, slot):
    env = dict(os.environ, BUNDLE_DIR=str(bundle))
    if TINY:
        env["OMP_NUM_THREADS"] = "1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
    (out / tag).mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(command(job, tag), env=env, stdout=open(out / f"{tag}.log", "w"), stderr=subprocess.STDOUT, cwd=str(bundle))


t0 = time.time()
tags = [(job, f"{job[0]}-{job[1]}") for job in JOBS]
if TINY:   # humo: el mismo trabajo dos veces; B debe coincidir consigo mismo
    tags = tags + [(job, f"{job[0]}-{job[1]}-again") for job in JOBS]
procs = [launch(job, tag, i % 2) for i, (job, tag) in enumerate(tags)]
rcs = [p.wait() for p in procs]
verdict = {}
for (job, tag), rc in zip(tags, rcs):
    problem, seed = job[0], job[1]
    f = out / tag / f"{problem}-seed{seed}.json"
    res = json.loads(f.read_text()) if f.exists() else None
    b = res["arms"]["B"] if res else None
    entry = dict(returncode=rc, b_state_sha256=b and b["state_sha256"], b_global_l2=b and b["global_l2"], b_seconds=b and b["lbfgs_seconds"], b_iterations=b and b["lbfgs_iterations"])
    if not TINY:
        r = ref[f"{problem}-{seed}"]
        entry.update(stageA_state_sha256=r["state_sha256"], stageA_global_l2=r["global_l2"], bit_identical=bool(b and b["state_sha256"] == r["state_sha256"]),
                     error_identical=bool(b and b["global_l2"] == r["global_l2"]))
    verdict[tag] = entry
if TINY:
    verdict["identical_twice"] = all(verdict[f"{j[0]}-{j[1]}"]["b_state_sha256"] == verdict[f"{j[0]}-{j[1]}-again"]["b_state_sha256"] for j in JOBS)
    passed = verdict["identical_twice"]
else:
    passed = all(v["bit_identical"] and v["returncode"] == 0 for v in verdict.values())
summary = dict(passed=bool(passed), wall_seconds=time.time() - t0, tiny=TINY, verdict=verdict)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1), flush=True)
