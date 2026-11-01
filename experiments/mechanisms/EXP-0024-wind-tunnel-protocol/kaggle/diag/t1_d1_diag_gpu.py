"""EXP-0024, diagnostico de la unica falla de la prueba corta T1: Burgers D1 diferia 20.11 % de la CPU al final de Adam (criterio 20 %).

Dos preguntas, ambas con el presupuesto reducido de la prueba (semilla 1, Adam 200 pasos, L-BFGS 50 iteraciones):
 1. ¿Cuanto varia la GPU consigo misma? Burgers D1 en float32 dos veces, cada una en su propio proceso (con calentamiento), una por GPU en paralelo.
 2. ¿La diferencia CPU-GPU es redondeo o error? Burgers D1 en float64 en la GPU (la CPU se corre aparte en la computadora del investigador). Si es redondeo, en float64 la diferencia
    con la CPU desaparece; si es un error del codigo, queda.
No puntua hipotesis. Salidas: un JSON por corrida (t1_trainer) y summary.json con los errores.
Variables de entorno (solo para probar en CPU): DIAG_TINY=1, BUNDLE_DIR, OUT_DIR.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

TINY = os.environ.get("DIAG_TINY") == "1"
bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("t1_trainer.py")).parent)
out = Path(os.environ.get("OUT_DIR", "/kaggle/working/t1_d1_diag"))
out.mkdir(parents=True, exist_ok=True)
ADAM, ITERS = (20, 10) if TINY else (200, 50)
RUNS = {"f32_run1": ("float32", 0), "f32_run2": ("float32", 1), "f64_gpu": ("float64", 0)}


def trainer(tag, dtype, slot):
    env = dict(os.environ, BUNDLE_DIR=str(bundle))
    if TINY:
        env["OMP_NUM_THREADS"] = "2"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
    cmd = [sys.executable, str(bundle / "t1_trainer.py"), "--problem", "burgers", "--designs", "D1", "--seeds", "1", "--adam", str(ADAM), "--lbfgs-iters", str(ITERS),
           "--dtype", dtype, "--out", str(out / tag), "--no-ledger", "--device", "cpu" if TINY else "cuda:0", "--warmup-steps", "10" if TINY else "50"]
    with open(out / f"{tag}.log", "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(bundle)).returncode


rc = {}


def lane(tags):
    for t in tags:
        rc[t] = trainer(t, RUNS[t][0], RUNS[t][1])


t0 = time.time()
lanes = [threading.Thread(target=lane, args=(["f32_run1"],)), threading.Thread(target=lane, args=(["f32_run2", "f64_gpu"],))]
[x.start() for x in lanes]
[x.join() for x in lanes]
res = {}
for tag in RUNS:
    f = out / tag / f"burgers-D1-seed1-a{ADAM}-l{ITERS}.json"
    if f.exists():
        r = json.loads(f.read_text())
        res[tag] = dict(dtype=RUNS[tag][0], primary_adam=r["primary_adam"], primary_final=r["primary_final"], seconds=r["total_seconds"],
                        regrid_fit=[(x["l1_fit"]["fit_rel_error"], x["l2_fit"]["fit_rel_error"]) for x in r["regrid_diag"]])
a, b = res.get("f32_run1"), res.get("f32_run2")
summary = dict(tiny=TINY, returncodes=rc, runs=res, wall_seconds=time.time() - t0,
               gpu_self_diff=None if not (a and b) else dict(adam=abs(a["primary_adam"] - b["primary_adam"]) / b["primary_adam"],
                                                              final=abs(a["primary_final"] - b["primary_final"]) / b["primary_final"],
                                                              bit_identical=bool(a["primary_adam"] == b["primary_adam"] and a["primary_final"] == b["primary_final"])))
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1), flush=True)
