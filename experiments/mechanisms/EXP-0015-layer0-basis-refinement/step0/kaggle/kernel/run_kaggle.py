"""EXP-0015 paso 0, tarea 1: semillas 2 y 3 de DFG, Heat y Burgers en CPU de Kaggle.

Misma receta que la semilla 1: --adam 2000 --lbfgs 2000 --ntrain 8000, float64.
Un trabajo = (problema, brazo, semilla); un hilo por trabajo, un trabajo por nucleo.
"""
import os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = next(Path("/kaggle/input").rglob("fit_step0.py")).parent  # la ruta de montaje varia
print("dataset en", SRC, flush=True)
AF = Path("/kaggle/working/AF")
OUT = Path("/kaggle/working/results")
BLR = AF / "var/agent_runs/baseline-competitive-2026-09-27"
(BLR / "burgers-close-20261003/reference").mkdir(parents=True, exist_ok=True)
(BLR / "ns2d-cg-20260930/reference").mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
# fit_step0.py evalua parents[4] aunque AF_ROOT este definido: se coloca a la misma profundidad que en el repo
SCRIPT = AF / "experiments/mechanisms/EXP-0015-layer0-basis-refinement/step0/fit_step0.py"
SCRIPT.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(SRC / "fit_step0.py", SCRIPT)
shutil.copy(SRC / "burgers_coleHopf_1025x129.npz", BLR / "burgers-close-20261003/reference/")
shutil.copy(SRC / "dfg_L3_fields.npz", BLR / "ns2d-cg-20260930/reference/")

# Burgers primero (los mas largos), luego DFG y Heat.
jobs = [("burgers", a, s) for s in (2, 3) for a in ("C", "O11", "U41", "X11")]
jobs += [(p, a, s) for p in ("dfg", "heat") for s in (2, 3) for a in ("C", "O11", "U41")]

env = dict(os.environ, AF_ROOT=str(AF), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
workers = os.cpu_count() or 4
print(f"{len(jobs)} trabajos, {workers} nucleos", flush=True)


def run(job):
    p, a, s = job
    t0 = time.time()
    r = subprocess.run([sys.executable, str(SCRIPT), "--problem", p, "--arms", a,
                        "--seeds", str(s), "--adam", "2000", "--lbfgs", "2000", "--ntrain", "8000",
                        "--out", str(OUT)], env=env, capture_output=True, text=True)
    print(f"[{time.time()-t0:6.0f}s rc={r.returncode}] {p} {a} seed{s} | {r.stdout.strip()[-200:]} {r.stderr.strip()[-300:]}",
          flush=True)
    return r.returncode


with ThreadPoolExecutor(workers) as ex:
    rcs = list(ex.map(run, jobs))
print("fallos:", sum(rc != 0 for rc in rcs), "de", len(jobs), "| JSON:", len(list(OUT.glob("*.json"))), flush=True)
