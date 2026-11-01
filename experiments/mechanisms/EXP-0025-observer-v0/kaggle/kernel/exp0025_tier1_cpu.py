"""EXP-0025, prueba 9 (Tier 1) en Kaggle CPU (SIN GPU): B y O en el mismo hardware, receta real (Adam 4000, K=500, n_p=25), Allen-Cahn d=1e-3, semillas 2901/2902
(fuera de T1=2001-2005 y V=2101-2105). Un proceso por semilla (2 hilos c/u). Sin L-BFGS ni error contra palancas fijas: solo reloj, acciones e inercia. Sin internet."""
import json, os, subprocess, sys, time
from pathlib import Path
bundle = Path(next(Path("/kaggle/input").rglob("tier1.py")).parent)
out = Path("/kaggle/working/exp0025_tier1"); out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, BUNDLE_DIR=str(bundle), OMP_NUM_THREADS="2")
procs = []
for seed in (2901, 2902):
    log = open(out / f"tier1_seed{seed}.log", "w")
    procs.append((seed, subprocess.Popen([sys.executable, str(bundle / "tier1.py"), "--seed", str(seed), "--threads", "2", "--out", str(out / f"tier1_seed{seed}.json")],
                                          stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle)), time.time()))
    print("start", seed, flush=True)
for seed, p, t0 in procs:
    rc = p.wait()
    print("end", seed, "rc", rc, "seconds", round(time.time() - t0), flush=True)
print(json.dumps({s: (out / f"tier1_seed{s}.json").exists() for s, _, _ in procs}), flush=True)
