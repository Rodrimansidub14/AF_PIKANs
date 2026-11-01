"""EXP-0024, nivel T1: prueba corta en T4 antes del bloque de 40 corridas. Dos procesos en paralelo (Burgers en la GPU 0, Allen-Cahn en la GPU 1), semilla 1.

Parte A (comparabilidad con la CPU): CTRL y D1 con presupuesto reducido (Adam 200 pasos, L-BFGS 50 iteraciones), el mismo que se corrio en la CPU del investigador
(`cpu_reference.json`). Criterio fijado ANTES de correr (la GPU no es bit a bit con la CPU): error relativo entre GPU y CPU de CTRL <= 5 % al final de Adam y <= 10 % tras
L-BFGS; de D1 <= 20 % en ambos (la reubicacion usa cuantiles y un ajuste lineal que amplifican diferencias pequenas).
Parte B (tiempo): CTRL y D3 con el presupuesto T1 completo (Adam 10 % del control, L-BFGS 500 iteraciones); D3 es el diseno con mas reubicaciones (cota superior de D1 y D2).
Estimacion del bloque: sum_{problema} 5 semillas x (t_CTRL + 3 x t_D3) / 2 GPU, mas 30 s por trabajo (calentamiento) y 15 % de margen. Se lanza solo si <= 4 h.
Todo con calentamiento previo por proceso (EXP-0016). Ninguna parte puntua hipotesis.

Variables de entorno (solo para probar en CPU): SHORTTEST_TINY=1, BUNDLE_DIR, OUT_DIR.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

TINY = os.environ.get("SHORTTEST_TINY") == "1"
bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("t1_trainer.py")).parent)
out = Path(os.environ.get("OUT_DIR", "/kaggle/working/t1_shorttest"))
out.mkdir(parents=True, exist_ok=True)
cpu_ref = json.loads((bundle / "cpu_reference.json").read_text())
TOL = {"CTRL": (0.05, 0.10), "D1": (0.20, 0.20)}
ADAM_A, ITERS_A = (20, 10) if TINY else (200, 50)
ADAM_FULL = {"burgers": 2000, "ac": 4000}


def trainer(problem, designs, tag, adam, iters, slot):
    env = dict(os.environ, BUNDLE_DIR=str(bundle))
    if TINY:
        env["OMP_NUM_THREADS"] = "2"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
    cmd = [sys.executable, str(bundle / "t1_trainer.py"), "--problem", problem, "--designs", designs, "--seeds", "1", "--adam", str(adam),
           "--lbfgs-iters", str(iters), "--out", str(out / tag), "--no-ledger", "--device", "cpu" if TINY else "cuda:0", "--warmup-steps", "10" if TINY else "50"]
    with open(out / f"{tag}.log", "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(bundle)).returncode


rc = {}


def job(problem, slot):
    rc[(problem, "A")] = trainer(problem, "CTRL,D1", f"{problem}-A", ADAM_A, ITERS_A, slot)
    adam_b = 20 if TINY else ADAM_FULL[problem]
    rc[(problem, "B")] = trainer(problem, "CTRL,D3", f"{problem}-B", adam_b, 10 if TINY else 500, slot)


t0 = time.time()
threads = [threading.Thread(target=job, args=("burgers", 0)), threading.Thread(target=job, args=("ac", 1))]
[t.start() for t in threads]
[t.join() for t in threads]


def load(problem, tag, design, adam, iters):
    f = out / tag / f"{problem}-{design}-seed1-a{adam}-l{iters}.json"
    return json.loads(f.read_text()) if f.exists() else None


report, ok_compare, finite = {}, True, True
for problem in ("burgers", "ac"):
    rep = {}
    for design in ("CTRL", "D1"):
        g = load(problem, f"{problem}-A", design, ADAM_A, ITERS_A)
        c = cpu_ref.get(f"{problem}-{design}")
        if g is None:
            rep[design] = dict(error="sin resultado en GPU")
            ok_compare = False
            continue
        finite &= bool(g.get("adam_status") == "adam_complete" and g.get("primary_final") is not None)
        entry = dict(gpu_adam=g["primary_adam"], gpu_final=g["primary_final"], gpu_seconds=g.get("total_seconds"))
        if c:
            da, df = abs(g["primary_adam"] - c["primary_adam"]) / c["primary_adam"], abs(g["primary_final"] - c["primary_final"]) / c["primary_final"]
            entry.update(cpu_adam=c["primary_adam"], cpu_final=c["primary_final"], rel_diff_adam=da, rel_diff_final=df,
                         within_tolerance=bool(da <= TOL[design][0] and df <= TOL[design][1]))
            ok_compare &= entry["within_tolerance"]
        elif not TINY:
            ok_compare = False
        rep[design] = entry
    for design in ("CTRL", "D3"):
        adam_b = 20 if TINY else ADAM_FULL[problem]
        b = load(problem, f"{problem}-B", design, adam_b, 10 if TINY else 500)
        rep[f"{design}_full"] = None if b is None else dict(total_seconds=b["total_seconds"], adam_seconds=b["adam_seconds"], lbfgs_seconds=b["lbfgs_seconds"],
                                                           s_per_adam_step=b["adam_seconds"] / b["adam_steps"], primary_final=b["primary_final"],
                                                           regrid_fit_errors=[(x["l1_fit"]["fit_rel_error"], x["l2_fit"]["fit_rel_error"]) for x in b["regrid_diag"]])
        if b is None:
            finite = False
    report[problem] = rep

est = None
if all(report[p].get("CTRL_full") and report[p].get("D3_full") for p in report):
    per_problem = {p: 5 * (report[p]["CTRL_full"]["total_seconds"] + 3 * report[p]["D3_full"]["total_seconds"]) + 5 * 30 for p in report}
    est = dict(per_problem_seconds=per_problem, wall_hours=sum(per_problem.values()) / 2 / 3600 * 1.15)
summary = dict(tiny=TINY, returncodes={f"{k[0]}-{k[1]}": v for k, v in rc.items()}, comparable_to_cpu=bool(ok_compare), finite=bool(finite), estimate=est,
               estimate_le_4h=bool(est and est["wall_hours"] <= 4.0), wall_seconds=time.time() - t0, report=report)
summary["passed"] = bool(all(v == 0 for v in rc.values()) and ok_compare and finite and summary["estimate_le_4h"])
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "report"}, indent=1), flush=True)
