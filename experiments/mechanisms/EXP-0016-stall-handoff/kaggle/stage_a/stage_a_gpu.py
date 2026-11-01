"""EXP-0016, etapa A (techo del traspaso): Burgers (semillas 301-305) y despues Allen-Cahn (601-603), en T4x2.

Un trabajo = una semilla de un problema = un proceso `sweep.py` (con calentamiento propio, ver spec) fijado a una GPU. Dos ranuras (una por GPU) toman
los trabajos en orden de una cola: Burgers primero (decide), luego Allen-Cahn. Heat NO se corre en esta firma.

Topes del spec (presupuesto): esperado ~3.1 h, tope duro 5 h de sesion. Implementacion:
  - no se despacha ningun trabajo nuevo despues de CUTOFF_S (3.5 h): un trabajo de Burgers dura ~0.65 h esperadas y, en el peor caso medido por tope de L-BFGS, ~1.5 h;
  - a HARD_CAP_S (5 h) se matan los trabajos en curso. Los JSON de las semillas ya completas quedan intactos.
Si hay que recortar, el spec dice: Allen-Cahn baja a 2 semillas (por eso va al final de la cola).

Variables de entorno (solo para probar en CPU): STAGEA_TINY=1, BUNDLE_DIR, OUT_DIR, STAGEA_CUTOFF_S, STAGEA_HARD_CAP_S.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

JOBS = [("burgers", s) for s in (301, 302, 303, 304, 305)] + [("ac", s) for s in (601, 602, 603)]
CUTOFF_S = 3.5 * 3600
HARD_CAP_S = 5.0 * 3600
RECIPES = {
    "burgers": dict(ref="burgers_coleHopf_1025x129.npz", width=48, adam=20000, lbfgs_seconds=600),
    "ac": dict(ref="Allen_Cahn.mat", width=16, adam=40000, lbfgs_seconds=300),
}


def run_jobs(jobs, n_workers, cutoff_s, hard_cap_s, launch, poll=2.0, clock=time.monotonic, sleep=time.sleep, on_event=None):
    """Planificador de ranuras. `launch(job, slot)` devuelve un objeto con poll() y kill() (como Popen).

    Estados del libro de registro: ok | failed | killed_hard_cap | not_started_cutoff.
    """
    t0 = clock()
    queue = list(jobs)
    running, ledger = {}, []
    emit = on_event or (lambda *a: None)
    while queue or running:
        elapsed = clock() - t0
        for slot, (job, proc, start) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                ledger.append(dict(job=job, slot=slot, start=start, end=elapsed, returncode=rc, status="ok" if rc == 0 else "failed"))
                emit("end", job, slot, elapsed, rc)
                del running[slot]
        if elapsed >= hard_cap_s:
            for slot, (job, proc, start) in list(running.items()):
                proc.kill()
                ledger.append(dict(job=job, slot=slot, start=start, end=elapsed, returncode=None, status="killed_hard_cap"))
                emit("killed", job, slot, elapsed, None)
                del running[slot]
        if elapsed >= cutoff_s or elapsed >= hard_cap_s:
            for job in queue:
                ledger.append(dict(job=job, slot=None, start=None, end=None, returncode=None, status="not_started_cutoff"))
                emit("skipped", job, None, elapsed, None)
            queue = []
        for slot in range(n_workers):
            if queue and slot not in running:
                job = queue.pop(0)
                running[slot] = (job, launch(job, slot), elapsed)
                emit("start", job, slot, elapsed, None)
        if queue or running:
            sleep(poll)
    return ledger


def main():
    tiny = os.environ.get("STAGEA_TINY") == "1"
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("sweep.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/stageA"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cutoff = float(os.environ.get("STAGEA_CUTOFF_S", CUTOFF_S))
    hard_cap = float(os.environ.get("STAGEA_HARD_CAP_S", HARD_CAP_S))
    import torch

    n_gpu = 0 if tiny else torch.cuda.device_count()
    n_workers = 2 if tiny else max(1, min(2, n_gpu))
    info = dict(tiny=tiny, n_gpu=n_gpu, n_workers=n_workers, torch=torch.__version__, cutoff_s=cutoff, hard_cap_s=hard_cap,
                gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
    print(json.dumps(info), flush=True)
    t_start = time.time()

    def command(job):
        problem, seed = job
        r = RECIPES[problem]
        cmd = [sys.executable, str(bundle / "sweep.py"), "--problem", problem, "--ref", str(bundle / r["ref"]),
               "--campaign-py", str(bundle / "pikan_campaign.py"), "--seeds", str(seed), "--out", str(out),
               "--width", str(r["width"]), "--bases", "11", "--adam-steps", str(r["adam"] if not tiny else 40),
               "--lbfgs-seconds", str(r["lbfgs_seconds"]), "--device", "cpu" if tiny else "cuda:0"]
        if tiny:
            cmd += ["--lbfgs-max-iter", "60", "--lbfgs-chunk", "10", "--record-every", "10", "--warmup-steps", "10"]
            if problem == "burgers":
                cmd += ["--n-col", "256"]
        return cmd

    def launch(job, slot):
        env = dict(os.environ)
        if not tiny:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        else:
            env["OMP_NUM_THREADS"] = "1"
        log = open(out / "logs" / f"{job[0]}-{job[1]}.log", "w")
        return subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle))

    def on_event(kind, job, slot, elapsed, rc):
        print(f"[{elapsed:8.0f}s] {kind:8s} {job[0]}-{job[1]} slot={slot} rc={rc}", flush=True)
        (out / "progress.json").write_text(json.dumps(dict(last_event=[kind, job, slot, round(elapsed), rc], wall_seconds=time.time() - t_start)))

    ledger = run_jobs(JOBS, n_workers, cutoff, hard_cap, launch, poll=2.0 if tiny else 20.0, on_event=on_event)
    done = sorted(p.name for p in out.glob("*-seed*.json"))
    summary = dict(info=info, ledger=ledger, finished_seeds=done, wall_seconds=time.time() - t_start,
                   complete=all(e["status"] == "ok" for e in ledger) and len(ledger) == len(JOBS))
    (out / "ledger.json").write_text(json.dumps(summary, indent=1))
    for log in sorted((out / "logs").glob("*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.name, "|", (lines[-1][:400] if lines else "(vacio)"), flush=True)
    print("complete:", summary["complete"], "seeds:", len(done), flush=True)


if __name__ == "__main__":
    main()
