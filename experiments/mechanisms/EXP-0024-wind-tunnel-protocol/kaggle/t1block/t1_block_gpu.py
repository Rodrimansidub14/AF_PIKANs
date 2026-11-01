"""EXP-0024, nivel T1: bloque de 40 corridas (CTRL, D1, D2, D3 x semillas 1-5 x Burgers y Allen-Cahn) en T4x2.

Un trabajo = (problema, semilla) = un proceso `t1_trainer.py` con su calentamiento, que corre los cuatro disenos en orden (CTRL primero, el comparador pareado de la misma
sesion). Dos ranuras (una por GPU), Burgers primero. Convencion T1: Adam = 10 % del control, L-BFGS con tope de 500 iteraciones.
Topes de la autorizacion: no se despacha trabajo nuevo despues de CUTOFF_S (4 h); a HARD_CAP_S (5 h) se matan los trabajos en curso. Lo que no termine es ausencia de medicion.
Variables de entorno (solo para probar en CPU): T1BLOCK_TINY=1, BUNDLE_DIR, OUT_DIR, T1BLOCK_CUTOFF_S, T1BLOCK_HARD_CAP_S.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

JOBS = [("burgers", s) for s in (1, 2, 3, 4, 5)] + [("ac", s) for s in (1, 2, 3, 4, 5)]
DESIGNS = "CTRL,D1,D2,D3"
CUTOFF_S = 4.0 * 3600
HARD_CAP_S = 5.0 * 3600


def run_jobs(jobs, n_workers, cutoff_s, hard_cap_s, launch, poll=2.0, clock=time.monotonic, sleep=time.sleep, on_event=None):
    """Planificador de ranuras (igual al de los bloques anteriores). Estados: ok | failed | killed_hard_cap | not_started_cutoff."""
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
    tiny = os.environ.get("T1BLOCK_TINY") == "1"
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("t1_trainer.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/t1block"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cutoff = float(os.environ.get("T1BLOCK_CUTOFF_S", CUTOFF_S))
    hard_cap = float(os.environ.get("T1BLOCK_HARD_CAP_S", HARD_CAP_S))
    import torch

    n_gpu = 0 if tiny else torch.cuda.device_count()
    n_workers = 2 if tiny else max(1, min(2, n_gpu))
    info = dict(tiny=tiny, n_gpu=n_gpu, n_workers=n_workers, torch=torch.__version__, cutoff_s=cutoff, hard_cap_s=hard_cap, designs=DESIGNS,
                gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
    print(json.dumps(info), flush=True)
    t_start = time.time()

    def command(job):
        problem, seed = job
        cmd = [sys.executable, str(bundle / "t1_trainer.py"), "--problem", problem, "--designs", DESIGNS, "--seeds", str(seed), "--out", str(out),
               "--local-ledger", "--device", "cpu" if tiny else "cuda:0", "--warmup-steps", "10" if tiny else "50"]
        if tiny:
            cmd += ["--adam", "20", "--lbfgs-iters", "10"]
        return cmd

    def launch(job, slot):
        env = dict(os.environ, BUNDLE_DIR=str(bundle))
        if tiny:
            env["OMP_NUM_THREADS"] = "2"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        log = open(out / "logs" / f"{job[0]}-seed{job[1]}.log", "w")
        return subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle))

    def on_event(kind, job, slot, elapsed, rc):
        print(f"[{elapsed:8.0f}s] {kind:8s} {job[0]}-seed{job[1]} slot={slot} rc={rc}", flush=True)
        (out / "progress.json").write_text(json.dumps(dict(last_event=[kind, job, slot, round(elapsed), rc], wall_seconds=time.time() - t_start)))

    ledger = run_jobs(JOBS, n_workers, cutoff, hard_cap, launch, poll=2.0 if tiny else 20.0, on_event=on_event)
    done = sorted(p.name for p in out.glob("*-seed*.json"))
    summary = dict(info=info, ledger=ledger, finished_runs=done, wall_seconds=time.time() - t_start,
                   complete=all(e["status"] == "ok" for e in ledger) and len(ledger) == len(JOBS))
    (out / "ledger.json").write_text(json.dumps(summary, indent=1))
    for log in sorted((out / "logs").glob("*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.name, "|", (lines[-1][:300] if lines else "(vacio)"), flush=True)
    print("complete:", summary["complete"], "runs:", len(done), flush=True)


if __name__ == "__main__":
    main()
