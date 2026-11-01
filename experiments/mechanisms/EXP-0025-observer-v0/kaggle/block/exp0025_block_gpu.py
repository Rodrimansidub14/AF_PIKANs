"""EXP-0025 (v1 firmada 2026-10-21; aG = variante E fijada por 1-C el 2026-10-22), bloque T1 en T4x2.
Problemas: Allen-Cahn d=1e-3 ('ac') y Burgers ('burgers'); semillas T1 = 2001-2005; brazos B, R, E, O, Orand (50 corridas) y, si O elige a0 en > 50 % de las decisiones de un problema,
el brazo Om (O-m: acciones al azar con las frecuencias de O en ese problema) para ese problema (+ hasta 10 corridas). Receta: ver observer_block.py (1-B: AC RBF16 Adam 4000, Burgers RBF48 Adam 2000, L-BFGS 500).
Fase 1: un trabajo = (problema, semilla) = un proceso que corre B, R, E, O, Orand en ese orden (B primero). Fase 2: Om por (problema, semilla), con las frecuencias de O de la fase 1.
Topes del spec (1.5 h de T4x2 por bloque): no se despacha trabajo nuevo despues de 1.0 h de pared del bloque; a 1.5 h se matan los trabajos en curso. Lo que no termine es ausencia de medicion.
Variables solo para humo en CPU: B25_TINY=1, BUNDLE_DIR, OUT_DIR, B25_CUTOFF_S, B25_HARD_CAP_S, B25_SEEDS, B25_PROBLEMS."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = tuple(int(s) for s in os.environ.get("B25_SEEDS", "2001,2002,2003,2004,2005").split(","))
PROBLEMS = tuple(os.environ.get("B25_PROBLEMS", "ac,burgers").split(","))
PHASE1_ARMS = "B,R,E,O,Orand"
CUTOFF_S = 1.0 * 3600
HARD_CAP_S = 1.5 * 3600


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


def freq_a0(out, problem):
    """Fraccion de decisiones con a0 de O en `problem` (JSON de la fase 1); None si no hay corridas de O."""
    ch = []
    for f in out.glob(f"{problem}-O-seed*.json"):
        ch += json.loads(f.read_text())["observer"]["chosen"]
    return (sum(c == "a0" for c in ch) / len(ch)) if ch else None


def main():
    tiny = os.environ.get("B25_TINY") == "1"
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("observer_block.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/exp0025_block_t1"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cutoff = float(os.environ.get("B25_CUTOFF_S", CUTOFF_S))
    hard_cap = float(os.environ.get("B25_HARD_CAP_S", HARD_CAP_S))
    import torch

    n_gpu = 0 if tiny else torch.cuda.device_count()
    n_workers = 2 if tiny else max(1, min(2, n_gpu))
    info = dict(tiny=tiny, n_gpu=n_gpu, n_workers=n_workers, torch=torch.__version__, cutoff_s=cutoff, hard_cap_s=hard_cap, seeds=SEEDS, problems=PROBLEMS,
                gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
    print(json.dumps(info), flush=True)
    t_start = time.time()

    def command(job):
        problem, seed, arms = job
        cmd = [sys.executable, str(bundle / "observer_block.py"), "--problem", problem, "--arms", arms, "--seeds", str(seed), "--out", str(out),
               "--device", "cpu" if tiny else "cuda:0", "--probs-from", str(out)]
        if tiny:
            cmd += ["--adam", "20", "--lbfgs-iters", "5", "--K", "10", "--n-p", "3", "--pre-n", "64", "--h-n", "256"]
        return cmd

    def launch(job, slot):
        env = dict(os.environ, BUNDLE_DIR=str(bundle))
        if tiny:
            env["OMP_NUM_THREADS"] = "2"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        log = open(out / "logs" / f"{job[0]}-seed{job[1]}-{job[2].replace(',', '')}.log", "w")
        return subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle))

    def on_event(kind, job, slot, elapsed, rc):
        print(f"[{elapsed:8.0f}s] {kind:8s} {job[0]}-seed{job[1]}-{job[2]} slot={slot} rc={rc}", flush=True)
        (out / "progress.json").write_text(json.dumps(dict(last_event=[kind, job, slot, round(elapsed), rc], wall_seconds=time.time() - t_start)))

    poll = 2.0 if tiny else 20.0
    jobs1 = [(p, s, PHASE1_ARMS) for p in PROBLEMS for s in SEEDS]
    ledger1 = run_jobs(jobs1, n_workers, cutoff, hard_cap, launch, poll=poll, on_event=on_event)
    elapsed1 = time.time() - t_start
    # fase 2: O-m solo en los problemas donde O eligio a0 en mas de la mitad de las decisiones (spec); con el reloj que queda del bloque
    fa0 = {p: freq_a0(out, p) for p in PROBLEMS}
    om_problems = [p for p in PROBLEMS if fa0[p] is not None and fa0[p] > 0.5]
    jobs2 = [(p, s, "Om") for p in om_problems for s in SEEDS]
    ledger2 = run_jobs(jobs2, n_workers, max(0.0, cutoff - elapsed1), max(0.0, hard_cap - elapsed1), launch, poll=poll, on_event=on_event) if jobs2 else []
    done = sorted(p.name for p in out.glob("*-seed*.json"))
    summary = dict(info=info, ledger_phase1=ledger1, ledger_phase2=ledger2, freq_a0_O=fa0, om_problems=om_problems, finished_runs=done, wall_seconds=time.time() - t_start,
                   complete=all(e["status"] == "ok" for e in ledger1 + ledger2) and len(ledger1) == len(jobs1) and len(ledger2) == len(jobs2))
    (out / "ledger.json").write_text(json.dumps(summary, indent=1))
    for log in sorted((out / "logs").glob("*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.name, "|", (lines[-1][:300] if lines else "(vacio)"), flush=True)
    print("freq a0 de O:", fa0, "Om en:", om_problems, "complete:", summary["complete"], "runs:", len(done), flush=True)


if __name__ == "__main__":
    main()
