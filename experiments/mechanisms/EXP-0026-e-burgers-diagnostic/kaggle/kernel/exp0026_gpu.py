"""EXP-0026 v1.1 en T4x2. Trabajos: (a) factorial B,E1000L,E1000,E500L,E500 sobre semillas 3001-3010 en Burgers y en Allen-Cahn (20 trabajos); (b) replica cruzada en Burgers, brazos B,E1000L,E500 sobre
las semillas de 1-B 1-5 y 101-105 (10 trabajos); (c) lectura de R: `rad_diag.py` (1 trabajo, solo lectura). Dos ranuras (una por GPU). Primero (b) y (c), luego (a) Burgers y (a) Allen-Cahn.
Topes del spec v1.1: no se despacha trabajo nuevo despues de 1.75 h; a 2.25 h se matan los trabajos en curso. Solo para humo en CPU: X26_TINY=1, BUNDLE_DIR, OUT_DIR, X26_SEEDS."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = tuple(int(s) for s in os.environ.get("X26_SEEDS", ",".join(str(s) for s in range(3001, 3011))).split(","))
ARMS = "B,E1000L,E1000,E500L,E500"
CUTOFF_S = 1.75 * 3600
HARD_CAP_S = 2.25 * 3600
REPLAY = (1, 2, 3, 4, 5, 101, 102, 103, 104, 105)


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
    tiny = os.environ.get("X26_TINY") == "1"
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("diag_e.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/exp0026"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    import torch

    n_gpu = 0 if tiny else torch.cuda.device_count()
    n_workers = 2 if tiny else max(1, min(2, n_gpu))
    info = dict(tiny=tiny, n_gpu=n_gpu, n_workers=n_workers, torch=torch.__version__, seeds=SEEDS, arms=ARMS, gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
    print(json.dumps(info), flush=True)
    t_start = time.time()

    def command(job):
        kind, problem, seed = job
        dev = "cpu" if tiny else "cuda:0"
        if kind == "raddiag":
            return [sys.executable, str(bundle / "rad_diag.py"), "--seeds", "3001,3002,3003", "--out", str(out / "rad_diag.json"), "--device", dev] + (["--adam", "20"] if tiny else [])
        arms = "B,E1000L,E500" if kind == "replay" else ARMS
        cmd = [sys.executable, str(bundle / "diag_e.py"), "--problem", problem, "--arms", arms, "--seeds", str(seed), "--out", str(out), "--device", dev]
        if tiny:
            cmd += ["--adam", "20", "--lbfgs-iters", "3", "--pre-n", "64"]
        return cmd

    def launch(job, slot):
        env = dict(os.environ, BUNDLE_DIR=str(bundle))
        if tiny:
            env["OMP_NUM_THREADS"] = "2"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        log = open(out / "logs" / f"{job[0]}-{job[1]}-seed{job[2]}.log", "w")
        return subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle))

    def on_event(kind, job, slot, elapsed, rc):
        print(f"[{elapsed:8.0f}s] {kind:8s} {job} slot={slot} rc={rc}", flush=True)

    jobs = [("raddiag", "burgers", 0)] + [("replay", "burgers", x) for x in REPLAY] + [("factorial", "burgers", x) for x in SEEDS] + [("factorial", "ac", x) for x in SEEDS]
    ledger = run_jobs(jobs, n_workers, float(os.environ.get("X26_CUTOFF_S", CUTOFF_S)), float(os.environ.get("X26_HARD_CAP_S", HARD_CAP_S)), launch,
                      poll=2.0 if tiny else 20.0, on_event=on_event)
    done = sorted(p.name for p in out.glob("burgers-*-seed*.json"))
    summary = dict(info=info, ledger=ledger, finished_runs=done, wall_seconds=time.time() - t_start,
                   complete=all(e["status"] == "ok" for e in ledger) and len(ledger) == len(jobs))
    (out / "ledger.json").write_text(json.dumps(summary, indent=1))
    for log in sorted((out / "logs").glob("*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.name, "|", (lines[-1][:300] if lines else "(vacio)"), flush=True)
    print("complete:", summary["complete"], "runs:", len(done), flush=True)


if __name__ == "__main__":
    main()
