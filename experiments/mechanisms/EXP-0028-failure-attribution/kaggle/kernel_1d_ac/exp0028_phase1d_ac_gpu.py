"""EXP-0028 Fase 1d en T4x2 (plantilla: kaggle_block.py de la fase 1; el planificador `run_jobs` es el mismo). Un trabajo = (problema, brazo, semilla) = un proceso de `run_phase1d.py`
(receta COMPLETA de la linea base; ver su docstring). Dos ranuras (una por GPU), cada proceso con calentamiento propio de 50 pasos.
Orden de despacho: por semilla y, dentro de cada semilla, B primero; si el tope corta, faltan las ultimas semillas completas y no un brazo suelto.
Topes: no se despacha trabajo nuevo despues de CUTOFF_S (3.4 h); a HARD_CAP_S (3.9 h) se matan los trabajos en curso (tope de 4 h por lanzamiento). Corridas ya escritas: reanudable.
BLOCK (fijado por build_bundle_1d.py): 'ac' | 'kpp' | 'bu1' | 'bu2' | 'bu3'. Solo humo en CPU: X28_TINY=1, BUNDLE_DIR, OUT_DIR, X28_SEEDS, X28_CUTOFF_S, X28_HARD_CAP_S.
NO LANZADO: preparado para revision."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BLOCK = "ac"   # fijado por build_bundle_1d.py
# bloque -> (problema, brazos, semillas); reparto fijado en budget_1d.md
PLAN = {
    "ac": ("ac", ("B", "E", "F", "C"), (4101, 4102, 4103, 4104, 4105)),
    "kpp": ("kpp", ("B", "E", "F"), (4101, 4102, 4103, 4104, 4105)),
    "bu1": ("burgers", ("B", "F", "E"), (4101, 4102)),
    "bu2": ("burgers", ("B", "F", "E"), (4103, 4104)),
    "bu3": ("burgers", ("B", "F", "E"), (4105,)),
}
CUTOFF_S = 3.4 * 3600
HARD_CAP_S = 3.9 * 3600


def make_jobs(block, seeds=None):
    problem, arms, block_seeds = PLAN[block]
    keep = set(seeds) if seeds else set(block_seeds)
    return [(problem, arm, s) for s in block_seeds if s in keep for arm in arms]


def run_jobs(jobs, n_workers, cutoff_s, hard_cap_s, launch, poll=2.0, clock=time.monotonic, sleep=time.sleep, on_event=None):
    """Planificador de ranuras (identico al de la fase 1). Estados: ok | failed | killed_hard_cap | not_started_cutoff."""
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
    tiny = os.environ.get("X28_TINY") == "1"
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("run_phase1d.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/exp0028_1d"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in os.environ["X28_SEEDS"].split(",")) if os.environ.get("X28_SEEDS") else None
    cutoff = float(os.environ.get("X28_CUTOFF_S", CUTOFF_S))
    hard_cap = float(os.environ.get("X28_HARD_CAP_S", HARD_CAP_S))
    import torch

    n_gpu = 0 if tiny else torch.cuda.device_count()
    n_workers = 2 if tiny else max(1, min(2, n_gpu))
    info = dict(tiny=tiny, block=BLOCK, plan=PLAN[BLOCK], n_gpu=n_gpu, n_workers=n_workers, torch=torch.__version__, cutoff_s=cutoff, hard_cap_s=hard_cap,
                gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
    print(json.dumps(info), flush=True)
    t_start = time.time()

    def command(job):
        problem, arm, seed = job
        cmd = [sys.executable, str(bundle / "run_phase1d.py"), "--problems", problem, "--arms", arm, "--seeds", str(seed), "--out", str(out), "--device", "cpu" if tiny else "cuda:0"]
        cmd += (["--adam", "10", "--lbfgs-iters", "5", "--lbfgs-seconds", "30", "--pre-every", "5", "--pre-n", "64", "--n-col", "512"] if tiny else ["--warmup-steps", "50"])
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
        print(f"[{elapsed:8.0f}s] {kind:8s} {job[0]}-{job[1]}-seed{job[2]} slot={slot} rc={rc}", flush=True)
        (out / "progress.json").write_text(json.dumps(dict(last_event=[kind, job, slot, round(elapsed), rc], wall_seconds=time.time() - t_start)))

    jobs = make_jobs(BLOCK, seeds)
    ledger = run_jobs(jobs, n_workers, cutoff, hard_cap, launch, poll=2.0 if tiny else 20.0, on_event=on_event)
    done = sorted(p.name for p in out.glob("*-seed*.json") if not p.name.endswith("note.json"))
    summary = dict(info=info, ledger=ledger, finished_runs=done, wall_seconds=time.time() - t_start, complete=all(e["status"] == "ok" for e in ledger) and len(ledger) == len(jobs))
    (out / "ledger.json").write_text(json.dumps(summary, indent=1))
    for log in sorted((out / "logs").glob("*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.name, "|", (lines[-1][:300] if lines else "(vacio)"), flush=True)
    print("complete:", summary["complete"], "runs:", len(done), flush=True)


if __name__ == "__main__":
    main()
