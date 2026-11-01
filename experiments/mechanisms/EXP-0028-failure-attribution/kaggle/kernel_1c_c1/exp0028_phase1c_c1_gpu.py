"""EXP-0028 Fase 1c (NS2d-CG, residuo vectorial) en T4x2 (plantilla: kaggle_block.py de la fase 1). Un trabajo = (semilla, brazo) = un proceso de `run_phase1c.py` (una corrida de ~20-25 min con la receta
completa: Adam 20 000 + L-BFGS 500 it.). Dos ranuras (una por GPU). Orden de despacho: semilla por semilla (B, F, J, E, I de la semilla s antes que la s+1), de modo que si el tope corta, lo que falta son las
ultimas semillas completas, no brazos sueltos. Topes: no se despacha trabajo nuevo despues de CUTOFF_S (3.4 h); a HARD_CAP_S (3.9 h) se matan los trabajos en curso. Las corridas escritas son reanudables.
Antes de despachar, el kernel corre en la GPU la prueba B == ID (run_phase1c.py --check-id) y la medicion de costo (--timing, chunk 8/16): quedan en OUT_DIR (check_B_ID_ns.json, timing_ns.json); si B != ID NO se
despacha nada. BLOCK (fijado por build_bundle_1c.py): c1 = semillas 4001-4003, c2 = 4004-4006, c3 = 4007-4009, c4 = 4010, all = 4001-4010, t = solo prueba+medicion (kernel de tiempos).
Solo para humo en CPU: X28_TINY=1, BUNDLE_DIR, OUT_DIR, X28_SEEDS, X28_CUTOFF_S, X28_HARD_CAP_S.
NO LANZADO: preparado para revision."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BLOCK = "c1"   # fijado por build_bundle_1c.py
SEEDS_BY_BLOCK = {"c1": (4001, 4002, 4003), "c2": (4004, 4005, 4006), "c3": (4007, 4008, 4009), "c4": (4010,), "all": tuple(range(4001, 4011)), "all_t1": tuple(range(4001, 4011)), "t": ()}
ARMS = ("B", "F", "J", "E", "I")
CUTOFF_S = 3.4 * 3600
HARD_CAP_S = 3.9 * 3600


def make_jobs(seeds):
    return [(s, arm) for s in seeds for arm in ARMS]


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
    tiny = os.environ.get("X28_TINY") == "1"
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("run_phase1c.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/exp0028_1c"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in os.environ["X28_SEEDS"].split(",")) if os.environ.get("X28_SEEDS") else SEEDS_BY_BLOCK[BLOCK]
    cutoff = float(os.environ.get("X28_CUTOFF_S", CUTOFF_S))
    hard_cap = float(os.environ.get("X28_HARD_CAP_S", HARD_CAP_S))
    import torch

    n_gpu = 0 if tiny else torch.cuda.device_count()
    n_workers = 2 if tiny else max(1, min(2, n_gpu))
    info = dict(tiny=tiny, block=BLOCK, n_gpu=n_gpu, n_workers=n_workers, torch=torch.__version__, seeds=seeds, cutoff_s=cutoff, hard_cap_s=hard_cap,
                gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
    print(json.dumps(info), flush=True)
    t_start = time.time()
    dev = "cpu" if tiny else "cuda:0"
    env0 = dict(os.environ, BUNDLE_DIR=str(bundle), **({"OMP_NUM_THREADS": "2"} if tiny else {"CUDA_VISIBLE_DEVICES": "0"}))

    def pre_step(flag, extra):
        """Prueba B == ID y medicion de costo en este dispositivo (una vez por lanzamiento). Devuelve el codigo de salida."""
        log = open(out / "logs" / f"pre{flag.strip('-')}.log", "w")
        return subprocess.run([sys.executable, str(bundle / "run_phase1c.py"), flag, "--device", dev, "--out", str(out)] + extra, stdout=log, stderr=subprocess.STDOUT, env=env0, cwd=str(bundle)).returncode

    smoke = ["--smoke-pool", "128", "--adam", "10"] if tiny else []
    rc_id = pre_step("--check-id", smoke if tiny else ["--adam", "20", "--smoke-pool", "512", "--warmup-steps", "50"])
    chk = out / "check_B_ID_ns.json"
    passed = chk.exists() and json.loads(chk.read_text()).get("pass") is True
    print("check B == ID:", passed, "rc", rc_id, flush=True)
    if BLOCK in ("t",) or not tiny:
        rc_t = pre_step("--timing", ["--smoke-pool", "128"] if tiny else [])
        print("timing rc", rc_t, flush=True)
    if not passed:
        (out / "ledger.json").write_text(json.dumps(dict(info=info, ledger=[], aborted="B != ID", complete=False), indent=1))
        print("ABORT: B != ID, nothing dispatched", flush=True)
        return

    def command(job):
        seed, arm = job
        cmd = [sys.executable, str(bundle / "run_phase1c.py"), "--arms", arm, "--seeds", str(seed), "--out", str(out), "--device", dev]
        cmd += ["--adam", "10", "--lbfgs-iters", "5", "--pre-every", "5", "--pre-n", "64", "--smoke-pool", "128"] if tiny else ["--warmup-steps", "50"]
        if BLOCK.endswith("_t1"):
            cmd += ["--recipe", "t1"]
        return cmd

    def launch(job, slot):
        env = dict(os.environ, BUNDLE_DIR=str(bundle))
        if tiny:
            env["OMP_NUM_THREADS"] = "2"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        log = open(out / "logs" / f"seed{job[0]}-{job[1]}.log", "w")
        return subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle))

    def on_event(kind, job, slot, elapsed, rc):
        print(f"[{elapsed:8.0f}s] {kind:8s} seed{job[0]}-{job[1]} slot={slot} rc={rc}", flush=True)
        (out / "progress.json").write_text(json.dumps(dict(last_event=[kind, job, slot, round(elapsed), rc], wall_seconds=time.time() - t_start)))

    jobs = make_jobs(seeds)
    ledger = run_jobs(jobs, n_workers, cutoff, hard_cap, launch, poll=2.0 if tiny else 20.0, on_event=on_event)
    done = sorted(p.name for p in out.glob("ns-*-seed*.json"))
    summary = dict(info=info, check_B_ID=passed, ledger=ledger, finished_runs=done, wall_seconds=time.time() - t_start, complete=all(e["status"] == "ok" for e in ledger) and len(ledger) == len(jobs))
    (out / "ledger.json").write_text(json.dumps(summary, indent=1))
    for log in sorted((out / "logs").glob("seed*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.name, "|", (lines[-1][:300] if lines else "(vacio)"), flush=True)
    print("complete:", summary["complete"], "runs:", len(done), flush=True)


if __name__ == "__main__":
    main()
