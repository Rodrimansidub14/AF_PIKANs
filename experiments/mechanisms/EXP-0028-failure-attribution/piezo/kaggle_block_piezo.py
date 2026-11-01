"""EXP-0028, bloque final de piezo (bimorfo PVDF, converso y directo) en T4x2. Un trabajo = (problema, semilla, brazo) = un proceso de `run_piezo.py` (receta del paper: Adam 20 000 pasos,
float64, sin L-BFGS; recalculo cada 1000 pasos). Dos ranuras (una por GPU). Orden de despacho: problema, luego semilla, luego B, F, J, E, I. Topes: no se despacha trabajo nuevo despues de
CUTOFF_S (3.4 h); a HARD_CAP_S (3.9 h) se matan los trabajos en curso. Las corridas escritas son reanudables (un JSON por corrida). Antes de despachar, el kernel corre en la GPU la prueba
B == ID con calentamiento (`run_piezo.py --check-id --warmup-steps 50`, los dos problemas) y la medicion de costo (`--timing`); si B != ID NO se despacha nada.
BLOCK (fijado por build_bundle_piezo.py): 't' = solo prueba + medicion; 'all' = los 52 trabajos; 'kNN' = el NN-esimo par de trabajos (k01..k25 = el mecanismo; k26 = las dos corridas de referencia MLP-paper, baja prioridad); cada trabajo ocupa una GPU 1-3 h.
Solo para humo en CPU: X28_TINY=1, BUNDLE_DIR, OUT_DIR, X28_CUTOFF_S, X28_HARD_CAP_S.
NO LANZADO: preparado para revision. Piezo no se corre antes de que el investigador firme predictions.md.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BLOCK = "t"   # @@BLOCK@@
PROBLEMS = ("converse", "direct")
ARMS = ("B", "F", "J", "E", "I")
SEEDS = tuple(range(20260728, 20260733))     # cohorte del paper / linea base congelada
REFERENCE_SEED = 20260728
CUTOFF_S = 3.4 * 3600
HARD_CAP_S = 3.9 * 3600


def all_jobs():
    """50 trabajos del mecanismo + (al final, BAJA PRIORIDAD) 2 de referencia: 'MLP-paper' = MLP 3x50 SiLU del paper, semilla 20260728, receta completa, uno por problema (ancla el arnes a los numeros publicados)."""
    return [(p, s, a) for p in PROBLEMS for s in SEEDS for a in ARMS] + [(p, REFERENCE_SEED, "MLP") for p in PROBLEMS]


def block_jobs(name):
    jobs = all_jobs()
    if name == "t":
        return []
    S1 = {"s1a": [("converse", REFERENCE_SEED, "B"), ("converse", REFERENCE_SEED, "E")],        # etapa 1 del PLAN_RECURSOS: semilla del paper
          "s1b": [("direct", REFERENCE_SEED, "B"), ("direct", REFERENCE_SEED, "E")],
          "s1c": [("direct", REFERENCE_SEED, "F"), ("direct", REFERENCE_SEED, "J")]}
    if name in S1:
        return S1[name]
    if name == "all":
        return jobs
    if name.startswith("k"):
        k = int(name[1:])
        return jobs[2 * (k - 1): 2 * k]
    raise ValueError(name)


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
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("run_piezo.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/exp0028_piezo"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cutoff = float(os.environ.get("X28_CUTOFF_S", CUTOFF_S))
    hard_cap = float(os.environ.get("X28_HARD_CAP_S", HARD_CAP_S))
    import torch

    n_gpu = 0 if tiny else torch.cuda.device_count()
    n_workers = 2 if tiny else max(1, min(2, n_gpu))
    jobs = block_jobs(BLOCK)
    info = dict(tiny=tiny, block=BLOCK, n_gpu=n_gpu, n_workers=n_workers, torch=torch.__version__, n_jobs=len(jobs), cutoff_s=cutoff, hard_cap_s=hard_cap,
                gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
    print(json.dumps(info), flush=True)
    t_start = time.time()
    dev = "cpu" if tiny else "cuda:0"
    env0 = dict(os.environ, BUNDLE_DIR=str(bundle), **({"OMP_NUM_THREADS": "2"} if tiny else {"CUDA_VISIBLE_DEVICES": "0"}))

    def pre_step(flag, extra):
        log = open(out / "logs" / f"pre{flag.strip('-')}.log", "w")
        return subprocess.run([sys.executable, str(bundle / "run_piezo.py"), flag, "--device", dev, "--out", str(out)] + extra, stdout=log, stderr=subprocess.STDOUT, env=env0, cwd=str(bundle)).returncode

    rc_id = pre_step("--check-id", ["--smoke-pool", "32", "--adam", "6"] if tiny else ["--adam", "12", "--smoke-pool", "32", "--warmup-steps", "50"])
    chk = out / "check_B_ID_piezo.json"
    passed = chk.exists() and json.loads(chk.read_text()).get("pass") is True
    print("check B == ID:", passed, "rc", rc_id, flush=True)
    if BLOCK.startswith("s1"):                       # el costo ya se midio en la etapa 0: no se gasta cuota en repetirlo
        print("timing omitido (bloque s1)", flush=True)
    else:
        rc_t = pre_step("--timing", ["--smoke-pool", "32", "--timing-chunks", "8"] if tiny else [])
        print("timing rc", rc_t, flush=True)
    if not passed:
        (out / "ledger.json").write_text(json.dumps(dict(info=info, ledger=[], aborted="B != ID", complete=False), indent=1))
        print("ABORT: B != ID, nothing dispatched", flush=True)
        return

    def command(job):
        problem, seed, arm = job
        cmd = [sys.executable, str(bundle / "run_piezo.py"), "--problems", problem, "--arms", arm, "--seeds", str(seed), "--out", str(out), "--device", dev]
        cmd += ["--adam", "10", "--eval-every", "5", "--pre-every", "5", "--pre-n", "64", "--smoke-pool", "16"] if tiny else ["--warmup-steps", "50"]
        return cmd

    def launch(job, slot):
        env = dict(os.environ, BUNDLE_DIR=str(bundle))
        if tiny:
            env["OMP_NUM_THREADS"] = "2"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        log = open(out / "logs" / f"{job[0]}-seed{job[1]}-{job[2]}.log", "w")
        return subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle))

    def on_event(kind, job, slot, elapsed, rc):
        print(f"[{elapsed:8.0f}s] {kind:8s} {job[0]}-seed{job[1]}-{job[2]} slot={slot} rc={rc}", flush=True)
        (out / "progress.json").write_text(json.dumps(dict(last_event=[kind, job, slot, round(elapsed), rc], wall_seconds=time.time() - t_start)))

    ledger = run_jobs(jobs, n_workers, cutoff, hard_cap, launch, poll=2.0 if tiny else 20.0, on_event=on_event)
    done = sorted(p.name for p in out.glob("piezo-*-seed*.json"))
    summary = dict(info=info, check_B_ID=passed, ledger=ledger, finished_runs=done, wall_seconds=time.time() - t_start, complete=all(e["status"] == "ok" for e in ledger) and len(ledger) == len(jobs))
    (out / "ledger.json").write_text(json.dumps(summary, indent=1))
    for log in sorted((out / "logs").glob("*-seed*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.name, "|", (lines[-1][:300] if lines else "(vacio)"), flush=True)
    print("complete:", summary["complete"], "runs:", len(done), flush=True)


if __name__ == "__main__":
    main()
