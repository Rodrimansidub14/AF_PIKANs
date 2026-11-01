"""EXP-0028, REBANADA con artefactos (Enmienda v1.6) en T4x2 (plantilla: kaggle_block_1e.py; el planificador `run_jobs` es el mismo). Un trabajo = (problema, brazo, semilla) = un proceso de `run_slice.py`
(calentamiento propio de 50 pasos, SIN registro; luego la corrida con registro de artefactos y W&B OFFLINE dentro de la salida). Dos ranuras (una por GPU). Orden de despacho: por problema, por semilla y, dentro de
cada semilla, B,F,J,E,I; si el tope corta, faltan semillas completas y no un brazo suelto. Topes: no se despacha trabajo nuevo despues de CUTOFF_S (3.4 h); a HARD_CAP_S (3.9 h) se matan los trabajos en curso.
Antes de despachar un problema se corre en la GPU la prueba `run_slice.py --check` (B == ID con registro, y B con registro == B sin registro, bit a bit); si falla NO se despacha ese problema.
Corridas ya escritas (JSON + npz): reanudable. BLOCK (fijado por build_bundle_slice.py): ns | dfg | ackpp | bu. Solo humo en CPU: X28_TINY=1, BUNDLE_DIR, OUT_DIR, X28_SEEDS, X28_CUTOFF_S, X28_HARD_CAP_S.
Sin internet: W&B queda OFFLINE en OUT/wandb; ninguna clave en el kernel ni en el bundle (la sincronizacion es local: wandb_tools/sync_wandb.py). NO LANZADO: preparado para revision."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BLOCK = "ns"   # @@BLOCK@@
ARMS = ("B", "F", "J", "E", "I")
S = lambda a, b: tuple(range(a, b + 1))  # noqa: E731
# bloque -> lista de (problema, brazos, semillas); reparto y prioridad en budget_slice.md
PLAN = {
    "ns": [("ns", ARMS, S(4201, 4202))],
    "dfg": [("dfg", ARMS, S(4201, 4202))],
    "ackpp": [("ac", ARMS, S(4201, 4202)), ("kpp", ARMS, S(4201, 4202))],
    "bu": [("burgers", ARMS, S(4201, 4202))],
}
CUTOFF_S = 3.4 * 3600
HARD_CAP_S = 3.9 * 3600
TINY = {"ns": ["--smoke-pool", "128"], "dfg": ["--smoke-pool", "128"], "ac": [], "kpp": [], "burgers": []}


def make_jobs(block, seeds=None):
    jobs = []
    for problem, arms, block_seeds in PLAN[block]:
        keep = set(seeds) if seeds else set(block_seeds)
        jobs += [(problem, arm, s) for s in block_seeds if s in keep for arm in arms]
    return jobs


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
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("run_slice.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/exp0028_slice"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in os.environ["X28_SEEDS"].split(",")) if os.environ.get("X28_SEEDS") else None
    cutoff = float(os.environ.get("X28_CUTOFF_S", CUTOFF_S))
    hard_cap = float(os.environ.get("X28_HARD_CAP_S", HARD_CAP_S))
    import torch

    n_gpu = 0 if tiny else torch.cuda.device_count()
    n_workers = 2 if tiny else max(1, min(2, n_gpu))
    try:
        import wandb
        wb = getattr(wandb, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        wb = f"no disponible ({type(e).__name__})"
    info = dict(tiny=tiny, block=BLOCK, plan=PLAN[BLOCK], n_gpu=n_gpu, n_workers=n_workers, torch=torch.__version__, wandb=wb, python=sys.version.split()[0], cutoff_s=cutoff, hard_cap_s=hard_cap,
                gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
    print(json.dumps(info), flush=True)
    t_start = time.time()
    dev = "cpu" if tiny else "cuda:0"
    base_env = dict(os.environ, BUNDLE_DIR=str(bundle), WANDB_MODE="offline", **({"OMP_NUM_THREADS": "2"} if tiny else {}))
    base_env.pop("WANDB_API_KEY", None)                       # el kernel nunca recibe ni usa una clave
    jobs = make_jobs(BLOCK, seeds)
    checks, ledger_pre = {}, []
    for problem in [p for p, _, _ in PLAN[BLOCK]]:
        if tiny:
            extra = ["--adam", "10", "--smoke-pool", "128", "--lbfgs-iters", "5", "--pre-every", "5", "--pre-n", "64"]
        else:
            extra = ["--adam", "20", "--smoke-pool", "512", "--warmup-steps", "50"]
        log = open(out / "logs" / f"check-{problem}.log", "w")
        env = dict(base_env, **({} if tiny else {"CUDA_VISIBLE_DEVICES": "0"}))
        rc = subprocess.run([sys.executable, str(bundle / "run_slice.py"), "--problem", problem, "--check", "--device", dev, "--out", str(out / "checks")] + extra,
                            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle)).returncode
        chk = out / "checks" / f"check_slice_{problem}.json"
        checks[problem] = chk.exists() and json.loads(chk.read_text()).get("pass") is True
        print(f"check B == ID / registro neutro [{problem}]:", checks[problem], "rc", rc, flush=True)
        if not checks[problem]:
            ledger_pre += [dict(job=j, slot=None, start=None, end=None, returncode=None, status="not_started_check_failed") for j in jobs if j[0] == problem]
            jobs = [j for j in jobs if j[0] != problem]

    def command(job):
        problem, arm, seed = job
        cmd = [sys.executable, str(bundle / "run_slice.py"), "--problem", problem, "--arms", arm, "--seeds", str(seed), "--out", str(out), "--device", dev]
        cmd += (["--adam", "10", "--lbfgs-iters", "5", "--pre-every", "5", "--pre-n", "64", "--light-every", "5"] + TINY[problem]) if tiny else ["--warmup-steps", "50"]
        return cmd

    def launch(job, slot):
        env = dict(base_env)
        if not tiny:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        log = open(out / "logs" / f"{job[0]}-{job[1]}-seed{job[2]}.log", "w")
        return subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bundle))

    def on_event(kind, job, slot, elapsed, rc):
        print(f"[{elapsed:8.0f}s] {kind:8s} {job[0]}-{job[1]}-seed{job[2]} slot={slot} rc={rc}", flush=True)
        (out / "progress.json").write_text(json.dumps(dict(last_event=[kind, job, slot, round(elapsed), rc], wall_seconds=time.time() - t_start)))

    ledger = ledger_pre + run_jobs(jobs, n_workers, cutoff, hard_cap, launch, poll=2.0 if tiny else 20.0, on_event=on_event)
    done = sorted(p.name for p in out.glob("*-seed*.json"))
    npz = sorted(out.glob("*-artifacts.npz"))
    n_total = len(make_jobs(BLOCK, seeds))
    summary = dict(info=info, checks=checks, ledger=ledger, finished_runs=done, artifacts_mb=sum(p.stat().st_size for p in npz) / 2 ** 20, n_artifacts=len(npz),
                   wandb_offline_runs=len(list((out / "wandb").glob("offline-run-*"))) if (out / "wandb").exists() else 0,
                   wall_seconds=time.time() - t_start, complete=all(e["status"] == "ok" for e in ledger) and len(ledger) == n_total)
    (out / "ledger.json").write_text(json.dumps(summary, indent=1))
    for log in sorted((out / "logs").glob("*-seed*.log")):
        lines = log.read_text().strip().splitlines()
        print(log.name, "|", (lines[-1][:300] if lines else "(vacio)"), flush=True)
    print("complete:", summary["complete"], "runs:", len(done), "artifacts MB: %.1f" % summary["artifacts_mb"], flush=True)


if __name__ == "__main__":
    main()
