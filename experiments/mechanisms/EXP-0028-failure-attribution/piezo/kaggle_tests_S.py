"""EXP-0029, pruebas previas del brazo S en Kaggle T4x2 (cuota ~95 min). T0: humo de GPU de S (cold, 40 pasos) en los dos problemas; T1: arranque en caliente desde el estado del paso 4000 de B,
semillas 20260728-30, brazos E y S, 1000 pasos, recalculo cada 500; converso en la GPU 0 y directo en la GPU 1 a la vez. Orden semilla-mayor (E y S de la misma semilla seguidos), para que un corte deje pares completos.
Topes: sin trabajo nuevo despues de CUTOFF_S (65 min); a HARD_CAP_S (80 min) se matan los trabajos en curso. Solo para humo en CPU: X29_TINY=1, BUNDLE_DIR, OUT_DIR."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = (20260728, 20260729, 20260730)
T2 = os.environ.get("X29_MODE") == "t2"      # t2: S sin amortiguar (P4 en sombra), 3 semillas; sin T0
ARMS = ("S",) if T2 else ("E", "S")
CUTOFF_S, HARD_CAP_S = (20 * 60, 30 * 60) if T2 else (65 * 60, 80 * 60)
LANES = {"converse": "0", "direct": "1"}


def main():
    tiny = os.environ.get("X29_TINY") == "1"
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("run_piezo.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/exp0029_tests"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    if not (bundle / "states").exists() and (bundle / "states.zip").exists():
        import zipfile
        zipfile.ZipFile(bundle / "states.zip").extractall(out / "states_unzipped")
        bundle_states = out / "states_unzipped"
    else:
        bundle_states = bundle / "states"
    dev = "cpu" if tiny else "cuda:0"
    t0 = time.time()
    seeds = SEEDS[:1] if tiny else SEEDS
    smoke = ["--smoke-pool", "16", "--pre-n", "64", "--pre-chunk", "4"] if tiny else []

    def run(args, env_extra, log):
        env = dict(os.environ, BUNDLE_DIR=str(bundle), **env_extra)
        return subprocess.Popen([sys.executable, str(bundle / "run_piezo.py")] + args, stdout=open(out / "logs" / log, "w"), stderr=subprocess.STDOUT, env=env, cwd=str(bundle))

    # T0: humo de GPU del brazo S (los dos problemas a la vez, una GPU cada uno)
    ps = []
    for pr, lane in ({} if T2 else LANES).items():
        a = ["--problems", pr, "--arms", "S", "--seeds", "4001", "--adam", "10" if tiny else "40", "--pre-every", "20", "--eval-every", "20", "--no-artifacts", "--no-wandb",
             "--smoke-pool", "32", "--out", str(out / "t0"), "--device", dev] + (["--pre-n", "64", "--pre-chunk", "4"] if tiny else [])
        ps.append((pr, run(a, {} if tiny else {"CUDA_VISIBLE_DEVICES": lane}, f"t0-{pr}.log")))
    rc = {pr: p.wait() for pr, p in ps} if not T2 else {}
    print("T0 rc", rc, f"{time.time() - t0:.0f}s", flush=True)
    if any(rc.values()):
        (out / "ledger.json").write_text(json.dumps(dict(aborted="T0 fallo", rc=rc)))
        print("ABORT: T0 fallo; ver logs/t0-*.log", flush=True)
        return
    # T1: arranque en caliente, un carril por problema
    ledger, lanes_state = [], {}
    jobs = {pr: [(s, a) for s in seeds for a in ARMS] for pr in LANES}
    running = {}
    while any(jobs.values()) or running:
        el = time.time() - t0
        for pr, (job, p, st) in list(running.items()):
            r = p.poll()
            if r is not None or el >= HARD_CAP_S:
                if r is None:
                    p.kill()
                ledger.append(dict(problem=pr, seed=job[0], arm=job[1], start=st, end=el, rc=r, status="ok" if r == 0 else ("killed" if r is None else "failed")))
                print(f"[{el:6.0f}s] end {pr} {job} rc={r}", flush=True)
                del running[pr]
        for pr, lane in LANES.items():
            if pr not in running and jobs[pr]:
                if el >= CUTOFF_S:
                    for j in jobs[pr]:
                        ledger.append(dict(problem=pr, seed=j[0], arm=j[1], status="not_started_cutoff"))
                    jobs[pr] = []
                    continue
                s, arm = jobs[pr].pop(0)
                a = ["--problems", pr, "--arms", arm, "--seeds", str(s), "--adam", "10" if tiny else "1000", "--start-step", "4000", "--init-dir", str(bundle_states), "--init-key", "s004000",
                     "--pre-every", "5" if tiny else "500", "--eval-every", "5" if tiny else "250", "--no-wandb", "--out", str(out / ("t2" if T2 else "t1")), "--device", dev] + (["--no-damp"] if T2 else []) + smoke
                running[pr] = ((s, arm), run(a, {} if tiny else {"CUDA_VISIBLE_DEVICES": lane}, f"t1-{pr}-{arm}-{s}.log"), el)
                print(f"[{el:6.0f}s] start {pr} {(s, arm)}", flush=True)
        if any(jobs.values()) or running:
            time.sleep(1 if tiny else 15)
    (out / "ledger.json").write_text(json.dumps(dict(ledger=ledger, wall_seconds=time.time() - t0), indent=1))
    print("done", f"{time.time() - t0:.0f}s", [e["status"] for e in ledger], flush=True)


if __name__ == "__main__":
    main()
