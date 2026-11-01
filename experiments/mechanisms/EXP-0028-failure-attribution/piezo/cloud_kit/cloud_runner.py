"""Despachador portátil de piezo (EXP-0028) para una máquina con una o varias GPU (Voltage Park, RunPod, Colab...). No usa rutas de Kaggle.
Orden de prioridad (se puede parar en cualquier punto; es REANUDABLE: salta lo que ya está escrito en --out):
  0) stage0 (medición de 10 minutos, una vez, en la GPU 0)  -> stage0_results.json
  1) semilla 20260728: F y J del converso y del directo (B y E ya corrieron en Kaggle)
  2) semillas 20260729-30: B, F, E, J en los dos problemas
  3) semillas 20260731-32: B, F, E, J
  4) brazo I (5 semillas, bloque de tamaño 2) y 5) MLP-paper (semilla 20260728)
Uso: python cloud_runner.py --out OUT [--gpus 0,1,2,3] [--only-stage0] [--upto 3] [--smoke]
Los resultados quedan en OUT/piezo-<problema>-<brazo>-seed<semilla>.json (+ artefactos .npz y W&B offline). La clave de W&B NO va en la máquina: sincronizar luego en local."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBLEMS = ("converse", "direct")
SEEDS = (20260728, 20260729, 20260730, 20260731, 20260732)


def job_list(upto):
    jobs = []
    jobs += [(p, SEEDS[0], a, 1) for p in PROBLEMS for a in ("F", "J")]
    for stage, seeds in ((2, SEEDS[1:3]), (3, SEEDS[3:5])):
        jobs += [(p, s, a, stage) for s in seeds for p in PROBLEMS for a in ("B", "F", "E", "J")]
    jobs += [(p, s, "I", 4) for s in SEEDS for p in PROBLEMS]
    jobs += [(p, SEEDS[0], "MLP", 5) for p in PROBLEMS]
    return [j for j in jobs if j[3] <= upto]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--only-stage0", action="store_true")
    ap.add_argument("--upto", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="humo en CPU: pocos pasos, pool reducido")
    ap.add_argument("--procs-per-gpu", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BUNDLE_DIR=str(HERE))
    py = sys.executable
    if not a.smoke and not (out / "stage0" / "stage0_results.json").exists():
        print("etapa 0 (medicion) en la GPU 0 ...", flush=True)
        subprocess.run([py, str(HERE / "stage0.py"), "--out", str(out / "stage0")], env=dict(env, CUDA_VISIBLE_DEVICES=a.gpus.split(",")[0]), cwd=str(HERE))
    if a.only_stage0:
        return
    gpus = [g for g in a.gpus.split(",") if g != ""] if not a.smoke else ["cpu"]
    slots = [(g, i) for g in gpus for i in range(a.procs_per_gpu)]
    queue = [j for j in job_list(a.upto) if not (out / f"piezo-{j[0]}-{j[2]}-seed{j[1]}.json").exists()]
    print(f"{len(queue)} trabajos pendientes, {len(slots)} ranuras", flush=True)
    running, ledger, t0 = {}, [], time.time()
    while queue or running:
        for slot, (job, proc, start) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                ledger.append(dict(job=job[:3], slot=slot, seconds=time.time() - start, rc=rc))
                print(f"[{time.time() - t0:8.0f}s] fin {job[:3]} rc={rc}", flush=True)
                (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
                del running[slot]
        for slot in slots:
            if slot not in running and queue:
                job = queue.pop(0)
                p, s, arm, _ = job
                cmd = [py, str(HERE / "run_piezo.py"), "--problems", p, "--arms", arm, "--seeds", str(s), "--out", str(out)]
                if a.smoke:
                    cmd += ["--device", "cpu", "--adam", "10", "--eval-every", "5", "--pre-every", "5", "--pre-n", "64", "--smoke-pool", "16"]
                else:
                    cmd += ["--device", "cuda:0", "--warmup-steps", "50"]
                e = dict(env, **({} if a.smoke else {"CUDA_VISIBLE_DEVICES": slot[0]}), **({"OMP_NUM_THREADS": "2"} if a.smoke else {}))
                log = open(out / "logs" / f"{p}-seed{s}-{arm}.log", "w")
                running[slot] = (job, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=e, cwd=str(HERE)), time.time())
                print(f"[{time.time() - t0:8.0f}s] inicio {job[:3]} en GPU {slot[0]}", flush=True)
        time.sleep(2 if a.smoke else 20)
    print("terminado", len(ledger), "trabajos", flush=True)


if __name__ == "__main__":
    main()
