"""EXP-0028 piezo en Modal (pago por segundo). NO lanza nada por sí solo.
Hallazgo de la etapa 0 en H100 (2026-10-25): la GPU está al 20 % con un proceso (limitada por lanzamientos, no por cálculo) y 3 procesos a la vez rinden 3 veces
(13.25 pasos/s frente a 4.37). Por eso cada contenedor corre K procesos concurrentes en la misma H100 (--k, 4 por defecto).
  modal run piezo/modal_piezo.py --mode smoke                       # 4 corridas de 300 pasos (≈ 0.5 USD): comprueba artefactos y el despacho en la H100
  modal run piezo/modal_piezo.py --mode run --upto 2 --k 4 --containers 2   # semilla del paper (F, J) + semillas 20260729-30 (B,F,E,J): 20 corridas
  modal run piezo/modal_piezo.py --mode run --upto 3 ...            # + semillas 20260731-32
  modal volume get af-pikans-exp0028-piezo piezo piezo_modal_results         # traer resultados
Reanudable (salta lo ya escrito). W&B queda offline en el volumen (sin claves)."""
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
KIT = HERE / "cloud_kit"
GPU = os.environ.get("X28_MODAL_GPU", "H100")

app = modal.App("af-pikans-exp0028-piezo")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "numpy", "scipy", "wandb")
         .add_local_dir(str(KIT), "/kit"))
vol = modal.Volume.from_name("af-pikans-exp0028-piezo", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/vol": vol})
def stage0():
    out = "/vol/piezo"
    os.makedirs(out, exist_ok=True)
    rc = subprocess.run([sys.executable, "/kit/stage0.py", "--out", f"{out}/stage0"], env=dict(os.environ, BUNDLE_DIR="/kit"), cwd="/kit").returncode
    vol.commit()
    return rc


@app.function(image=image, gpu=GPU, cpu=6.0, timeout=10 * 3600, volumes={"/vol": vol})
def run_batch(jobs, k=4, steps=0, out_sub="piezo"):
    """jobs = [(problema, semilla, brazo)]; k procesos concurrentes en la misma GPU; commit del volumen tras cada corrida."""
    out = f"/vol/{out_sub}"
    os.makedirs(f"{out}/provenance", exist_ok=True)
    for f in ("EXP-0029_spec.md", "DISENO_FINAL.md"):                       # la firma y el diseño viajan con los resultados
        if os.path.exists(f"/kit/{f}"):
            shutil.copyfile(f"/kit/{f}", f"{out}/provenance/{f}")
    lock = threading.Lock()

    def one(job):
        problem, seed, arm = job
        if os.path.exists(f"{out}/piezo-{problem}-{arm}-seed{seed}.json"):
            return f"skip {problem} {arm} {seed}"
        cmd = [sys.executable, "/kit/run_piezo.py", "--problems", problem, "--arms", arm, "--seeds", str(seed), "--out", out, "--device", "cuda:0", "--warmup-steps", "50"]
        if steps:
            cmd += ["--adam", str(steps), "--eval-every", "100"]
        os.makedirs(f"{out}/logs", exist_ok=True)
        with open(f"{out}/logs/{problem}-seed{seed}-{arm}.log", "w") as log:
            extra = dict(X28_HIST_EVERY="1", X28_STATE_EVERY="1000", X28_FIELD_EVERY="2500", X28_SAVE_B="1") if out_sub == "piezo_exp0029" else {}     # bloque final: registro por paso, estados cada 1000, bloques GN completos
            rc = subprocess.run(cmd, env=dict(os.environ, BUNDLE_DIR="/kit", X28_K=str(k), **extra), cwd="/kit", stdout=log, stderr=subprocess.STDOUT).returncode
        with lock:
            vol.commit()
        return f"rc={rc} {problem} {arm} {seed}"

    stop = threading.Event()

    def heartbeat():                          # publica los logs parciales cada 5 min (para medir la velocidad sin esperar a que termine la primera corrida)
        while not stop.wait(300):
            with lock:
                vol.commit()

    threading.Thread(target=heartbeat, daemon=True).start()
    with ThreadPoolExecutor(max_workers=k) as ex:
        res = list(ex.map(one, jobs))
    stop.set()
    vol.commit()
    return res


@app.local_entrypoint()
def main(mode: str = "stage0", upto: int = 2, k: int = 4, containers: int = 1, max_jobs: int = 0):
    if mode == "stage0":
        print("stage0 rc:", stage0.remote())
        return
    if mode == "smoke":
        jobs = [("converse", 20260728, "F"), ("converse", 20260728, "J"), ("direct", 20260728, "F"), ("direct", 20260728, "J")]
        for r in run_batch.remote(jobs, k=4, steps=300, out_sub="piezo_smoke"):
            print(r, flush=True)
        return
    if mode == "exp0029":                     # bloque final firmado: M (semillas 28-30, dos problemas) + B, E, M (semillas 31-32, converso); 3 contenedores x 4 procesos
        S = lambda n: 20260700 + n
        batches = [[("converse", S(28), "M"), ("converse", S(29), "M"), ("converse", S(30), "M"), ("direct", S(28), "M")],
                   [("direct", S(29), "M"), ("direct", S(30), "M"), ("converse", S(31), "M"), ("converse", S(32), "M")],
                   [("converse", S(31), "B"), ("converse", S(32), "B"), ("converse", S(31), "E"), ("converse", S(32), "E")]]
        print(sum(map(len, batches)), "trabajos en", len(batches), "contenedores; GPU", GPU, "; k =", k, flush=True)
        for res in run_batch.starmap([(b, k, 0, "piezo_exp0029") for b in batches]):
            for r in res:
                print(r, flush=True)
        return
    sys.path.insert(0, str(KIT))
    from cloud_runner import job_list
    jobs = [(p, s, a) for p, s, a, _ in job_list(upto)]
    if max_jobs:
        jobs = jobs[:max_jobs]
    batches = [jobs[i::containers] for i in range(containers) if jobs[i::containers]]
    print(len(jobs), "trabajos en", len(batches), "contenedores; GPU", GPU, "; k =", k, flush=True)
    for res in run_batch.starmap([(b, k, 0, "piezo") for b in batches]):
        for r in res:
            print(r, flush=True)
