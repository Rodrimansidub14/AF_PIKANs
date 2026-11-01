"""Etapa 0 del plan de recursos de piezo (PLAN_RECURSOS.md): medir en la GPU real antes de gastar horas.
Mide, para converso y directo (float64, receta del paper, brazos B, E, J):
  1. s/paso de Adam (sin el recálculo) y s por recálculo, con la utilización de GPU muestreada con nvidia-smi;
  2. concurrencia: 1, 2 y 3 procesos en una GPU (y 2 por GPU en las dos GPU si hay dos): rendimiento agregado;
  3. CUDA graphs: el paso de entrenamiento (pérdida + backward + Adam capturable) capturado en un grafo frente al modo normal, con comprobación de
     que la pérdida final coincide (misma semilla, mismos pasos); si la captura falla, se registra el error.
No entrena ningún modelo completo (unos 150 pasos por medición). Escribe stage0_results.json. Uso: python stage0.py --out DIR [--device cuda:0] [--steps 150]
Humo en CPU: python stage0.py --out DIR --device cpu --smoke (pocos pasos, pool reducido; omite CUDA graphs y nvidia-smi)."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_piezo as rp  # noqa: E402

PROBLEMS = ("converse", "direct")
SEED = 20260728


# ------------------------------------------------------------------ utilización de GPU
class Sampler:
    def __init__(self, period=0.5):
        self.period, self.rows, self._stop, self._t = period, [], threading.Event(), None

    def _loop(self):
        while not self._stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                vals = [[int(x) for x in line.split(",")] for line in r.stdout.strip().splitlines() if line.strip()]
                self.rows.append((time.time(), vals))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)

    def summary(self):
        if not self.rows:
            return None
        per = {}
        for _, vals in self.rows:
            for idx, util, mem in vals:
                d = per.setdefault(idx, dict(util=[], mem=[]))
                d["util"].append(util)
                d["mem"].append(mem)
        return {str(i): dict(util_mean=sum(d["util"]) / len(d["util"]), util_max=max(d["util"]), mem_max_mb=max(d["mem"]), n=len(d["util"])) for i, d in per.items()}


# ------------------------------------------------------------------ trabajador: s/paso de un (problema, brazo)
def worker(problem, arm, steps, device, smoke):
    dev = torch.device(device)
    dt = torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = rp.make_problem(problem, 32 if smoke else 0)
    model = rp.build(prob, "B", SEED, dt).to(dev)
    pre = None
    if arm != "B":
        pre = rp.gp.GramPreconditioner(arm, every=10 ** 9, n_points=64 if smoke else rp.N_PER_KEY, chunk=2 if dev.type == "cpu" else rp.PRE_CHUNK)
        pre.attach(model)
    rp.adam_piezo(prob, rp.build(prob, "B", SEED, dt).to(dev), SEED, 5, dev, dt, None, eval_every=10 ** 9, sync=sync)     # calentamiento del proceso
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    sync()
    t0 = time.perf_counter()
    A = rp.adam_piezo(prob, model, SEED, steps, dev, dt, pre, eval_every=10 ** 9, sync=sync)
    sync()
    total = time.perf_counter() - t0
    rec = pre.seconds if pre else 0.0
    return dict(problem=problem, arm=arm, steps=steps, total_s=total, recompute_s=rec, s_per_step=(total - rec) / steps, peak_mb=(torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else None),
                status=A["status"])


def spawn(problem, arm, steps, device_index, smoke, out_file):
    env = dict(os.environ)
    if device_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    cmd = [sys.executable, str(HERE / "stage0.py"), "--worker", "--problem", problem, "--arm", arm, "--steps", str(steps), "--worker-out", str(out_file), "--device", "cpu" if smoke else "cuda:0"]
    if smoke:
        cmd.append("--smoke")
    return subprocess.Popen(cmd, env=env, cwd=str(HERE))


def run_group(jobs, steps, smoke, tag, out_dir):
    """jobs = [(problem, arm, gpu_index)] concurrentes. Devuelve filas por trabajo, utilización y rendimiento agregado."""
    procs = []
    with Sampler() as smp:
        t0 = time.perf_counter()
        for i, (problem, arm, gpu) in enumerate(jobs):
            f = Path(out_dir) / f"w_{tag}_{i}.json"
            procs.append((spawn(problem, arm, steps, gpu, smoke, f), f))
        for p, _ in procs:
            p.wait()
        wall = time.perf_counter() - t0
    rows = []
    for p, f in procs:
        rows.append(json.loads(f.read_text()) if f.exists() else dict(error=f"worker rc={p.returncode} sin salida"))
    ok = [r for r in rows if "s_per_step" in r]
    agg = (sum(r["steps"] for r in ok) / wall) if ok else None            # pasos totales por segundo de pared (incluye arranque)
    steady = (sum(1.0 / r["s_per_step"] for r in ok)) if ok else None     # pasos por segundo sumando los trabajadores en régimen
    return dict(tag=tag, n_workers=len(jobs), wall_s=wall, rows=rows, steps_per_s_steady=steady, steps_per_s_wall=agg, gpu=smp.summary())


# ------------------------------------------------------------------ CUDA graphs
def graph_test(problem, n=100):
    dev = torch.device("cuda:0")
    dt = torch.float64
    prob = rp.make_problem(problem, 0)
    P = prob.pools(SEED, dev, dt, 0)
    res = dict(problem=problem, n=n)

    def make():
        m = rp.build(prob, "B", SEED, dt).to(dev)
        return m, torch.optim.Adam(m.parameters(), lr=rp.LR, capturable=True)

    def step(m, o):
        o.zero_grad(set_to_none=True)
        loss, _ = prob.loss(m, P)
        loss.backward()
        o.step()
        return loss

    try:
        me, oe = make()
        for _ in range(3):
            step(me, oe)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            le = step(me, oe)
        torch.cuda.synchronize()
        res["eager_s_per_step"] = (time.perf_counter() - t) / n
        res["eager_final_loss"] = float(le.detach())
    except Exception as e:                                                  # noqa: BLE001
        res["eager_error"] = f"{type(e).__name__}: {e}"[:400]
        return res
    try:
        mg, og = make()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step(mg, og)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        og.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            loss_static, _ = prob.loss(mg, P)
            loss_static.backward()
            og.step()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            g.replay()
        torch.cuda.synchronize()
        res["graph_s_per_step"] = (time.perf_counter() - t) / n
        res["graph_final_loss"] = float(loss_static.detach())
        res["speedup"] = res["eager_s_per_step"] / res["graph_s_per_step"]
        a, b = res["eager_final_loss"], res["graph_final_loss"]
        res["rel_diff_final_loss"] = abs(a - b) / max(abs(a), 1e-300)
    except Exception as e:                                                  # noqa: BLE001
        res["graph_error"] = f"{type(e).__name__}: {e}"[:600]
    return res


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stage0_out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--problem", default="converse")
    ap.add_argument("--arm", default="B")
    ap.add_argument("--worker-out", default="")
    a = ap.parse_args()
    if a.worker:
        r = worker(a.problem, a.arm, a.steps, a.device, a.smoke)
        Path(a.worker_out).write_text(json.dumps(r))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smoke = a.smoke
    steps = 3 if smoke else a.steps
    n_gpu = 0 if smoke else torch.cuda.device_count()
    res = dict(torch=torch.__version__, smoke=smoke, steps=steps, n_gpu=n_gpu, gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)], t_start=time.time())
    path = out / "stage0_results.json"

    def save():
        path.write_text(json.dumps(res, indent=1))

    gpu0 = None if smoke else 0
    # 1. s/paso y recálculo, un proceso, por problema y brazo
    res["single"] = []
    for problem in PROBLEMS:
        for arm in ("B", "E", "J"):
            res["single"].append(run_group([(problem, arm, gpu0)], steps, smoke, f"single-{problem}-{arm}", out))
            print("single", problem, arm, json.dumps(res["single"][-1]["rows"]), flush=True)
            save()
    # 2. concurrencia (converso, brazo B)
    res["concurrency"] = []
    for k in (1, 2, 3):
        res["concurrency"].append(run_group([("converse", "B", gpu0)] * k, steps, smoke, f"conc-{k}", out))
        print("concurrency", k, res["concurrency"][-1]["steps_per_s_steady"], flush=True)
        save()
    if n_gpu >= 2:
        res["concurrency"].append(run_group([("converse", "B", 0), ("converse", "B", 0), ("converse", "B", 1), ("converse", "B", 1)], steps, smoke, "conc-2x2", out))
        print("concurrency 2x2", res["concurrency"][-1]["steps_per_s_steady"], flush=True)
        save()
    # 3. CUDA graphs
    if not smoke:
        res["cuda_graphs"] = []
        for p_ in PROBLEMS:
            try:
                res["cuda_graphs"].append(graph_test(p_))
            except Exception as e:                                              # noqa: BLE001
                res["cuda_graphs"].append(dict(problem=p_, error=f"{type(e).__name__}: {e}"[:400]))
            save()
        print("cuda_graphs", json.dumps(res["cuda_graphs"]), flush=True)
    res["t_end"] = time.time()
    res["total_s"] = res["t_end"] - res["t_start"]
    save()


if __name__ == "__main__":
    main()
