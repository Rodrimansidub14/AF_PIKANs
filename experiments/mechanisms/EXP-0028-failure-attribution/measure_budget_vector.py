"""EXP-0028 fase 4: costo en CPU del Jacobiano del residuo vectorial (NS2d-CG, 8 terminos, 3 salidas) para el precondicionador de Gram.

Mide, para RBF16 float32 (receta de la campana): segundos por refresh (Jacobiano + eigh) por estructura y chunk, pico de memoria (working set), y
segundos por paso de Adam. No entrena, no lee referencia (NSProblem se crea sin su .dat). Escribe budget_vector.json.
Uso: python measure_budget_vector.py [--threads 2] [--full 1]
"""
import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import psutil
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gp = load(ROOT / "src/mechanisms/gram_preconditioning.py", "b28_gp")
pc = load(ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning/kaggle/dataset/pikan_campaign.py", "b28_pc")
rt = gp._residual_terms_module()
proc = psutil.Process()


def peak_mb():
    mi = proc.memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2 ** 20


def refresh_cost(arm, n_points, chunk, width, seed=1):
    prob = rt.install(pc.NSProblem.__new__(pc.NSProblem))
    model = prob.build("rbf", width, 11, seed).to(torch.float32)
    P = prob.pools(seed, torch.device("cpu"), torch.float32)
    pre = gp.GramPreconditioner(arm, n_points=n_points, chunk=chunk)
    pre.attach(model)
    t = time.perf_counter()
    d = pre.refresh(model, prob, P)
    return dict(arm=arm, n_points_per_key=n_points, points_total=d["n_points"], chunk=chunk, seconds=time.perf_counter() - t, peak_mb_so_far=peak_mb())


def adam_cost(width, steps=10, seed=1):
    prob = pc.NSProblem.__new__(pc.NSProblem)
    model = prob.build("rbf", width, 11, seed).to(torch.float32)
    P = prob.pools(seed, torch.device("cpu"), torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for k in range(steps + 2):
        if k == 2:
            t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        prob.loss(model, P)[0].backward()
        opt.step()
    return (time.perf_counter() - t) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--full", type=int, default=1, help="also one E refresh at the default N = 2048 per key")
    ap.add_argument("--out", default=str(HERE / "budget_vector.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = dict(problem="ns2d_cg", model=f"RBF{a.width} 2-{a.width}-{a.width}-3, 11 bases", dtype="float32", device="cpu", threads=a.threads,
               torch=torch.__version__, machine=platform.processor(), logical_cpus=psutil.cpu_count(),
               jacobian_columns_per_layer=None, notes=[], rows=[])
    prob = pc.NSProblem.__new__(pc.NSProblem)
    m = prob.build("rbf", a.width, 11, 1)
    out["jacobian_columns_per_layer"] = [int(l.coefficients.numel()) for l in m.layers]
    out["adam_seconds_per_step_full_batch"] = adam_cost(a.width)
    print("adam", out["adam_seconds_per_step_full_batch"], flush=True)
    # warm-up: torch.func pays ~15-20 s once per process on the first call; reported separately
    out["warmup_seconds"] = refresh_cost("E", 16, 8, a.width)["seconds"]
    print("warmup", out["warmup_seconds"], flush=True)
    for arm, n, chunk in (("E", 128, 4), ("E", 128, 8), ("E", 128, 16), ("J", 128, 8), ("I", 128, 8)):
        r = refresh_cost(arm, n, chunk, a.width)
        out["rows"].append(r)
        print(r, flush=True)
    if a.full:
        r = refresh_cost("E", 2048, 8, a.width)
        out["rows"].append(r)
        out["refresh_E_default_seconds"] = r["seconds"]
        print(r, flush=True)
    out["peak_working_set_mb"] = peak_mb()
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
