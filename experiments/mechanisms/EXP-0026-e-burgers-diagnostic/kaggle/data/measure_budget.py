"""EXP-0023, prueba previa 6 (parte CPU): costo de recalcular los bloques y sobrecosto por paso de la reparametrización, en el modelo de la campaña.

Mide, por problema y brazo: segundos de un `refresh` (N = 2 048), segundos por paso de Adam con y sin reparametrización. No entrena ni lee la referencia.
Uso: python measure_budget.py [--device cpu] [--reps 3] [--out results/budget_<device>.json]
"""
import argparse, json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pre_trainer as pt


def measure(problem, device, dtype, reps, steps=30):
    cfg = pt.PROBLEMS[problem]
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    res = {}
    for arm in pt.ARMS:
        prob = pt.pc.make_problem(cfg["name"], str(cfg["ref"]))
        model = prob.build("rbf", cfg["width"], 11, 1).to(device=dev, dtype=dt)
        P = prob.pools(1, dev, dt)
        pre = None
        if arm != "B":
            pre = pt.gp.GramPreconditioner(arm)
            pre.attach(model)
        pt.adam_with_pre(prob, model, P, 5, None, sync=sync)                                   # calentamiento
        t = pt.adam_with_pre(prob, model, P, steps, None, sync=sync)["seconds"] / steps
        entry = dict(adam_s_per_step=t)
        if pre is not None and arm in pt.gp.RECOMPUTED:
            times = []
            for _ in range(reps):
                times.append(pre.refresh(model, prob, P["residual"], None, sync=sync)["seconds"])
            entry.update(refresh_seconds=times, refresh_min=min(times))
        res[arm] = entry
        print(problem, arm, entry, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--problems", default="ac,burgers")
    ap.add_argument("--out", default=str(HERE / "results/budget_cpu.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = {p: measure(p, a.device, a.dtype, a.reps) for p in a.problems.split(",")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
