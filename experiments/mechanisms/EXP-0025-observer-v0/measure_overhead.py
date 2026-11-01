"""EXP-0025, Tier 0 prueba 6 (borrador): sobrecosto del observador en Fisher-KPP (CPU, pocos pasos). Compara Adam plano (B) con el observador (O) y con O- (sin sondeo).
Reporta la proporción de pasos extra y de reloj. NO mide error: es solo costo. Uso: python measure_overhead.py [--steps 400 --K 100 --n-p 5 --seed 1]"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning/round1C"))
sys.path.insert(0, str(ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ob = load(ROOT / "src/mechanisms/observer_v0.py", "observer_v0_cli")
import arnes_round1c as ar  # noqa: E402
gp = ar.gp


def fresh(seed):
    prob = ar.make_problem_1c("kpp")
    model = prob.build("rbf", 16, 11, seed).to(torch.float32)
    return prob, model, prob.pools(seed, torch.device("cpu"), torch.float32)


def plain_adam(steps, seed):
    prob, model, P = fresh(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = prob.loss(model, P)
        loss.backward()
        opt.step()
    return time.perf_counter() - t0, float(loss)


def observed(steps, seed, K, n_p, mode):
    prob, model, P = fresh(seed)
    pre = gp.GramPreconditioner("E", n_points=512)
    pre.attach(model)
    lo, hi = prob.lo, prob.hi
    cfg = ob.ObserverConfig(K=K, n_p=n_p)
    obs = ob.Observer([ob.RadLever(lo, hi, 4096), ob.GramLever(pre, "E")], lo, hi, ob.heldout_points(lo, hi, 8192, seed), seed, cfg, mode)
    out = obs.run_adam(prob, model, P, steps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--n-p", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(2)
    tB, lB = plain_adam(a.steps, a.seed)
    res = dict(problem="fisher_kpp_1d", steps=a.steps, K=a.K, n_p=a.n_p, B_seconds=tB, B_final_loss=lB)
    for mode in ("probe", "random"):
        o = observed(a.steps, a.seed, a.K, a.n_p, mode)
        res[mode] = dict(seconds=o["seconds"], clock_ratio_vs_B=o["seconds"] / tB, extra_steps=o["extra_steps"], extra_steps_ratio=o["extra_steps"] / a.steps,
                         probe_seconds=o["probe_seconds"], chosen=[d["chosen"] for d in o["decisions"]], status=o["status"], inert=o["inert"])
    res["spec_expected_extra_steps_ratio"] = 3 * a.n_p / a.K
    res["spec_clock_target"] = "<= 1.5x B"
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
