"""Tier 1 (CPU): humo de la receta + tiempos por paso para corregir el presupuesto del EXP-0016.

Por problema (Burgers RBF48 f32, Allen-Cahn RBF16 f32): 200 pasos de Adam, luego L-BFGS limitado por reloj (--lbfgs-seconds), con el mismo
cierre y la misma regla de meseta de la receta (pikan_campaign.py). Mide segundos por paso de Adam, segundos por iteracion de L-BFGS y
evaluaciones por iteracion. No lee la referencia (el score no se llama). Salida: timing_smoke.json.
"""
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[3]))
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
spec = importlib.util.spec_from_file_location("pikan_campaign", BL / "burgers-close-20261003/pikan_campaign.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("TIMING_THREADS", "4")))
CASES = {
    "burgers": dict(ref=BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz", arm="rbf", width=48, bases=11, seed=301),
    "ac": dict(ref=BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat", arm="rbf", width=16, bases=11, seed=601),
}
ADAM = int(os.environ.get("TIMING_ADAM", "200"))
LB_SEC = float(os.environ.get("TIMING_LBFGS_SECONDS", "20"))
out = {}
for name, c in CASES.items():
    prob = pc.make_problem(name, str(c["ref"]))
    model = prob.build(c["arm"], c["width"], c["bases"], c["seed"]).to(dtype=torch.float32)
    P = prob.pools(c["seed"], torch.device("cpu"), torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad(); prob.loss(model, P)[0].backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(ADAM):
        opt.zero_grad(set_to_none=True)
        l, _ = prob.loss(model, P)
        l.backward()
        opt.step()
    adam_s = time.perf_counter() - t0
    loss_adam = float(l.detach())
    opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=25, max_eval=125, history_size=100,
                             tolerance_grad=1e-8, tolerance_change=0.0, line_search_fn="strong_wolfe")
    evals = [0]

    def closure():
        evals[0] += 1
        opt2.zero_grad(set_to_none=True)
        l, _ = prob.loss(model, P)
        l.backward()
        return l

    t1, iters = time.perf_counter(), 0
    while time.perf_counter() - t1 < LB_SEC:
        b = opt2.state[opt2._params[0]].get("n_iter", 0)
        opt2.step(closure)
        iters += opt2.state[opt2._params[0]].get("n_iter", 0) - b
    lb_s = time.perf_counter() - t1
    loss_lb = float(prob.loss(model, P)[0].detach())
    out[name] = dict(adam_steps=ADAM, adam_seconds=adam_s, sec_per_adam_step=adam_s / ADAM, loss_after_adam=loss_adam,
                     lbfgs_seconds=lb_s, lbfgs_iterations=iters, lbfgs_evals=evals[0],
                     sec_per_lbfgs_iteration=lb_s / max(iters, 1), evals_per_iteration=evals[0] / max(iters, 1),
                     loss_after_lbfgs=loss_lb, finite=bool(math.isfinite(loss_lb)), threads=torch.get_num_threads())
    print(name, json.dumps(out[name]), flush=True)
(Path(__file__).parent / "timing_smoke.json").write_text(json.dumps(out, indent=1))
