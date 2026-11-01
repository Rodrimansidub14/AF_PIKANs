"""EXP-0024 nivel T1: corrida corta de la PIKAN-PDE (pérdida de la PDE, receta del control del EXP-0016) con el gancho de reubicación de centros ocultos.

Convención T1 (fijada aquí): Adam = 10 % de los pasos del control (Burgers 2 000, Allen-Cahn 4 000); L-BFGS con tope de ITERACIONES (no de segundos) para que sea
determinista y comparable entre brazos; se reporta el reloj aparte (guardia: reloj del brazo <= 1.2 x el del control). El costo de reubicar entra en el reloj.
El entrenamiento solo usa prob.loss; la puntuación contra la referencia ocurre DESPUÉS, como en sweep.py. Con regrid=None el camino es idéntico al de
sweep.adam_stage + sweep.lbfgs_stage (prueba en tests/test_wind_tunnel_t1.py).
Uso: python t1_trainer.py --problem ac --designs CTRL,D1 --seeds 1,2,3 [--adam 4000 --lbfgs-iters 500 --device cpu --dtype float32]
"""
import argparse, importlib.util, json, math, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# En Kaggle (BUNDLE_DIR) todo vive junto en el dataset; en el repo, en sus rutas de siempre.
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
sw = _load((BUNDLE or EXP16) / "sweep.py", "exp0016_sweep_t1")
rg = _load((BUNDLE or HERE) / "regrid.py", "wt_regrid_t1")
pc = sw.load_campaign((BUNDLE / "pikan_campaign.py") if BUNDLE else BL / "burgers-close-20261003/pikan_campaign.py")
PROBLEMS = {
    "burgers": dict(ref=(BUNDLE / "burgers_coleHopf_1025x129.npz") if BUNDLE else BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz",
                    width=48, adam_full=20000, name="burgers"),
    "ac": dict(ref=(BUNDLE / "Allen_Cahn.mat") if BUNDLE else BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat", width=16, adam_full=40000, name="ac"),
}
DESIGNS = {
    "CTRL": dict(desc="control: sin reubicacion", steps=[], mix=1.0),
    "D1": dict(desc="reubicar capas ocultas por cuantiles de activacion una vez al 25 % de Adam", steps=[0.25], mix=1.0),
    "D2": dict(desc="como D1 con mezcla 0.5 cuantiles / 0.5 uniforme", steps=[0.25], mix=0.5),
    "D3": dict(desc="cuantiles puros cuatro veces (10, 30, 50, 70 % de Adam)", steps=[0.10, 0.30, 0.50, 0.70], mix=1.0),
}


def adam_with_hook(prob, model, P, steps, regrid=None, log_every=100, sync=lambda: None, clock=time.perf_counter):
    """Copia de sweep.adam_stage (mismo lazo) con un gancho de reubicación. Solo usa prob.loss y las activaciones de los puntos de colocación."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    at = set(int(round(f * steps)) for f in (regrid["steps"] if regrid else []))
    hist, diag, status, step = [], [], "adam_complete", 0
    t0 = clock()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        total, _ = prob.loss(model, P)
        total.backward()
        opt.step()
        if step % log_every == 0:
            val = float(total.detach())
            if not math.isfinite(val):
                status = "nonfinite_loss"
                break
            hist.append({"step": step, "loss": val, "seconds": clock() - t0})
        if step in at:
            d = rg.regrid_kan(model, P["residual"].detach(), mix=regrid["mix"])
            d["step"] = step
            diag.append(d)
            for layer in (model.layers[1], model.layers[2]):
                opt.state[layer.coefficients].clear()
    sync()
    return dict(hist=hist, regrid_diag=diag, seconds=clock() - t0, steps=step, status=status)


def run_one(problem, design, seed, adam_steps, lbfgs_iters, device="cpu", dtype="float32", n_col=0):
    cfg = PROBLEMS[problem]
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = pc.make_problem(cfg["name"] if cfg["name"] != "ac" else "ac", str(cfg["ref"]))
    if n_col and hasattr(prob, "n_col"):
        prob.n_col = n_col
    model = prob.build("rbf", cfg["width"], 11, seed).to(device=dev, dtype=dt)
    P = prob.pools(seed, dev, dt)
    spec = DESIGNS[design]
    reg = None if not spec["steps"] else dict(steps=spec["steps"], mix=spec["mix"])
    t0 = time.perf_counter()
    A = adam_with_hook(prob, model, P, adam_steps, regrid=reg, sync=sync)
    adam_state = sw.clone_state(model)
    t_adam = time.perf_counter() - t0
    out = dict(problem=problem, design=design, seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=t_adam, adam_status=A["status"],
               regrid_diag=A["regrid_diag"], device=device, dtype=dtype, hist=A["hist"], scientific_accepted=False)
    if A["status"] != "adam_complete":
        return out
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_iters, chunk=50, snapshot_every=1e12, sync=sync)
    out.update(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], total_seconds=time.perf_counter() - t0)
    # puntuación contra la referencia: DESPUÉS de entrenar
    out["score_adam"] = sw.score_run(prob, model, dev, dt, adam_state)
    out["score_final"] = sw.score_run(prob, model, dev, dt, L["final_state"])
    out["primary_adam"] = sw.primary(out["score_adam"])
    out["primary_final"] = sw.primary(out["score_final"])
    return out


def warmup_t1(problem, device, dtype="float32", steps=50):
    """Corrida desechable (mismas formas) antes de la primera corrida del proceso: sweep.warmup (Adam, registro, L-BFGS, puntuacion) y una reubicacion de
    centros, que usa cuantiles y resolucion lineal en doble precision. Ver EXP-0016: sin calentamiento la primera corrida de un proceso en GPU no es reproducible."""
    from argparse import Namespace
    cfg = PROBLEMS[problem]
    args = Namespace(problem=cfg["name"], ref=str(cfg["ref"]), n_col=0, seed=0, arm="rbf", width=cfg["width"], bases=11, device=device, dtype=dtype, lbfgs_chunk=50)
    info = sw.warmup(pc, args, steps=steps)
    dev = torch.device(device)
    prob = pc.make_problem(cfg["name"], str(cfg["ref"]))
    model = prob.build("rbf", cfg["width"], 11, 1).to(device=dev, dtype=torch.float32 if dtype == "float32" else torch.float64)
    P = prob.pools(1, dev, torch.float32 if dtype == "float32" else torch.float64)
    adam_with_hook(prob, model, P, 20)
    rg.regrid_kan(model, P["residual"].detach(), mix=0.5)
    return info


def ledger(entry, path=None):
    with open(path or (ROOT / "registry/designs.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(PROBLEMS))
    ap.add_argument("--designs", default="CTRL,D1")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--adam", type=int, default=0, help="pasos de Adam (0 = 10 %% del control)")
    ap.add_argument("--lbfgs-iters", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results/t1"))
    ap.add_argument("--no-ledger", action="store_true", help="no anotar en registry/designs.jsonl (pruebas y corridas de verificacion)")
    ap.add_argument("--local-ledger", action="store_true", help="anotar en designs.jsonl dentro de --out (Kaggle: el registro del repo no existe alli)")
    ap.add_argument("--warmup-steps", type=int, default=0, help="pasos del calentamiento previo (0 = sin; en GPU usar 50)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    steps = a.adam or PROBLEMS[a.problem]["adam_full"] // 10
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.warmup_steps > 0:
        print(json.dumps({"warmup": warmup_t1(a.problem, a.device, a.dtype, a.warmup_steps)}), flush=True)
    for d in a.designs.split(","):
        for s in map(int, a.seeds.split(",")):
            f = out / f"{a.problem}-{d}-seed{s}-a{steps}-l{a.lbfgs_iters}.json"
            if f.exists():
                continue
            r = run_one(a.problem, d, s, steps, a.lbfgs_iters, a.device, a.dtype)
            f.write_text(json.dumps(r))
            if not a.no_ledger:
              ledger(dict(design=d, level="T1", date="2026-10-20", description=DESIGNS[d]["desc"], problem=a.problem, seed=s,
                        budget=dict(adam=steps, lbfgs_iters=a.lbfgs_iters, device=a.device, dtype=a.dtype),
                        primary_adam=r.get("primary_adam"), primary_final=r.get("primary_final"), seconds=r.get("total_seconds"),
                        file=str(f.relative_to(ROOT)) if not BUNDLE else f.name), path=(out / "designs.jsonl") if a.local_ledger else None)
            print(a.problem, d, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
