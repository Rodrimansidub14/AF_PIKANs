"""EXP-0025, bloque (T1/V): brazos B, R, E, O, Orand (O-) y Om (O-m) sobre Allen-Cahn d=1e-3 ('ac', RBF16, Adam 4000) y Burgers ('burgers', RBF48, Adam 2000), L-BFGS 500,
con el arnes y la receta de 1-B/1-C. R = fixed:aR y E = fixed:aG (variante E, fijada por 1-C) dentro de observer_v0 (mismo camino de codigo que O); O = probe; Orand = elige al azar
uniforme entre {a0,aR,aG}; Om = al azar con las frecuencias que mostro O en ese problema (--probs-from). B = Adam plano (control de la misma sesion).
El observador solo gobierna Adam; el L-BFGS arranca con el P vigente (sin recalculo extra). Valores del spec: K=500, n_p=25, delta=0.05, gamma=0.25, rejilla 16x16, 5 %, piso 0.5, H_d de 8192.
Uso: python observer_block.py --problem ac --arms B,R,E,O,Orand --seeds 2001 --out DIR [--probs-from DIR para Om]. Los flags K/n-p/pre-n/h-n son SOLO para humo en CPU."""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
ROOT = HERE.parents[2]
sys.path.insert(0, str(BUNDLE or ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ob = _load((BUNDLE / "observer_v0.py") if BUNDLE else ROOT / "src/mechanisms/observer_v0.py", "observer_v0_block")
import pre_trainer as pt  # noqa: E402
pc, gp, sw = pt.pc, pt.gp, pt.sw
ADAM = {"ac": 4000, "burgers": 2000}
ARMS = ("B", "R", "E", "O", "Orand", "Om")


def probs_from(dirs, problem):
    """Frecuencias de accion de O en `problem` (todas las semillas y decisiones de los JSON de O de esas carpetas)."""
    cnt, tot = {"a0": 0, "aR": 0, "aG": 0}, 0
    for d in dirs:
        for f in Path(d).glob(f"{problem}-O-seed*.json"):
            for c in json.loads(f.read_text())["observer"]["chosen"]:
                cnt[c] += 1
                tot += 1
    if tot == 0:
        raise SystemExit(f"sin corridas de O para {problem} en {dirs}")
    return {k: v / tot for k, v in cnt.items()}


def run_one(problem, arm, seed, adam_steps, lbfgs_iters, device="cpu", dtype="float32", cfg=None, probs=None):
    spec = pt.PROBLEMS[problem]
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob = pc.make_problem(spec["name"], str(spec["ref"]))
    model = prob.build("rbf", spec["width"], 11, seed).to(device=dev, dtype=dt)
    P = prob.pools(seed, dev, dt)
    cfg = cfg or {}
    t0 = time.perf_counter()
    obs = None
    if arm == "B":
        A = pt.adam_with_pre(prob, model, P, adam_steps, None, sync=sync)
    else:
        lo, hi = prob.lo, prob.hi
        pre = gp.GramPreconditioner("E", n_points=cfg.get("pre_n", 2048))
        pre.attach(model)
        levers = [ob.RadLever(lo, hi, cfg.get("h_n", 8192)), ob.GramLever(pre, "E")]
        mode = {"R": "fixed:aR", "E": "fixed:aG", "O": "probe", "Orand": "random", "Om": "matched"}[arm]
        ocfg = ob.ObserverConfig(K=cfg.get("K", 500), n_p=cfg.get("n_p", 25))
        obs = ob.Observer(levers, lo, hi, ob.FreshHeldout(lo, hi, cfg.get("h_n", 8192), seed), seed, ocfg, mode, probs=probs)
        A = obs.run_adam(prob, model, P, adam_steps)
    adam_state = sw.clone_state(model)
    out = dict(problem=problem, arm=arm, seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=time.perf_counter() - t0, adam_status=A["status"],
               device=device, dtype=dtype, hist=A["hist"], scientific_accepted=False, probs=probs)
    if obs is not None:
        out["observer"] = dict(chosen=[d["chosen"] for d in A["decisions"]], improvement=[d["improvement"] for d in A["decisions"]],
                               worse_cells=[d["worse_cells"] for d in A["decisions"]], null_reason=[d["null_reason"] for d in A["decisions"]],
                               h_block=[d["h_block"] for d in A["decisions"]], extra_steps=A["extra_steps"], probe_seconds=A["probe_seconds"], inert=A["inert"])
    if A["status"] != "adam_complete":
        return out
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_iters, chunk=50, snapshot_every=1e12, sync=sync)
    out.update(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], total_seconds=time.perf_counter() - t0)
    out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
    out["score_adam"] = sw.score_run(prob, model, dev, dt, adam_state)
    out["score_final"] = sw.score_run(prob, model, dev, dt, L["final_state"])
    out["primary_adam"], out["primary_final"] = sw.primary(out["score_adam"]), sw.primary(out["score_final"])
    try:   # guarda de region / H-OBS-3: error por region (5 % de mayor |u_x| de la referencia), solo al puntuar, despues de todo el entrenamiento
        import numpy as np
        for tag, st in (("adam", adam_state), ("final", L["final_state"])):
            model.load_state_dict(st)
            xx, tt = np.meshgrid(prob.x, prob.t, indexing="xy")        # misma malla (nt, nx) que `prob.score`
            xt = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=dt, device=dev)
            with torch.no_grad():
                pred = prob.transform(xt, model(xt)).double().cpu().numpy().reshape(prob.u.shape)
            ux = np.abs(np.gradient(prob.u, prob.x, axis=1))
            top = ux >= np.quantile(ux, 0.95)
            e2 = (pred - prob.u) ** 2
            out[f"region_{tag}"] = dict(l2re_top=float(np.sqrt(e2[top].sum() / (prob.u[top] ** 2).sum())), l2re_rest=float(np.sqrt(e2[~top].sum() / (prob.u[~top] ** 2).sum())))
    except Exception as e:
        out["region_error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(ADAM))
    ap.add_argument("--arms", default="B,R,E,O,Orand")
    ap.add_argument("--seeds", default="2001")
    ap.add_argument("--adam", type=int, default=0)
    ap.add_argument("--lbfgs-iters", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probs-from", action="append", default=[])
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--n-p", type=int, default=25)
    ap.add_argument("--pre-n", type=int, default=2048)
    ap.add_argument("--h-n", type=int, default=8192)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    steps = a.adam or ADAM[a.problem]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(K=a.K, n_p=a.n_p, pre_n=a.pre_n, h_n=a.h_n)
    for s in map(int, a.seeds.split(",")):
        for arm in a.arms.split(","):
            f = out / f"{a.problem}-{arm}-seed{s}.json"
            if f.exists():
                continue
            probs = probs_from(a.probs_from, a.problem) if arm == "Om" else None
            r = run_one(a.problem, arm, s, steps, a.lbfgs_iters, a.device, a.dtype, cfg, probs)
            f.write_text(json.dumps(r))
            print(a.problem, arm, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
