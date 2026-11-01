"""EXP-0024 nivel T0: ajuste supervisado sin PDE (arnes del paso 0 del EXP-0015) con un mecanismo opcional de reubicacion de centros ocultos.
Con `regrid=None` el camino es identico, linea por linea, a `fit_step0.fit` (prueba de identidad en tests/test_wind_tunnel_t0.py).
Uso: python t0_harness.py --problem allen_cahn --designs D1,D2 --seeds 1,2,3,4,5 [--adam 1000 --lbfgs 500 --ntrain 4000]"""
import argparse, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
S0 = ROOT / "experiments/mechanisms/EXP-0015-layer0-basis-refinement/step0/fit_step0.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fs = _load(S0, "fit_step0")
rg = _load(HERE / "regrid.py", "wt_regrid")

# Libro de diseños de la ronda 1 (cada uno se anota en registry/designs.jsonl al correrse)
DESIGNS = {
    "CTRL": dict(desc="control: sin reubicacion", steps=[], mix=1.0),
    "D1": dict(desc="reubicar l1 y l2 por cuantiles de activacion una vez al 25 % de Adam", steps=[0.25], mix=1.0),
    "D2": dict(desc="como D1 pero mezcla 0.5 cuantiles / 0.5 uniforme", steps=[0.25], mix=0.5),
    "D3": dict(desc="reubicar por cuantiles cuatro veces (10, 30, 50, 70 % de Adam)", steps=[0.10, 0.30, 0.50, 0.70], mix=1.0),
    "D4": dict(desc="reubicar por cuantiles una vez al 5 % de Adam", steps=[0.05], mix=1.0),
}


def fit_t0(P, arm, seed, adam_steps, lbfgs_iters, n_train, regrid=None, log_every=500):
    """Copia de fit_step0.fit con un gancho de reubicacion en la fase de Adam. regrid = dict(steps=[pasos absolutos], mix=...) o None."""
    torch.manual_seed(seed)
    cs, ws = fs.arm_layer0(P, arm)
    net = fs.Net(P, cs, ws)
    X = torch.as_tensor(P["pts"])
    Y = torch.as_tensor(P["fields"])
    scale = Y.std(0, keepdim=True)
    if P["constraint"]:
        scale = torch.ones_like(scale)
    from scipy.spatial import cKDTree
    pts = P["pts"]
    sc = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
    _, nb = cKDTree(sc).query(sc, k=7)
    g = np.zeros(len(pts))
    for c in range(P["fields"].shape[1]):
        f = P["fields"][:, c] / (P["fields"][:, c].std() + 1e-300)
        d = np.linalg.norm(sc[nb[:, 1:]] - sc[:, None], axis=-1)
        g += (np.abs(f[nb[:, 1:]] - f[:, None]) / np.maximum(d, 1e-12)).max(1)
    prob = 0.5 / len(pts) + 0.5 * g / g.sum()
    idx = np.random.default_rng(1000 + seed).choice(len(pts), size=min(n_train, len(pts)), replace=False, p=prob)
    Xt, Yt = X[idx], Y[idx]
    w = 1.0 / (Yt.std(0, keepdim=True) ** 2)
    loss_fn = lambda: (((net(Xt, scale) - Yt) ** 2) * w).mean()
    curve, diag = [], []
    t0 = time.time()

    def evaluate():
        with torch.no_grad():
            pred = torch.cat([net(X[i:i + 20000], scale) for i in range(0, len(X), 20000)]).numpy()
        comps = {n: fs.rel_l2(pred[:, j], P["fields"][:, j]) for j, n in enumerate(P["names"])}
        return comps, fs.rel_l2(pred, P["fields"])

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    at = set(int(round(f * adam_steps)) for f in (regrid["steps"] if regrid else []))
    for step in range(1, adam_steps + 1):
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        opt.step()
        if step in at:
            s = 2 * (Xt - net.lo) / (net.hi - net.lo) - 1
            d = rg.regrid_net(net, s, mix=regrid["mix"])
            d["step"] = step
            diag.append(d)
            for p_ in (net.l1.coefficients, net.l2.coefficients):
                opt.state[p_].clear()
        if step % log_every == 0 or step == adam_steps:
            comps, flat = evaluate()
            curve.append(dict(stage="adam", step=step, t=time.time() - t0, loss=float(L.detach()), **{f"l2_{k}": v for k, v in comps.items()}))
    adam_comps, _ = evaluate()
    lb = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=25, history_size=100, line_search_fn="strong_wolfe", tolerance_grad=0.0, tolerance_change=0.0)

    def closure():
        lb.zero_grad()
        L = loss_fn()
        L.backward()
        return L

    for blk in range(lbfgs_iters // 25):
        L = lb.step(closure)
        comps, flat = evaluate()
        curve.append(dict(stage="lbfgs", step=adam_steps + 25 * (blk + 1), t=time.time() - t0, loss=float(L.detach()), **{f"l2_{k}": v for k, v in comps.items()}))
        if not math.isfinite(float(L.detach())):
            break
    comps, flat = evaluate()
    return dict(arm=arm, seed=seed, adam=adam_comps, final=comps, seconds=time.time() - t0, params=sum(p.numel() for p in net.parameters()), curve=curve, regrid_diag=diag)


def ledger(entry):
    with open(ROOT / "registry/designs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--designs", default="CTRL,D1")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--adam", type=int, default=1000)
    ap.add_argument("--lbfgs", type=int, default=500)
    ap.add_argument("--ntrain", type=int, default=4000)
    ap.add_argument("--out", default=str(HERE / "results/t0"))
    a = ap.parse_args()
    P = fs.load_problem(a.problem)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for dname in a.designs.split(","):
        spec = DESIGNS[dname]
        for s in map(int, a.seeds.split(",")):
            f = out / f"{a.problem}-{dname}-seed{s}-a{a.adam}-l{a.lbfgs}-n{a.ntrain}.json"
            if f.exists():
                continue
            rg_spec = None if not spec["steps"] else dict(steps=spec["steps"], mix=spec["mix"])
            r = fit_t0(P, "C", s, a.adam, a.lbfgs, a.ntrain, regrid=rg_spec)
            r.update(problem=a.problem, design=dname, adam_steps=a.adam, lbfgs_iters=a.lbfgs, ntrain=a.ntrain)
            f.write_text(json.dumps(r))
            ledger(dict(design=dname, level="T0", date="2026-10-20", description=spec["desc"], problem=a.problem, seed=s, budget=dict(adam=a.adam, lbfgs=a.lbfgs, ntrain=a.ntrain),
                        final_l2=r["final"], adam_l2=r["adam"], file=str(f.relative_to(ROOT))))
            print(a.problem, dname, s, {k: f"{v:.2e}" for k, v in r["final"].items()}, f"{r['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
