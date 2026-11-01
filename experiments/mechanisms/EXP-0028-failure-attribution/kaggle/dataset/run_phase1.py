"""EXP-0028 Fase 1 (+1b, +ampliacion v1.1): ejecutor de la descomposicion del mecanismo y de las palancas revividas, solo PIKAN (RBF-KAN).

Un JSON por corrida (`{problema}-{brazo}-seed{semilla}.json`), reanudable (si el JSON existe se salta). Arnes de 1-B/1-C SIN cambios: Adam 10 % del control
(Burgers 2000, Allen-Cahn y Fisher-KPP 4000), L-BFGS tope de 500 iteraciones, float32, recalculo del precondicionador cada 1000 pasos y al arrancar L-BFGS, N = 2048.

Problemas : ac (Allen-Cahn d=1e-3, RBF16), ac2e4 (Allen-Cahn d=2e-4, RBF16), burgers (RBF48), kpp (Fisher-KPP, RBF16).
Brazos    : B (control), F, J, E, I  [seeds 4001-4010]  | palancas R, C, W [seeds 4001-4005].  ID (identidad) solo para la prueba B == ID.
  F, J, E, I : GramPreconditioner (src/mechanisms/gram_preconditioning.py, EXP-0023), via el mismo camino que EXP-0023/pre_trainer.py.
  S          : muestreador autogobernado de EXP-0014 (self_gated_sampler, modo 'gated'): 32768 candidatos Sobol, gate lambda = clip(Spearman(|r|, ||grad u||), 0, 1), p = u/N + (1-u)(lambda w(|r|) + (1-lambda) w(s)),
               potencia 2, piso uniforme u = 0.5, con reposicion, mismo numero de puntos. Calendario del spec (calentamiento 2000, cada 1000, sobre un control de 20000/40000 pasos) ESCALADO a la
               convencion T1 (10 %): calentamiento 200 y cada 100 pasos (--s-warmup/--s-every; con los valores literales Burgers, Adam 2000, nunca remuestrea). L-BFGS sobre el pool uniforme inicial.
  R          : RAD de EXP-0025 (observer_v0.RadLever, Observer modo fixed:aR, K=500, 8192 candidatos, piso uniforme 0.5, presupuesto de puntos fijo).
  C          : regrid de centros ocultos, idea A de EXP-0024 (diseno D1 de t1_trainer: cuantiles puros una vez al 25 % de Adam).
  W          : foco por responsabilidad de EXP-0018, rama P (foco suave, beta = 1) en L-BFGS desde el MISMO estado final de Adam que B de esa semilla. Es un techo ORACULO:
               la responsabilidad A* usa el error real contra la referencia (etiqueta `oracle_ceiling`); no es un mecanismo practicable, se reporta aparte del resto.
Sin codigo ejecutable en este arnes (ver missing_arms.md): H (0016).

Por corrida se guardan: error al final de Adam (`primary_adam`) y al final de L-BFGS (`primary_final`) = Fase 1b, la perdida de entrenamiento en ambos puntos
(`loss_adam_end`, `loss_final`, sobre el pool vigente del brazo), reloj (`adam_seconds`, `lbfgs_seconds`, `total_seconds`) y semilla. Solo se puntua contra la referencia DESPUES de entrenar.
B guarda ademas el estado final de Adam (`*-B-seed*-adam.pt`, lo usa W) y la decision que tomaria el disparador de traspaso de 0016 sobre su historia de perdida (`handoff_probe`).

Uso: python run_phase1.py --problems ac,ac2e4,burgers,kpp --arms B,F,J,E,I,R,C,W --seeds 4001,4002 --out DIR [--device cuda:0] [--warmup-steps 50]
Solo humo en CPU: --adam N --lbfgs-iters M --pre-every K --pre-n n.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[2]))
EXP23 = ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning"
sys.path.insert(0, str(BUNDLE or EXP23 / "round1C"))
sys.path.insert(0, str(BUNDLE or EXP23))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import pre_trainer as pt  # noqa: E402  (EXP-0023; carga t1_trainer / sweep / campana / gram_preconditioning)
import arnes_round1c as a1c  # noqa: E402  (ronda 1-C; ac_d2e-4 y kpp)

pc, gp, sw, t1 = pt.pc, pt.gp, pt.sw, pt.t1
ob = _load((BUNDLE / "observer_v0.py") if BUNDLE else ROOT / "src/mechanisms/observer_v0.py", "observer_v0_exp0028")
sgs = _load((BUNDLE / "self_gated_sampler.py") if BUNDLE else ROOT / "src/mechanisms/self_gated_sampler.py", "self_gated_sampler_exp0028")
handoff_step = _load((BUNDLE / "stall_handoff.py") if BUNDLE else ROOT / "src/mechanisms/stall_handoff.py", "stall_handoff_exp0028").handoff_step   # solo para la sonda

PROBLEMS = {
    "ac": dict(label="Allen-Cahn d=1e-3", kind="pt", key="ac", adam=4000),
    "ac2e4": dict(label="Allen-Cahn d=2e-4", kind="r1c", key="ac_d2e-4", adam=4000),
    "burgers": dict(label="Burgers RBF48", kind="pt", key="burgers", adam=2000),
    "kpp": dict(label="Fisher-KPP", kind="r1c", key="kpp", adam=4000),
}
MAIN_ARMS = ("B", "F", "J", "E", "I")
LEVER_ARMS = ("S", "R", "C", "W")
ALL_ARMS = MAIN_ARMS + LEVER_ARMS
SEEDS_MAIN = tuple(range(4001, 4011))
SEEDS_LEVER = tuple(range(4001, 4006))
LBFGS_ITERS = 500
_rc = None


def make_problem(problem):
    cfg = PROBLEMS[problem]
    if cfg["kind"] == "pt":
        c = pt.PROBLEMS[cfg["key"]]
        return pc.make_problem(c["name"], str(c["ref"])), c["width"]
    return a1c.make_problem_1c(cfg["key"]), a1c.WIDTH[cfg["key"]]


def run_ceiling():
    """EXP-0018 (oraculo de responsabilidad). Carga perezosa: solo si se pide W."""
    global _rc
    if _rc is None:
        if BUNDLE:
            _rc = _load(BUNDLE / "run_ceiling.py", "run_ceiling_exp0028")
        else:
            sys.path.insert(0, str(ROOT))
            _rc = _load(ROOT / "experiments/mechanisms/EXP-0018-responsibility-ceiling/run_ceiling.py", "run_ceiling_exp0028")
    return _rc


def loss_at(prob, model, P, state):
    model.load_state_dict(state)
    return float(prob.loss(model, P)[0].detach())


def handoff_probe(hist, k_max):
    """Que decidiria el disparador de traspaso (EXP-0016, valores firmados) con la historia de perdida de B. Solo informativo; no cambia la corrida."""
    try:
        d = handoff_step([h["step"] for h in hist], [h["loss"] for h in hist], k_max)
        return dict(fired=bool(d["fired"]), k_t=int(d["k_t"]), reason=d["reason"], k_max=int(k_max))
    except Exception as e:  # k_max < k_min, historia corta, etc.
        return dict(fired=False, k_t=int(k_max), reason=f"{type(e).__name__}: {e}", k_max=int(k_max))


S_WARMUP, S_EVERY, S_CANDIDATES = 200, 100, 32768


def _structure(prob, model, X, chunk=2048):
    """s = ||grad u|| (norma sobre (x, t)) en X, por autograd del modelo; solo usa el modelo."""
    out = []
    for lo in range(0, X.shape[0], chunk):
        xt = X[lo:lo + chunk].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            u = prob.transform(xt, model(xt))
            g = torch.autograd.grad(u, xt, torch.ones_like(u))[0]
        out.append(g.detach().norm(dim=1).double().cpu().numpy())
    import numpy as np
    return np.concatenate(out)


def adam_with_sampler(prob, model, P, steps, seed, warmup, every, n_cand, log_every=100, sync=lambda: None, clock=time.perf_counter):
    """Mismo lazo que pt.adam_with_pre; tras `warmup` pasos y luego cada `every`, remuestrea P['residual'] con el muestreador autogobernado (EXP-0014). Devuelve tambien el pool inicial."""
    import numpy as np
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pool0 = P["residual"].clone()
    cand = torch.as_tensor(ob.heldout_points(prob.lo, prob.hi, n_cand, seed), dtype=pool0.dtype, device=pool0.device)
    cand_np = cand.double().cpu().numpy()
    hist, diag, status, step = [], [], "adam_complete", 0
    t0 = clock()
    for step in range(1, steps + 1):
        done = step - 1
        if done >= warmup and (done - warmup) % every == 0:
            ts = clock()
            r = ob.residual_at(prob, model, cand)
            s = _structure(prob, model, cand)
            smp = sgs.sample(cand_np, r, s, int(pool0.shape[0]), np.random.default_rng([seed, done, 14]), mode="gated")
            P["residual"] = torch.as_tensor(smp["points"], dtype=pool0.dtype, device=pool0.device)
            diag.append(dict(step=done, **{"lambda": smp["lambda"], "effective_candidates": smp["effective_candidates"], "unique_count": smp["unique_count"], "seconds": clock() - ts}))
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
    sync()
    return dict(hist=hist, sampler_diag=diag, seconds=clock() - t0, steps=step, status=status, pool0=pool0)


def run_one(problem, arm, seed, adam_steps, lbfgs_iters, out_dir, device="cpu", dtype="float32", pre_every=gp.EVERY, pre_n=gp.N_POINTS):
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    prob, width = make_problem(problem)
    model = prob.build("rbf", width, 11, seed).to(device=dev, dtype=dt)
    P = prob.pools(seed, dev, dt)
    pre, obs, W = None, None, None
    extra = {}
    t0 = time.perf_counter()
    if arm in ("B", "ID", "F", "J", "E", "I"):
        if arm != "B":
            pre = gp.GramPreconditioner(arm, every=pre_every, n_points=pre_n)
            pre.attach(model)
        A = pt.adam_with_pre(prob, model, P, adam_steps, pre, sync=sync)
    elif arm == "S":
        A = adam_with_sampler(prob, model, P, adam_steps, seed, S_WARMUP, S_EVERY, S_CANDIDATES if pre_n >= 1024 else 512, sync=sync)
        P["residual"] = A.pop("pool0")                     # L-BFGS y las perdidas finales: pool uniforme inicial (spec 0014)
        extra["sampler"] = dict(warmup=S_WARMUP, every=S_EVERY, lambdas=[d["lambda"] for d in A["sampler_diag"]], diag=A["sampler_diag"])
    elif arm == "R":
        cfg = ob.ObserverConfig(K=500, n_p=25)
        obs = ob.Observer([ob.RadLever(prob.lo, prob.hi, extra_n(pre_n))], prob.lo, prob.hi, ob.FreshHeldout(prob.lo, prob.hi, extra_n(pre_n), seed), seed, cfg, "fixed:aR")
        A = obs.run_adam(prob, model, P, adam_steps)
        extra["rad"] = dict(chosen=[d["chosen"] for d in A["decisions"]], n_resamples=sum(d["chosen"] == "aR" for d in A["decisions"]))
    elif arm == "C":
        spec = t1.DESIGNS["D1"]
        A = t1.adam_with_hook(prob, model, P, adam_steps, regrid=dict(steps=spec["steps"], mix=spec["mix"]), sync=sync)
        extra["regrid_diag"] = A["regrid_diag"]
    elif arm == "W":
        return run_w(prob, model, P, problem, seed, adam_steps, lbfgs_iters, out_dir, dev, dt, sync, device, dtype)
    else:
        raise ValueError(arm)
    adam_state = sw.clone_state(model)
    out = dict(problem=problem, arm=arm, seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=time.perf_counter() - t0, adam_status=A["status"],
               pre_diag=A.get("pre_diag", []), device=device, dtype=dtype, hist=A["hist"], scientific_accepted=False, **extra)
    if arm == "B":
        torch.save(adam_state, Path(out_dir) / f"{problem}-B-seed{seed}-adam.pt")
        out["handoff_probe"] = handoff_probe(A["hist"], adam_steps)
    if A["status"] != "adam_complete":
        return out
    if pre is not None and pre.due_lbfgs():
        d = pre.refresh(model, prob, P["residual"], None, sync=sync)
        d["step"] = "lbfgs_start"
        out["pre_diag"].append(d)
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_iters, chunk=50, snapshot_every=1e12, sync=sync)
    out.update(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], total_seconds=time.perf_counter() - t0,
               pre_recomputes=(pre.n_recomputes if pre else 0), pre_seconds=(pre.seconds if pre else 0.0))
    out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
    out["score_adam"] = sw.score_run(prob, model, dev, dt, adam_state)
    out["score_final"] = sw.score_run(prob, model, dev, dt, L["final_state"])
    out["primary_adam"], out["primary_final"] = sw.primary(out["score_adam"]), sw.primary(out["score_final"])
    out["loss_adam_end"] = loss_at(prob, model, P, adam_state)
    out["loss_final"] = loss_at(prob, model, P, L["final_state"])
    return out


def extra_n(pre_n):
    """Candidatos de RAD / puntos de H: 8192 firmado (0025). En humo (pre_n pequeno) se baja a 512 para que la prueba corra rapido."""
    return 8192 if pre_n >= 1024 else 512


def run_w(prob, model, P, problem, seed, adam_steps, lbfgs_iters, out_dir, dev, dt, sync, device, dtype):
    """Rama P de EXP-0018 desde el estado final de Adam de B (misma semilla). Si no esta guardado, entrena Adam sin mecanismo."""
    rc = run_ceiling()
    f = Path(out_dir) / f"{problem}-B-seed{seed}-adam.pt"
    if f.exists():
        state, src, adam_seconds = torch.load(f, map_location=dev, weights_only=False), "B_saved_state", None
        bj = Path(out_dir) / f"{problem}-B-seed{seed}.json"
        if bj.exists():
            adam_seconds = json.loads(bj.read_text()).get("adam_seconds")
    else:
        t0 = time.perf_counter()
        A = pt.adam_with_pre(prob, model, P, adam_steps, None, sync=sync)
        state, src, adam_seconds = sw.clone_state(model), "trained_here", time.perf_counter() - t0
        if A["status"] != "adam_complete":
            return dict(problem=problem, arm="W", seed=seed, adam_status=A["status"], adam_state_source=src, scientific_accepted=False)
    model.load_state_dict(state)
    pts, u_ref = rc.eval_grid(prob, problem)
    pts = pts.to(dev)
    import numpy as np
    with torch.no_grad():
        err_full = np.abs(prob.transform(pts, model(pts)).cpu().numpy().reshape(u_ref.shape) - u_ref).reshape(-1)
    idx = np.sort(np.random.default_rng(rc.SUB_SEED).choice(pts.shape[0], size=min(rc.N_SUB, pts.shape[0]), replace=False))
    A_star, O = rc.oracle_responsibility(model, pts[idx], err_full[idx])
    s_list, masks = rc.arm_transforms("P", A_star, O, seed)
    rc.rf.apply_scales(model, s_list)
    t0 = time.perf_counter()
    L = sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_iters, chunk=50, snapshot_every=1e12, sync=sync)
    rc.rf.release(model)
    final = sw.clone_state(model)
    out = dict(problem=problem, arm="W", seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=adam_seconds, adam_status="adam_complete", adam_state_source=src,
               lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], total_seconds=(adam_seconds or 0.0) + (time.perf_counter() - t0),
               device=device, dtype=dtype, label="oracle_ceiling", beta=rc.BETA, scientific_accepted=False)
    out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
    out["score_adam"] = sw.score_run(prob, model, dev, dt, state)
    out["score_final"] = sw.score_run(prob, model, dev, dt, final)
    out["primary_adam"], out["primary_final"] = sw.primary(out["score_adam"]), sw.primary(out["score_final"])
    out["loss_adam_end"] = loss_at(prob, model, P, state)
    out["loss_final"] = loss_at(prob, model, P, final)
    return out


def warmup(problem, device, dtype, arms, steps):
    """Corrida desechable por proceso (EXP-0016: sin calentamiento la primera corrida en GPU no reproduce a las siguientes). Usa otra semilla y no guarda nada."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for arm in ("B",) + tuple(a for a in arms if a in ("J", "E", "I", "S", "R", "C")):
            run_one(problem, arm, 1, steps, 20, td, device, dtype, pre_every=max(1, steps // 2), pre_n=64)
        if "W" in arms:
            run_one(problem, "W", 1, steps, 20, td, device, dtype, pre_every=steps, pre_n=64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="ac,ac2e4,burgers,kpp")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--seeds", default="", help="vacio = 4001-4010 para B,F,J,E,I y 4001-4005 para R,C,W")
    ap.add_argument("--adam", type=int, default=0, help="0 = receta T1 (Burgers 2000, resto 4000); otro valor solo para humo")
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_ITERS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pre-every", type=int, default=gp.EVERY)
    ap.add_argument("--pre-n", type=int, default=gp.N_POINTS)
    ap.add_argument("--s-warmup", type=int, default=S_WARMUP)
    ap.add_argument("--s-every", type=int, default=S_EVERY)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    globals().update(S_WARMUP=a.s_warmup, S_EVERY=a.s_every)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",")
    for problem in a.problems.split(","):
        steps = a.adam or PROBLEMS[problem]["adam"]
        if a.warmup_steps > 0:
            warmup(problem, a.device, a.dtype, arms, a.warmup_steps)
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS_MAIN)
        for s in seeds:
            for arm in arms:                                    # el orden de a.arms manda: B primero (W usa su estado de Adam)
                if not a.seeds and arm in LEVER_ARMS and s not in SEEDS_LEVER:
                    continue
                f = out / f"{problem}-{arm}-seed{s}.json"
                if f.exists():
                    continue
                r = run_one(problem, arm, s, steps, a.lbfgs_iters, out, a.device, a.dtype, a.pre_every, a.pre_n)
                f.write_text(json.dumps(r))
                print(problem, arm, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
