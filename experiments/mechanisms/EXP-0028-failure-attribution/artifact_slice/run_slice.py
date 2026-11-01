"""EXP-0028, REBANADA con artefactos completos (Enmienda v1.6): Allen-Cahn d=1e-3, Burgers RBF48, Fisher-KPP, NS2d-CG y DFG; brazos B, F, J, E, I; semillas 4201, 4202 (verificadas: no usadas).
Reutiliza SIN modificar los ejecutores existentes (`run_phase1.py` para ac/kpp/burgers, `run_phase1c.py` para NS, `run_phase1c_dfg.py` para DFG): cada `run_one` se llama tal cual y el registro se engancha
por envoltorios temporales (ver artifacts_core.py). La receta es la RECETA MEDIA de budget_slice.md (Adam de M pasos con instantaneas, L-BFGS por iteraciones), no la corta ni la completa.

Receta media (M = pasos de Adam; L-BFGS 500 it. strong-Wolfe float32 salvo donde el ejecutor dice otra cosa):
  ns 10000 | dfg 6000 | ac 10000 | kpp 8000 | burgers 10000 (16 384 puntos como la receta completa de la linea base; el ejecutor de la fase 1 usa 8 192)
Calendario de instantaneas: COMPLETAS en {0, 2000, M/2, M} U {cada 5000} y al final de L-BFGS; LIVIANAS (error en la malla, c_eff, terminos, asentamiento) cada 1000 pasos y en 100/250 it. de L-BFGS.

Uso:  python run_slice.py --problem ns --arms B,F,J,E,I --seeds 4201,4202 --out DIR [--device cuda:0] [--warmup-steps 50]
      python run_slice.py --problem ns --check --out DIR [--device cuda:0]      # B == ID (con registro) y registro neutro (B con y sin registro identicos); escribe check_slice_<problema>.json
Humo en CPU: --adam N --lbfgs-iters K --pre-every E --pre-n n --smoke-pool S (marca smoke=true y levanta la guardia de semillas).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SRC = HERE if (HERE / "run_phase1.py").exists() else HERE.parent          # bundle plano o arbol del repositorio
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC))

import artifacts_core as ac  # noqa: E402

RECIPES = {
    "ns": dict(adam=10000, lbfgs=500, kind="ns"),
    "dfg": dict(adam=6000, lbfgs=500, kind="dfg"),
    "ac": dict(adam=10000, lbfgs=500, kind="scalar"),
    "kpp": dict(adam=8000, lbfgs=500, kind="scalar"),
    "burgers": dict(adam=10000, lbfgs=500, kind="scalar", n_col=16384),
}
ARMS = ("B", "F", "J", "E", "I")
SEEDS = (4201, 4202)
USED_SEEDS = set(range(4001, 4011)) | set(range(4101, 4106))
CAP_MB = ac.SIZE_CAP_MB


def full_schedule(M, smoke=False):
    s = {0, M // 2, M}
    if not smoke:
        s |= {2000} | set(range(5000, M, 5000))
    return sorted(x for x in s if 0 <= x <= M)


def _modules(problem):
    """Importa el ejecutor del problema (perezoso: cada uno carga sus propios datos)."""
    if RECIPES[problem]["kind"] == "scalar":
        import run_phase1 as m
    elif problem == "ns":
        import run_phase1c as m
    else:
        import run_phase1c_dfg as m
    return m


def data_files(problem, m):
    try:
        if problem == "dfg":
            return [m.DFG_REF]
        if problem == "ns":
            return [m.NS_REF]
        if problem == "kpp":
            return [Path(m.a1c.__file__).parent / "fisher_kpp_ref.npz"]
        return [m.pt.PROBLEMS[m.PROBLEMS[problem]["key"]]["ref"]]
    except Exception:  # noqa: BLE001
        return []


def code_files(m):
    names = ["run_phase1.py", "run_phase1c.py", "run_phase1c_dfg.py", "run_slice.py", "artifacts_core.py", "gram_preconditioning.py", "residual_terms.py", "sweep.py", "pre_trainer.py", "dfg.py"]
    out = [HERE / "run_slice.py", HERE / "artifacts_core.py"]
    for n in names:
        for d in (SRC, HERE):
            if (d / n).exists():
                out.append(d / n)
                break
    try:
        out.append(Path(m.gp.__file__))
    except Exception:  # noqa: BLE001
        pass
    return list(dict.fromkeys(out))


def run_job(problem, arm, seed, M, lbfgs_iters, out, device="cpu", pre_every=None, pre_n=None, smoke_pool=0, smoke=False, record=True, light_every=1000, wandb_on=True):
    """Una corrida con (record=True) o sin (record=False) artefactos. Devuelve (dict del ejecutor, resumen del registro | None)."""
    m = _modules(problem)
    kind = RECIPES[problem]["kind"]
    gp, sw = m.gp, m.sw
    pre_every = gp.EVERY if pre_every is None else pre_every
    pre_n = gp.N_POINTS if pre_n is None else pre_n
    tag = f"{problem}-{arm}-seed{seed}"
    rec = None
    if record:
        rec = ac.SliceRecorder(problem, arm, seed, out, tag, dict(problem=problem, arm=arm, seed=seed, adam=M, lbfgs_iters=lbfgs_iters, recipe=RECIPES[problem], pre_every=pre_every, pre_n=pre_n,
                                                                   smoke=smoke, smoke_pool=smoke_pool, device=device),
                               M, full_schedule(M, smoke), light_every=light_every, files=code_files(m) + data_files(problem, m), gp=gp, wandb_on=wandb_on)
    saved = []

    def patch(obj, name, new):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, new)

    def adapter_for(prob):
        if kind == "scalar":
            return ac.make_adapter(problem, prob, ob=m.ob)
        if problem == "ns":
            return ac.make_adapter("ns", prob, ns_ref=prob.ref)
        return ac.make_adapter("dfg", prob, dfg_ref=m._fem())

    if rec is not None:
        if kind == "scalar":
            orig_adam = m.pt.adam_with_pre

            def adam_w(prob, model, P, steps, pre=None, *a, **k):
                rec.attach(prob, model, P, adapter_for(prob), pre)
                prob.loss = rec.wrap_loss(prob.loss)
                try:
                    A = orig_adam(prob, model, P, steps, pre, *a, **k)
                finally:
                    prob.__dict__.pop("loss", None)
                rec.end_adam()
                return A
            patch(m.pt, "adam_with_pre", adam_w)
            if problem == "burgers" and RECIPES[problem].get("n_col"):
                orig_mp = m.make_problem

                def mp(p):
                    prob, width = orig_mp(p)
                    prob.n_col = RECIPES[problem]["n_col"] if not smoke else getattr(prob, "n_col", 8192)
                    return prob, width
                patch(m, "make_problem", mp)
        elif problem == "ns":
            orig_adam = m.adam_with_pre_vec

            def adam_w(prob, model, P, steps, pre=None, *a, **k):
                rec.attach(prob, model, P, adapter_for(prob), pre)
                prob.loss = rec.wrap_loss(prob.loss)
                try:
                    A = orig_adam(prob, model, P, steps, pre, *a, **k)
                finally:
                    prob.__dict__.pop("loss", None)
                rec.end_adam()
                return A
            patch(m, "adam_with_pre_vec", adam_w)
        else:
            orig_adam, orig_dfg_loss = m.adam_dfg, m.dfg_loss

            def adam_w(prob, model, P, steps, seed_, dev, pre=None, *a, **k):
                rec.attach(prob, model, P, adapter_for(prob), pre)
                m.dfg_loss = rec.wrap_loss(orig_dfg_loss)
                try:
                    A = orig_adam(prob, model, P, steps, seed_, dev, pre, *a, **k)
                finally:
                    m.dfg_loss = orig_dfg_loss
                rec.end_adam()
                return A
            patch(m, "adam_dfg", adam_w)
        patch(sw, "lbfgs_stage", ac.make_lbfgs_stage(sw, rec))
    elif problem == "burgers" and RECIPES[problem].get("n_col") and not smoke:       # sin registro: mismo cambio de puntos, para que la comparacion sea limpia
        orig_mp = m.make_problem

        def mp(p):
            prob, width = orig_mp(p)
            prob.n_col = RECIPES[problem]["n_col"]
            return prob, width
        patch(m, "make_problem", mp)
    t0 = time.perf_counter()
    try:
        if kind == "scalar":
            r = m.run_one(problem, arm, seed, M, lbfgs_iters, Path(out), device, "float32", pre_every, pre_n)
        elif problem == "ns":
            r = m.run_one(arm, seed, M, lbfgs_iters, device, "float32", pre_every, pre_n, smoke_pool)
        else:
            r = m.run_one(arm, seed, M, lbfgs_iters, device, "float32", pre_every, pre_n, smoke_pool)
    finally:
        for obj, name, old in reversed(saved):
            setattr(obj, name, old)
        if rec is not None:
            rec.close()
    r["slice"] = dict(recipe="medium", M=M, lbfgs_iters=lbfgs_iters, full_steps=full_schedule(M, smoke), recorded=bool(record), wall_seconds=time.perf_counter() - t0,
                      note="hist[*].seconds y adam_seconds incluyen el tiempo de las instantaneas (snapshot_seconds)")
    summ = rec.finalize(extra=dict(primary_adam=r.get("primary_adam"), primary_final=r.get("primary_final"))) if rec is not None else None
    if summ is not None:
        r["slice"]["artifacts"] = summ
    return r, summ


def check(problem, out, device, adam, a):
    """B == ID con registro, y registro neutro: B con registro == B sin registro, ambos bit a bit (hist, perdidas, puntuaciones, iteraciones de L-BFGS). Sin errores de registro."""
    out = Path(out)
    kw = dict(device=device, pre_every=a.pre_every or max(1, adam // 2), pre_n=a.pre_n or 64, smoke_pool=a.smoke_pool, smoke=True, wandb_on=False)
    lb = a.lbfgs_iters or 5
    runs = {}
    for name, arm, record in (("B_rec", "B", True), ("ID_rec", "ID", True), ("B_plain", "B", False)):
        sub = out / f"check_{name}"
        sub.mkdir(parents=True, exist_ok=True)
        runs[name] = run_job(problem, arm, 4201, adam, lb, sub, record=record, **kw)
    keys = ("loss_adam_end", "loss_final", "score_adam", "score_final", "lbfgs_iterations")

    def same(x, y):
        return [h["loss"] for h in x["hist"]] == [h["loss"] for h in y["hist"]] and all(x.get(k) == y.get(k) for k in keys)
    res = dict(problem=problem, device=device, adam=adam, B_rec_equals_ID_rec=same(runs["B_rec"][0], runs["ID_rec"][0]), B_rec_equals_B_plain=same(runs["B_rec"][0], runs["B_plain"][0]),
               recorder_errors={k: (v[1] or {}).get("errors") for k, v in runs.items() if v[1]}, artifacts={k: (v[1] or {}).get("megabytes") for k, v in runs.items() if v[1]})
    res["pass"] = bool(res["B_rec_equals_ID_rec"] and res["B_rec_equals_B_plain"] and not any(res["recorder_errors"].values()))
    (out / f"check_slice_{problem}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(RECIPES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--adam", type=int, default=0, help="solo humo/check: sustituye M")
    ap.add_argument("--lbfgs-iters", type=int, default=0)
    ap.add_argument("--pre-every", type=int, default=0)
    ap.add_argument("--pre-n", type=int, default=0)
    ap.add_argument("--smoke-pool", type=int, default=0)
    ap.add_argument("--light-every", type=int, default=1000)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = RECIPES[a.problem]
    smoke = bool(a.adam or a.lbfgs_iters or a.pre_every or a.pre_n or a.smoke_pool)
    M = a.adam or rec["adam"]
    if a.check:
        if a.warmup_steps > 0:                               # la 1.a corrida de un proceso en GPU no reproduce a las siguientes (EXP-0016): se calienta antes de comparar
            m0 = _modules(a.problem)
            if rec["kind"] == "scalar":
                m0.warmup(a.problem, a.device, "float32", ["B"], a.warmup_steps)
            else:
                for arm in ("B", "ID"):
                    m0.warmup(arm, a.device, "float32", a.warmup_steps)
        r = check(a.problem, out, a.device, a.adam or 20, a)
        print(json.dumps(r), flush=True)
        sys.exit(0 if r["pass"] else 1)
    seeds = [int(s) for s in a.seeds.split(",")]
    if not smoke and not set(seeds) <= set(SEEDS):
        sys.exit(f"semillas {seeds}: la rebanada usa {SEEDS} (4001-4010 y 4101-4105 ya estan usadas); --adam etc. levantan la guardia solo para humo")
    arms = a.arms.split(",")
    if not set(arms) <= set(ARMS):
        sys.exit(f"brazos {arms} fuera de {ARMS}")
    m = _modules(a.problem)
    if a.warmup_steps > 0:                                   # corrida desechable por proceso (EXP-0016); sin registro, otra semilla, nada se guarda
        if rec["kind"] == "scalar":
            m.warmup(a.problem, a.device, "float32", arms, a.warmup_steps)
        else:
            for arm in arms:
                m.warmup(arm, a.device, "float32", a.warmup_steps)
    for s in seeds:
        for arm in arms:
            f = out / f"{a.problem}-{arm}-seed{s}.json"
            if f.exists() and (out / f"{a.problem}-{arm}-seed{s}-artifacts.npz").exists():
                continue
            r, summ = run_job(a.problem, arm, s, M, a.lbfgs_iters or rec["lbfgs"], out, a.device, a.pre_every or None, a.pre_n or None, a.smoke_pool, smoke, True, a.light_every, not a.no_wandb)
            r["smoke"] = smoke
            f.write_text(json.dumps(r, default=str))
            print(a.problem, arm, s, f"adam {r.get('primary_adam')} final {r.get('primary_final')} {r.get('total_seconds', 0):.0f}s art {summ['megabytes']:.1f}MB snaps {summ['n_snapshots']} err {summ['n_errors']} wandb {summ['wandb']}", flush=True)


if __name__ == "__main__":
    main()
