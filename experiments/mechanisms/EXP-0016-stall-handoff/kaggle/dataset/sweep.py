"""EXP-0016, etapa A: barrido de traspaso Adam -> L-BFGS (techo del oraculo) para Burgers y Allen-Cahn.

Por semilla se entrena Adam UNA vez con la receta del control, se guardan los pesos en los pasos de la malla de traspaso (fracciones de los
pasos de Adam) y, desde cada uno, corre L-BFGS con la receta del control (pikan_campaign.py: strong-Wolfe, chunk 50, parada por meseta o tope).
El tope de L-BFGS del traspaso k es  T_total - t_adam(k),  con T_total = t_adam(k_max) + tope del control, de modo que k = 1.00 es exactamente
el control y ningun traspaso puede exceder su reloj total.

Separacion entrenamiento/evaluacion (amenaza de validez central): `adam_stage` y `lbfgs_stage` solo reciben `prob.loss`; no llaman a `score` ni
conocen la referencia. La puntuacion contra la referencia ocurre en `score_run`, DESPUES de todo el entrenamiento de la semilla.
Tambien se calcula, sin referencia, el paso k_t que daria la senal de estancamiento (stall_handoff) sobre la historia de pérdida del control.

Heat (control negativo) NO esta cubierto por este ejecutor: usa el codigo de la campana S2 y necesita un adaptador aparte.

Uso (GPU, solo tras la firma):
  python sweep.py --problem burgers --ref <npz> --campaign-py <pikan_campaign.py> --seeds 301 302 --out OUT --width 48 --bases 11 \
      --adam-steps 20000 --lbfgs-seconds 600
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for _p in HERE.parents:
    if (_p / "src" / "mechanisms" / "stall_handoff.py").exists():
        sys.path.insert(0, str(_p))
        break
else:
    sys.path.insert(0, str(HERE))
try:
    from src.mechanisms.settling_focus import settling_step
    from src.mechanisms.stall_handoff import handoff_step
except ImportError:  # bundle: los modulos copiados junto a este archivo
    from settling_focus import settling_step
    from stall_handoff import handoff_step

GRID = (0.25, 0.40, 0.60, 0.80, 1.00)
LBFGS_PARAMS = dict(lr=1.0, history_size=100, tolerance_grad=1e-8, tolerance_change=0.0, line_search_fn="strong_wolfe")


def load_campaign(path):
    spec = importlib.util.spec_from_file_location("pikan_campaign", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def state_sha256(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def layer0_view(model):
    """Coeficientes (O, D, G) y base lineal (O, D) de la capa 0 de una RBF-PIKAN, como arreglos numpy."""
    layer = model.layers[0]
    return layer.coefficients.detach().cpu().numpy().copy(), layer.base.detach().cpu().numpy().copy()


def residual_per_point(pc, prob, model, P):
    """Residuo de la PDE en cada punto del pool fijo (la misma expresion que `prob.loss`, sin la media). No usa la referencia."""
    xt = P["residual"].detach().requires_grad_(True)
    with torch.enable_grad():
        u = prob.transform(xt, model(xt))
        g = pc.grad(u, xt)
        uxx = pc.grad(g[:, 0:1], xt)[:, 0:1]
        if prob.name == "burgers_pinnacle_1d":
            res = g[:, 1:2] + u * g[:, 0:1] - prob.nu * uxx
        elif prob.name == "allen_cahn":
            res = g[:, 1:2] - 0.001 * uxx - 5.0 * (u - u.pow(3))
        else:
            raise ValueError(f"sin residuo por punto para {prob.name}")
    return res.detach().reshape(-1)


class TraceRecorder:
    """Registro cada `every` pasos de Adam: coeficientes de la capa 0 y r^2 por punto del pool. Solo lee el modelo y la perdida."""

    def __init__(self, pc, prob, P, every):
        self.pc, self.prob, self.P, self.every = pc, prob, P, every
        self.steps, self.coeffs, self.base, self.resid2 = [], [], [], []

    def __call__(self, step, model):
        c, b = layer0_view(model)
        r = residual_per_point(self.pc, self.prob, model, self.P)
        self.steps.append(step)
        self.coeffs.append(c)
        self.base.append(b)
        self.resid2.append(r.square().float().cpu().numpy())

    def arrays(self, model):
        import numpy as np

        layer = model.layers[0]
        return dict(steps=np.asarray(self.steps, dtype=np.int64), coeffs=np.stack(self.coeffs), base=np.stack(self.base),
                    resid2=np.stack(self.resid2), points=self.P["residual"].detach().cpu().numpy(),
                    centers=layer.centers.detach().cpu().numpy(), width=np.asarray(float(layer.width)),
                    lo=model.lo.detach().cpu().numpy(), hi=model.hi.detach().cpu().numpy())


def grid_steps(steps, fractions=GRID):
    return {f: int(round(f * steps)) for f in fractions}


def adam_stage(prob, model, P, steps, ckpt_steps=(), log_every=100, seconds_cap=0.0, sync=lambda: None, clock=time.perf_counter,
               record=None, record_every=0):
    """Mismo lazo que `run_worker` (pikan_campaign.py); ademas guarda los pesos en `ckpt_steps` y, si se pide, llama a
    `record(step, model)` cada `record_every` pasos (y en el paso 0). El registro no toca el optimizador ni el RNG y su tiempo no
    cuenta en el reloj. Solo usa prob.loss."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    wanted = set(ckpt_steps)
    hist, ckpts, ckpt_seconds = [], {}, {}
    status, skipped, step = "adam_complete", 0.0, 0
    t0 = clock()
    if record is not None and record_every:
        tc = clock()
        record(0, model)
        skipped += clock() - tc
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        total, _terms = prob.loss(model, P)
        total.backward()
        opt.step()
        if step % log_every == 0:
            val = float(total.detach())
            if not math.isfinite(val):
                status = "nonfinite_loss"
                break
            hist.append({"step": step, "loss": val, "seconds": clock() - t0 - skipped})
        if record is not None and record_every and step % record_every == 0:
            sync()
            tc = clock()
            record(step, model)
            skipped += clock() - tc
        if step in wanted:
            sync()
            ckpt_seconds[step] = clock() - t0 - skipped
            tc = clock()
            ckpts[step] = clone_state(model)
            skipped += clock() - tc
        if seconds_cap and clock() - t0 - skipped > seconds_cap:
            status = "adam_time_cap"
            break
    sync()
    return dict(hist=hist, ckpts=ckpts, ckpt_seconds=ckpt_seconds, seconds=clock() - t0 - skipped, steps=step, status=status,
                record_seconds=skipped)


def lbfgs_stage(prob, model, P, seconds_cap, max_iter=15000, chunk=50, snapshot_every=30.0, sync=lambda: None, clock=time.perf_counter):
    """Mismo lazo y misma regla de parada que `run_worker`; ademas guarda instantaneas cada `snapshot_every` s para puntuar despues.

    Solo usa prob.loss. Devuelve el estado aceptado (el mismo que deja el control) y las instantaneas.
    """
    opt2 = torch.optim.LBFGS(model.parameters(), max_iter=chunk, max_eval=chunk * 5, **LBFGS_PARAMS)
    accepted = clone_state(model)
    last = float(prob.loss(model, P)[0].detach())
    t1, iters, status, plateau = clock(), 0, "lbfgs_time_cap", 0
    traj = [{"t": 0.0, "iters": 0, "loss": last}]
    snaps = []
    next_snap = snapshot_every

    def closure():
        opt2.zero_grad(set_to_none=True)
        loss, _ = prob.loss(model, P)
        loss.backward()
        return loss

    while True:
        if clock() - t1 >= seconds_cap:
            status = "lbfgs_time_cap"
            break
        if iters >= max_iter:
            status = "lbfgs_iter_cap"
            break
        before = opt2.state[opt2._params[0]].get("n_iter", 0)
        try:
            opt2.step(closure)
        except Exception as exc:  # noqa: BLE001 - registrado; el estado se restaura abajo
            status = f"lbfgs_exception:{type(exc).__name__}"
            break
        after = opt2.state[opt2._params[0]].get("n_iter", 0)
        cur = float(prob.loss(model, P)[0].detach())
        if not math.isfinite(cur) or cur > last * 1.5:
            model.load_state_dict(accepted)
            status = "lbfgs_rejected_nonfinite_or_worse"
            break
        improved = (last - cur) / max(abs(last), 1e-30)
        accepted = clone_state(model)
        last = cur
        iters += after - before
        elapsed = clock() - t1
        traj.append({"t": elapsed, "iters": iters, "loss": cur})
        if elapsed >= next_snap:
            snaps.append((elapsed, iters, accepted))
            next_snap = elapsed + snapshot_every
        if after == before:
            status = "optimizer_stopped_early"
            break
        plateau = plateau + 1 if improved < 1e-9 else 0
        if plateau >= 3:
            status = "lbfgs_plateau"
            break
    model.load_state_dict(accepted)
    sync()
    seconds = clock() - t1
    snaps.append((seconds, iters, accepted))
    return dict(status=status, iterations=iters, seconds=seconds, traj=traj, snaps=snaps, final_state=accepted)


def score_run(prob, model, device, dtype, state):
    """Unico punto que toca la referencia. Se llama despues de todo el entrenamiento."""
    model.load_state_dict(state)
    return prob.score(model, device, dtype)


def warmup(pc, args, steps=50, clock=time.perf_counter):
    """Corrida desechable que usa los mismos nucleos que una semilla real (misma red, mismo pool, Adam, registro, un bloque de L-BFGS y puntuacion).

    Motivo (prueba previa en T4, 2026-10-19): en Burgers la primera corrida de un proceso no reproducia a las siguientes (estado a 35 % de distancia,
    error final 1.7e-3 frente a 3.8e-3), mientras que las corridas 2, 3 y el barrido eran identicas bit a bit. Se calienta una vez por proceso, antes
    de la primera semilla. Usa otra semilla y nada de lo que produce se guarda; la puntuacion solo existe para calentar la ruta de evaluacion.
    """
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    t0 = clock()
    prob = pc.make_problem(args.problem, args.ref)
    if args.n_col and hasattr(prob, "n_col"):
        prob.n_col = args.n_col
    seed = args.seed + 1
    model = prob.build(args.arm, args.width, args.bases, seed).to(device=device, dtype=dtype)
    P = prob.pools(seed, device, dtype)
    every = max(1, steps // 2)
    adam_stage(prob, model, P, steps, ckpt_steps=[steps], sync=sync, record=TraceRecorder(pc, prob, P, every), record_every=every)
    lbfgs_stage(prob, model, P, seconds_cap=1e9, max_iter=args.lbfgs_chunk, chunk=args.lbfgs_chunk, snapshot_every=1e9, sync=sync)
    score_run(prob, model, device, dtype, clone_state(model))
    sync()
    return dict(seconds=clock() - t0, steps=steps, n_params=sum(q.numel() for q in model.parameters()),
                pool_points=int(P["residual"].shape[0]))


def primary(score):
    return next(iter(score.values()))


def run_seed(pc, args, out_dir=None, clock=time.perf_counter):
    """Barrido completo de una semilla. Devuelve (resultado serializable, estados por fraccion)."""
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    prob = pc.make_problem(args.problem, args.ref)
    if args.n_col and hasattr(prob, "n_col"):
        prob.n_col = args.n_col
    model = prob.build(args.arm, args.width, args.bases, args.seed).to(device=device, dtype=dtype)
    P = prob.pools(args.seed, device, dtype)
    steps = args.adam_steps or prob.adam_steps_default
    fractions = tuple(args.grid)
    gsteps = grid_steps(steps, fractions)
    if gsteps[max(fractions)] != steps:
        raise ValueError("la malla debe incluir 1.00 (el control)")

    # --- entrenamiento: sin referencia ---
    rec_every = getattr(args, "record_every", 0)
    recorder = TraceRecorder(pc, prob, P, rec_every) if rec_every else None
    A = adam_stage(prob, model, P, steps, ckpt_steps=gsteps.values(), seconds_cap=args.adam_seconds_cap, sync=sync, clock=clock,
                   record=recorder, record_every=rec_every)
    result = dict(problem=args.problem, arm=args.arm, width=args.width, bases=args.bases, seed=args.seed, dtype=args.dtype,
                  adam_steps=A["steps"], adam_seconds=A["seconds"], adam_status=A["status"], grid={str(f): s for f, s in gsteps.items()},
                  ckpt_seconds={str(s): t for s, t in A["ckpt_seconds"].items()}, lbfgs_seconds_cap=args.lbfgs_seconds,
                  lbfgs_chunk=args.lbfgs_chunk, lbfgs_max_iter=args.lbfgs_max_iter, device=str(device),
                  record_every=rec_every, record_seconds=A["record_seconds"],
                  status="DRAFT_UNVERIFIED", scientific_accepted=False, hist=A["hist"], runs={})
    states = {}
    trace = None
    if A["status"] != "adam_complete":
        return result, states, trace
    result["trigger"] = {}
    for rho in (0.05, 0.10, 0.20):
        try:
            t = handoff_step([h["step"] for h in A["hist"]], [h["loss"] for h in A["hist"]], k_max=steps, rho_star=rho)
            result["trigger"][str(rho)] = dict(k_t=t["k_t"], fired=t["fired"], reason=t["reason"], r_ref=t["r_ref"])
        except ValueError as exc:  # p. ej. receta mas corta que k_min: ausencia de medicion, no cero
            result["trigger"][str(rho)] = dict(k_t=None, fired=None, reason=f"not_computable: {exc}", r_ref=None)
    if recorder is not None:
        trace = recorder.arrays(model)
        result["trigger_settling"] = {}
        for name, kw in {"q0.8_rho0.10": {}, "q0.6_rho0.10": dict(q_star=0.6), "q0.9_rho0.10": dict(q_star=0.9),
                         "q0.8_rho0.05": dict(rho_star=0.05), "q0.8_rho0.20": dict(rho_star=0.20)}.items():
            try:
                t = settling_step(trace["steps"], trace["coeffs"], k_max=steps, **kw)
                result["trigger_settling"][name] = dict(k_t=t["k_t"], fired=t["fired"], reason=t["reason"])
            except ValueError as exc:  # p. ej. ventanas no equiespaciadas o receta mas corta que k_min
                result["trigger_settling"][name] = dict(k_t=None, fired=None, reason=f"not_computable: {exc}")
    t_total = A["ckpt_seconds"][steps] + args.lbfgs_seconds
    result["t_total"] = t_total
    runs = {}
    for f in (args.order or sorted(fractions)):
        k = gsteps[f]
        model.load_state_dict(A["ckpts"][k])
        cap = t_total - A["ckpt_seconds"][k]
        L = lbfgs_stage(prob, model, P, cap, max_iter=args.lbfgs_max_iter, chunk=args.lbfgs_chunk,
                        snapshot_every=args.snapshot_every, sync=sync, clock=clock)
        runs[f] = dict(k=k, cap=cap, adam_seconds_at_k=A["ckpt_seconds"][k], L=L)
        states[f] = L["final_state"]
        if out_dir is not None:
            torch.save({"state_dict": L["final_state"], "k": k}, Path(out_dir) / f"{args.problem}-seed{args.seed}-lbfgs-k{k}.pt")
    if out_dir is not None:
        for k, st in A["ckpts"].items():
            torch.save({"state_dict": st, "k": k}, Path(out_dir) / f"{args.problem}-seed{args.seed}-adam-k{k}.pt")

    # --- puntuacion contra la referencia: despues de entrenar ---
    result["ref_sha256"] = pc.sha(args.ref)
    for f in sorted(runs):
        r, L = runs[f], runs[f]["L"]
        curve = [dict(t=t, iters=i, **score_run(prob, model, device, dtype, st)) for t, i, st in L["snaps"]]
        result["runs"][str(f)] = dict(
            k=r["k"], cap=r["cap"], lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"],
            total_seconds=r["adam_seconds_at_k"] + L["seconds"], traj=L["traj"], curve=curve,
            adam_score_at_k=score_run(prob, model, device, dtype, A["ckpts"][r["k"]]),
            final_score=curve[-1] and {k2: v for k2, v in curve[-1].items() if k2 not in ("t", "iters")},
            final_state_sha256=state_sha256(L["final_state"]))
    states["adam_final"] = A["ckpts"][steps]
    return result, states, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=("burgers", "ac"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--campaign-py", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="rbf")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--bases", type=int, default=11)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--adam-steps", type=int, default=0)
    ap.add_argument("--adam-seconds-cap", type=float, default=0.0)
    ap.add_argument("--lbfgs-seconds", type=float, default=600.0)
    ap.add_argument("--lbfgs-chunk", type=int, default=50)
    ap.add_argument("--lbfgs-max-iter", type=int, default=15000)
    ap.add_argument("--n-col", type=int, default=0)
    ap.add_argument("--snapshot-every", type=float, default=30.0)
    ap.add_argument("--warmup-steps", type=int, default=50, help="pasos de Adam de la corrida desechable previa a la primera semilla (0 = sin calentamiento)")
    ap.add_argument("--record-every", type=int, default=500, help="pasos de Adam entre registros de la capa 0 y del residuo (0 = no registrar)")
    ap.add_argument("--grid", nargs="+", type=float, default=list(GRID))
    ap.add_argument("--order", nargs="+", type=float, default=None)
    a = ap.parse_args()
    pc = load_campaign(a.campaign_py)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.warmup_steps > 0:
        a.seed = a.seeds[0]
        print(json.dumps({"warmup": warmup(pc, a, steps=a.warmup_steps)}), flush=True)
    for seed in a.seeds:
        f = out / f"{a.problem}-seed{seed}.json"
        if f.exists():
            continue
        a.seed = seed
        res, _, trace = run_seed(pc, a, out_dir=out)
        if trace is not None:
            import numpy as np

            np.savez_compressed(out / f"{a.problem}-seed{seed}-adam-trace.npz", **trace)
            res["trace_file"] = f"{a.problem}-seed{seed}-adam-trace.npz"
        f.write_text(json.dumps(res))
        print(json.dumps({"seed": seed, "adam_seconds": res["adam_seconds"],
                          "k_t": res.get("trigger", {}).get("0.1", {}).get("k_t"),
                          "runs": {k: (v["lbfgs_status"], round(v["lbfgs_seconds"]), v["final_score"]) for k, v in res["runs"].items()}}),
              flush=True)


if __name__ == "__main__":
    main()
