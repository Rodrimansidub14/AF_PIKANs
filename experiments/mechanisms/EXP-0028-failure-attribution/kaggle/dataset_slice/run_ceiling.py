"""EXP-0018: techo oraculo del foco por responsabilidad. Por semilla, desde el MISMO estado final de Adam, corre el refinamiento (L-BFGS de la receta) en cada rama:

  B  control, L-BFGS sin cambios          P  foco suave: coeficientes reparametrizados por base, s = clip((A*/mediana)^(beta/2), 0.1, 10)
  H  foco duro: solo el 50 % de bases con mayor A* se mueve   N  P con las escalas permutadas dentro de cada capa (sin informacion)
  O  P con la ocupacion en lugar de A* (exploratorio)         NH H con el 50 % de bases elegido al azar (control de H; se corre solo si H pasa H-CEIL-1)

A* (responsabilidad con el error REAL, `oracle_ceiling`) se calcula una vez, en el estado de partida, con el unico acceso a la referencia de este modulo
(`oracle_responsibility`); el refinamiento recibe solo numeros. La referencia tambien puntua los resultados, despues de entrenar.
Metricas por rama: error L2 relativo global (el de la receta), error L2 relativo en la region critica (frente de Burgers |x| <= 5 nu, t >= 1/pi; interfaz de
Allen-Cahn |u_ref| < 0.9), iteraciones, segundos, estado de parada.

Uso (GPU, solo tras la firma): python run_ceiling.py --problem burgers --seeds 301 302 --state-dir <carpeta con los *-adam-k*.pt> --out OUT ...
Si falta el estado de Adam de una semilla, se entrena con la receta (sweep.adam_stage).
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP16 = HERE.parent / "EXP-0016-stall-handoff"
EXP17 = HERE.parent / "EXP-0017-representation-battery"
for _p in HERE.parents:
    if (_p / "src" / "mechanisms" / "responsibility_focus.py").exists():
        sys.path.insert(0, str(_p))
        break


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find(name, *candidates):
    for c in candidates:
        if (c / name).exists():
            return c / name
    raise FileNotFoundError(name)


BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
sw = _load(_find("sweep.py", *([BUNDLE] if BUNDLE else []), EXP16), "exp0018_sweep")
sig = _load(_find("signals.py", *([BUNDLE] if BUNDLE else []), EXP17), "exp0018_signals")
try:
    from src.mechanisms import responsibility_focus as rf
except ImportError:
    rf = _load(_find("responsibility_focus.py", *([BUNDLE] if BUNDLE else [])), "exp0018_rf")

ARMS_DEFAULT = ["B", "P", "H", "N", "O"]
N_SUB, SUB_SEED, BETA, Q_HARD = 8192, 20261020, 1.0, 0.5
NU = 0.01 / math.pi


def eval_grid(prob, name):
    sy, sx = (2, 4) if name == "burgers" else (1, 1)
    xx, tt = np.meshgrid(prob.x[::sx], prob.t[::sy], indexing="xy")
    pts = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=torch.float32)
    return pts, prob.u[::sy, ::sx]


def critical_mask(name, pts_np, u_ref_flat):
    if name == "burgers":
        return (np.abs(pts_np[:, 0]) <= 5 * NU) & (pts_np[:, 1] >= 1 / math.pi)
    return np.abs(u_ref_flat) < 0.9


def region_errors(prob, name, model, device, dtype):
    """Error L2 relativo global y en la region critica, sobre la malla completa de la referencia (puntuacion: despues de entrenar)."""
    xx, tt = np.meshgrid(prob.x, prob.t, indexing="xy")
    xt = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=dtype, device=device)
    with torch.no_grad():
        pred = prob.transform(xt, model(xt)).double().cpu().numpy().reshape(-1)
    u = prob.u.reshape(-1)
    mask = critical_mask(name, xt.cpu().numpy(), u)
    e = pred - u
    return dict(global_l2=float(np.linalg.norm(e) / np.linalg.norm(u)),
                critical_l2=float(np.linalg.norm(e[mask]) / np.linalg.norm(u[mask])) if mask.any() else None, critical_points=int(mask.sum()))


def oracle_responsibility(model, xs, err):
    """UNICO uso de la referencia (via `err` = |u - u_ref| en los puntos xs): A* y, sin referencia, la ocupacion O. Etiquetado oracle_ceiling."""
    A, O = sig.responsibility(model, xs, torch.as_tensor(err, dtype=torch.float32, device=xs.device))
    return A, O


def arm_transforms(arm, A_star, O, seed):
    """Devuelve (escalas o None, mascaras o None) de la rama."""
    s_p = rf.scales_from_responsibility(A_star, beta=BETA)
    if arm == "B":
        return None, None
    if arm == "P":
        return s_p, None
    if arm == "N":
        return rf.permute_within_layer(s_p, seed=seed), None
    if arm == "O":
        return rf.scales_from_responsibility(O, beta=BETA), None
    if arm == "H":
        return None, rf.top_fraction_masks(A_star, q=Q_HARD)
    if arm == "NH":
        rng = np.random.default_rng([seed, 99])
        shapes = [np.asarray(a.detach().cpu()).shape for a in A_star]
        total = sum(int(np.prod(s)) for s in shapes)
        keep = max(1, int(math.ceil(Q_HARD * total)))
        flat = np.zeros(total, dtype=bool)
        flat[rng.permutation(total)[:keep]] = True
        out, pos = [], 0
        for s in shapes:
            n = int(np.prod(s))
            out.append(flat[pos:pos + n].reshape(s))
            pos += n
        return None, out
    raise ValueError(arm)


def run_job(pc, args, seed, clock=time.perf_counter):
    device = torch.device(args.device)
    dtype = torch.float32
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    prob = pc.make_problem(args.problem, args.ref)
    if args.n_col and hasattr(prob, "n_col"):
        prob.n_col = args.n_col
    P = prob.pools(seed, device, dtype)

    def fresh():
        return prob.build("rbf", args.width, args.bases, seed).to(device=device, dtype=dtype)

    state_file = Path(args.state_dir) / f"{args.problem}-seed{seed}-adam-k{args.adam_steps}.pt" if args.state_dir else None
    trained = False
    if state_file is not None and state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=False)["state_dict"]
    else:
        m = fresh()
        sw.adam_stage(prob, m, P, args.adam_steps, sync=sync)
        state, trained = sw.clone_state(m), True
    model = fresh()
    model.load_state_dict(state)

    pts, u_ref = eval_grid(prob, args.problem)
    pts = pts.to(device)
    with torch.no_grad():
        err_full = np.abs(prob.transform(pts, model(pts)).cpu().numpy().reshape(u_ref.shape) - u_ref).reshape(-1)
    idx = np.sort(np.random.default_rng(SUB_SEED).choice(pts.shape[0], size=min(N_SUB, pts.shape[0]), replace=False))
    xs = pts[idx]
    A_star, O = oracle_responsibility(model, xs, err_full[idx])
    start = region_errors(prob, args.problem, model, device, dtype)

    out = dict(problem=args.problem, seed=seed, adam_state="trained_here" if trained else str(state_file.name), adam_steps=args.adam_steps,
               start=start, beta=BETA, q_hard=Q_HARD, n_bases=int(sum(np.asarray(a.detach().cpu()).size for a in A_star)),
               top20_share_Astar=sig.top_share(sig.flat(A_star), 0.2), arms={}, status="DRAFT_UNVERIFIED", scientific_accepted=False,
               label="oracle_ceiling")
    for arm in args.arms:
        s_list, masks = arm_transforms(arm, A_star, O, seed)
        m = fresh()
        m.load_state_dict(state)
        if s_list is not None:
            rf.apply_scales(m, s_list)
        if masks is not None:
            rf.apply_freeze(m, masks)
        L = sw.lbfgs_stage(prob, m, P, seconds_cap=args.lbfgs_seconds, max_iter=args.lbfgs_max_iter, chunk=args.lbfgs_chunk,
                           snapshot_every=1e9, sync=sync, clock=clock)
        rf.release(m)
        out["arms"][arm] = dict(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], finite=bool(
            all(torch.isfinite(p).all() for p in m.parameters())), **region_errors(prob, args.problem, m, device, dtype),
            state_sha256=sw.state_sha256(sw.clone_state(m)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=("burgers", "ac"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--campaign-py", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-dir", default="")
    ap.add_argument("--arms", nargs="+", default=ARMS_DEFAULT)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--bases", type=int, default=11)
    ap.add_argument("--adam-steps", type=int, default=20000)
    ap.add_argument("--lbfgs-seconds", type=float, default=600.0)
    ap.add_argument("--lbfgs-chunk", type=int, default=50)
    ap.add_argument("--lbfgs-max-iter", type=int, default=15000)
    ap.add_argument("--n-col", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    pc = sw.load_campaign(a.campaign_py)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.warmup_steps > 0:
        a.dtype, a.seed, a.arm = "float32", a.seeds[0], "rbf"
        print(json.dumps({"warmup": sw.warmup(pc, a, steps=a.warmup_steps)}), flush=True)
    for seed in a.seeds:
        f = out / f"{a.problem}-seed{seed}.json"
        if f.exists():
            continue
        res = run_job(pc, a, seed)
        f.write_text(json.dumps(res))
        print(json.dumps({"seed": seed, "arms": {k: (v["lbfgs_status"], round(v["lbfgs_seconds"]), v["global_l2"], v["critical_l2"]) for k, v in res["arms"].items()}}),
              flush=True)


if __name__ == "__main__":
    main()
