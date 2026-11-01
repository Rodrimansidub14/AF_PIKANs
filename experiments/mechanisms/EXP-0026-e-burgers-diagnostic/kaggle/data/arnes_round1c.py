"""Ronda 1-C (BORRADOR SIN FIRMA): integra fisher_kpp_1d y allen_cahn_d2e-4 al arnes de la campana con la MISMA interfaz que ACProblem/BurgersProblem
(build, pools, loss, score, transform) y anade la instrumentacion preregistrada: region de mayor |u_x| de la referencia, sonda de cond(P) y nota por corrida.

Definiciones preregistradas (borrador):
  * Region "interfaz": el 5 % de puntos de la rejilla de la referencia (t:101 x x:201) con mayor |u_x| (u_x por diferencias finitas, np.gradient en x);
    region "resto": el complemento. Error por region = ||pred - ref||_R / ||ref||_R (L2 relativo restringido a la region) y share_sq_err = fraccion del error
    cuadratico total que cae en la region. La region se define con la referencia SOLO para puntuar (nunca entra al entrenamiento).
  * cond(P) (capa 1): ver `p_cond_probe`. P es la de la Enmienda 1 de EXP-0023: P = (B/m + 1e-3 I)^(-1/2) por grupo (arista), con B el bloque Gauss-Newton 11x11
    de gram_preconditioning.gn_blocks (N = 2048 puntos equiespaciados del pool) y m = traza(B)/11. cond(P_g) = sigma_max(P_g)/sigma_min(P_g)
    = sqrt((lambda_max(B/m)+eps)/(lambda_min(B/m)+eps)); el predictor es la MEDIANA de cond(P_g) sobre los 48 x 48 = 2304 grupos de la capa de indice 1
    (model.layers[1], oculta -> oculta, ancho 16 en estos problemas: 16 x 16 = 256 grupos). Se calcula igual en CUALQUIER brazo (B, E, F) sobre el modelo, sin
    modificarlo, al inicio (antes de entrenar) y al final: asi es una propiedad de la semilla (el modelo inicial es identico en los tres brazos).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP23 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP23))
import problems_round1c as p1c            # noqa: E402  (archivo del investigador, sin modificar)
import pre_trainer as pt                  # noqa: E402

pc, gp = pt.pc, pt.gp
PROVENANCE = HERE / "provenance_round1c.json"


class _Round1CBase(pc.ACProblem):
    """Hereda build/pools/transform del Allen-Cahn de la campana (mismo contrato: x en [-1,1], t en [0,1], residual en 8000+400+800 puntos Sobol)."""
    key = None
    lo, hi, n_in, n_out = [-1.0, 0.0], [1.0, 1.0], 2, 1
    adam_steps_default = 4000

    def __init__(self, ref_path=None):
        ref = p1c.load_reference(self.key, ref_path or HERE)
        self.x, self.t, self.u = ref["x"], ref["t"], ref["u"]      # u (nt, nx) = (101, 201)

    def loss(self, model, P):
        xt = P["residual"].detach().requires_grad_(True)
        u = self.transform(xt, model(xt))
        g = pc.grad(u, xt)
        uxx = pc.grad(g[:, 0:1], xt)[:, 0:1]
        term = self._res(u, g, uxx).square().mean()
        return term, {"pde": term}

    def _res(self, u, g, uxx):
        return self.pde_residual(u, (g[:, 0:1], g[:, 1:2]), uxx)

    def predict_grid(self, model, device, dtype):
        xx, tt = np.meshgrid(self.x, self.t, indexing="xy")
        xt = torch.as_tensor(np.stack((xx.reshape(-1), tt.reshape(-1)), 1), dtype=dtype, device=device)
        with torch.no_grad():
            return self.transform(xt, model(xt)).double().cpu().numpy().reshape(self.u.shape)

    @torch.no_grad()
    def score(self, model, device, dtype):
        err = self.predict_grid(model, device, dtype) - self.u
        return {"l2re_mat": float(np.linalg.norm(err) / np.linalg.norm(self.u)), "max_abs_mat": float(np.abs(err).max())}


class FisherKPP(_Round1CBase):
    name = "fisher_kpp_1d"
    key = "fisher_kpp"

    @staticmethod
    def transform(xt, raw):
        x, t = xt[:, 0:1], xt[:, 1:2]
        return p1c.kpp_ic_torch(x) + (1.0 - x.square()) * t * raw

    def pde_residual(self, u, g, uxx):
        return g[1] - p1c.KPP_D * uxx - p1c.KPP_R * u * (1.0 - u)


class AllenCahnDose(_Round1CBase):
    name = "allen_cahn_d2e-4"
    key = "allen_cahn_dose"
    d = p1c.AC_D_DOSE

    @staticmethod
    def transform(xt, raw):
        x, t = xt[:, 0:1], xt[:, 1:2]
        return x.square() * torch.cos(math.pi * x) + (1.0 - x.square()) * t * raw

    def pde_residual(self, u, g, uxx):
        return g[1] - self.d * uxx - p1c.AC_A * (u - u ** 3)


PROBLEMS_1C = {"kpp": FisherKPP, "ac_d2e-4": AllenCahnDose}
WIDTH = {"kpp": 16, "ac_d2e-4": 16}          # receta heredada de Allen-Cahn (NO firmada)
ADAM_FULL = {"kpp": 4000, "ac_d2e-4": 4000}      # = receta T1/V de 1-B (4000 Adam + L-BFGS 500 it.), leida de results/*-block/*.json


def make_problem_1c(key, ref_dir=None):
    return PROBLEMS_1C[key](ref_dir)


# ------------------------------------------------------------------ regiones (con la referencia, solo para puntuar)
def region_masks(u_ref, x, top_frac=0.05):
    """Mascaras (nt, nx): 'interfaz' = top_frac de puntos con mayor |u_x|; 'resto' = complemento."""
    ux = np.abs(np.gradient(u_ref, x, axis=1))
    thr = np.quantile(ux, 1.0 - top_frac)
    top = ux >= thr
    return top, ~top, float(thr)


def region_errors(pred, u_ref, x, top_frac=0.05):
    top, rest, thr = region_masks(u_ref, x, top_frac)
    e2 = (pred - u_ref) ** 2
    out = dict(n_top=int(top.sum()), n_rest=int(rest.sum()), ux_threshold=thr, share_sq_err_top=float(e2[top].sum() / e2.sum()) if e2.sum() > 0 else 0.0)
    for name, m in (("top", top), ("rest", rest)):
        out[f"l2re_{name}"] = float(np.sqrt(e2[m].sum()) / np.linalg.norm(u_ref[m]))
    return out


# ------------------------------------------------------------------ cond(P) de la capa 1
def p_cond_probe(prob, model, pool, n_points=gp.N_POINTS, chunk=128, layers=(0, 1, 2)):
    """Mediana (sobre grupos-arista) de cond(P_g) por capa, con P_g = (B_g/m + 1e-3 I)^(-1/2) (Enmienda 1). No modifica el modelo ni el optimizador.
    Funciona con el modelo plano o con uno ya precondicionado (usa los coeficientes efectivos)."""
    X = gp.subset(pool, n_points)
    blocks = gp.gn_blocks(prob, model, X, "edge", chunk)
    out = []
    for l in layers:
        B = blocks[l]
        B = B.reshape(-1, B.shape[-2], B.shape[-1])
        Bn, eps = gp._normalize(B, gp.EPS_REL)
        P, _ = gp._inv_sqrt_and_sqrt(Bn, eps)
        sv = torch.linalg.svdvals(P)
        out.append(float((sv[:, 0] / sv[:, -1]).median()))
    return out


# ------------------------------------------------------------------ nota por corrida (W&B offline si existe; siempre un JSON)
def note_run(out_dir, run_id, note, metrics=None, project=None):
    """Una nota por run_id. Siempre escribe <run_id>.note.json. Si `wandb` esta instalado la registra ADEMAS en modo OFFLINE (carpeta wandb/ dentro de out_dir);
    nunca sube nada: la sincronizacion es un `wandb sync` manual y autorizado (docs/reference/analysis-dashboard.md). No lee WANDB_API_KEY."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = "wandb_not_installed"
    try:
        import wandb
        run = wandb.init(project=project or os.environ.get("WANDB_PROJECT") or "EXP-0023-round1C-DRAFT", name=run_id, id=run_id, dir=str(out_dir), mode="offline",
                         config=dict(note=note), tags=["round1C", "signed-v1"], reinit=True, resume="allow")
        wandb.log({k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float)) and math.isfinite(v)})
        run.finish()
        status = "wandb_offline_logged"
    except ImportError:
        pass
    except Exception as e:                       # la nota es auxiliar: un wandb roto (p. ej. carpeta homonima) no mata la corrida; queda el .note.json
        status = f"wandb_error: {type(e).__name__}: {e}"
    (out_dir / f"{run_id}.note.json").write_text(json.dumps(dict(run_id=run_id, note=note, metrics=metrics or {}, wandb=status), indent=1))
    return status


# ------------------------------------------------------------------ corrida instrumentada
def run_one_1c(key, arm, seed, adam_steps, lbfgs_iters, out_dir, device="cpu", dtype="float32", save_ckpt=True, pre_kw=None):
    dev = torch.device(device)
    dt = torch.float32 if dtype == "float32" else torch.float64
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{key}-{arm}-seed{seed}-a{adam_steps}-l{lbfgs_iters}"
    prob = make_problem_1c(key)
    model = prob.build("rbf", WIDTH[key], 11, seed).to(device=dev, dtype=dt)
    P = prob.pools(seed, dev, dt)
    cond_init = p_cond_probe(prob, model, P["residual"])          # antes de tocar nada; identico en B/E/F de la misma semilla
    pre = None
    if arm != "B":
        pre = gp.GramPreconditioner(arm, **(pre_kw or {}))
        pre.attach(model)
    t0 = time.perf_counter()
    A = pt.adam_with_pre(prob, model, P, adam_steps, pre, sync=sync)
    adam_state = pt.sw.clone_state(model)
    out = dict(run_id=run_id, problem=key, arm=arm, seed=seed, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, adam_seconds=time.perf_counter() - t0,
               adam_status=A["status"], pre_diag=A["pre_diag"], device=device, dtype=dtype, hist=A["hist"], cond_init=cond_init, scientific_accepted=False)
    if A["status"] != "adam_complete":
        (out_dir / f"{run_id}.json").write_text(json.dumps(out))
        note_run(out_dir, run_id, dict(status=A["status"], cond_init=cond_init))
        return out
    if pre is not None and pre.due_lbfgs():
        d = pre.refresh(model, prob, P["residual"], None, sync=sync)
        d["step"] = "lbfgs_start"
        out["pre_diag"].append(d)
    L = pt.sw.lbfgs_stage(prob, model, P, 1e12, max_iter=lbfgs_iters, chunk=50, snapshot_every=1e12, sync=sync)
    out.update(lbfgs_status=L["status"], lbfgs_iterations=L["iterations"], lbfgs_seconds=L["seconds"], total_seconds=time.perf_counter() - t0,
               pre_recomputes=(pre.n_recomputes if pre else 0), pre_seconds=(pre.seconds if pre else 0.0))
    out["finite_params"] = all(bool(torch.isfinite(q).all()) for q in model.parameters())
    out["score_adam"] = pt.sw.score_run(prob, model, dev, dt, adam_state)
    out["score_final"] = pt.sw.score_run(prob, model, dev, dt, L["final_state"])
    out["primary_adam"] = pt.sw.primary(out["score_adam"])
    out["primary_final"] = pt.sw.primary(out["score_final"])
    # regiones (con la referencia, despues de entrenar) y cond(P) final
    model.load_state_dict(adam_state)
    out["region_adam"] = region_errors(prob.predict_grid(model, dev, dt), prob.u, prob.x)
    model.load_state_dict(L["final_state"])
    out["region_final"] = region_errors(prob.predict_grid(model, dev, dt), prob.u, prob.x)
    out["cond_final"] = p_cond_probe(prob, model, P["residual"])
    if save_ckpt:
        torch.save(adam_state, out_dir / f"{run_id}-adam_end.pt")
        torch.save(L["final_state"], out_dir / f"{run_id}-lbfgs_end.pt")
        out["checkpoints"] = [f"{run_id}-adam_end.pt", f"{run_id}-lbfgs_end.pt"]
    (out_dir / f"{run_id}.json").write_text(json.dumps(out))
    note_run(out_dir, run_id, dict(problem=key, arm=arm, seed=seed, draft="round1C firmada v1 2026-10-22; confirmatoria", cond_init=cond_init, cond_final=out["cond_final"]),
             metrics=dict(primary_adam=out["primary_adam"], primary_final=out["primary_final"], l2re_top_final=out["region_final"]["l2re_top"],
                          l2re_rest_final=out["region_final"]["l2re_rest"], cond_init_layer1=cond_init[1], cond_final_layer1=out["cond_final"][1],
                          total_seconds=out["total_seconds"]))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ronda 1-C (BORRADOR): NO lanzar en GPU sin firma. Pruebas de humo en CPU con --adam <= 50.")
    ap.add_argument("--problem", required=True, choices=list(PROBLEMS_1C))
    ap.add_argument("--arms", default="B,E,F")
    ap.add_argument("--seeds", default="6")
    ap.add_argument("--adam", type=int, default=0, help="0 = receta 1-B (4000)")
    ap.add_argument("--lbfgs-iters", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results_smoke"))
    ap.add_argument("--pre-every", type=int, default=gp.EVERY)
    ap.add_argument("--pre-n", type=int, default=gp.N_POINTS)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.device != "cpu" and os.environ.get("ROUND1C_GPU_AUTHORIZED") != "1":
        raise SystemExit("ronda 1-C sin firma: no se lanza en GPU (defina ROUND1C_GPU_AUTHORIZED=1 solo tras la firma del investigador)")
    steps = a.adam or ADAM_FULL[a.problem]
    for arm in a.arms.split(","):
        for s in map(int, a.seeds.split(",")):
            r = run_one_1c(a.problem, arm, s, steps, a.lbfgs_iters, a.out, a.device, a.dtype, pre_kw=dict(every=a.pre_every, n_points=a.pre_n))
            print(r["run_id"], "adam", r.get("primary_adam"), "final", r.get("primary_final"), "cond_init", [round(c, 1) for c in r["cond_init"]], flush=True)


if __name__ == "__main__":
    main()
