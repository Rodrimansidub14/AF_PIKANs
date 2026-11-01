"""Artefactos de corrida para analisis posterior (EXP-0028, piezo; reutilizable por Allen-Cahn, Burgers, KPP, NS y DFG). Autocontenido: solo numpy/torch (+ wandb OFFLINE opcional).

Por corrida escribe UN .npz comprimido junto al JSON (`<tag>-artifacts.npz`) y, una vez por problema, `fem_<problema>_reference.npz`:
 (1) estado: state_dict completo (float64: coeficientes RBF `layers.k.coefficients`, `base`, centros, anchos, `pre_P` y los coeficientes reparametrizados c') en el paso 0, cada `state_every` (2000) y al final;
 (2) campos en la malla de evaluacion (201x21 del paper, float32) al final y cada `field_every` (5000): u, v, phi y las cinco salidas mixtas (sigma_xx, sigma_yy, tau_xy, D_x, D_y);
 (3) residuos por punto y por termino de la perdida (los del adaptador `residual_terms`: constitutivos, equilibrio, Gauss, borde) sobre el pool del paso y sobre la malla (interior + sus cuatro bordes), mismos pasos;
 (4) historia por termino (con peso) y total cada `hist_every` (100) y normas de gradiente por tensor de parametros (aqui `layers.k.coefficients` y `layers.k.base`);
 (5) diagnostico de cada recalculo del bloque (J/E/I): traza por arista de B, cond de cada bloque normalizado, valores singulares de P (antes de usarla), paso;
 (6) ocupacion de cada base RBF por capa (fraccion de evaluaciones con activacion > umbral, por entrada y base) en los pasos de campos, y el mapa de asentamiento de coeficientes: cambio relativo
     de los coeficientes efectivos por arista entre pasos consecutivos de `settle_every` (1000, que coincide con los recalculos);
 (7) consistencia constitutiva (lo que el paper declara como limitacion): discrepancia en unidades fisicas entre las cinco salidas mixtas predichas y las reconstruidas desde du, dv, dphi, y su L2 relativo;
 (8) meta: semilla, versiones (python, torch, numpy, wandb), hash SHA-256 del codigo y de los datos, configuracion completa.
W&B: siempre OFFLINE (`WANDB_MODE=offline`, `wandb.init(project='AF_PIKANS', entity='man22611-uvg', group='EXP-0028-piezo', job_type=<converse|direct>, mode='offline', dir=<salida>)`); no lee ni escribe WANDB_API_KEY, no sube nada;
la sincronizacion es local, con la clave del .env en memoria, por `wandb_tools/sync_wandb.py` (acto del investigador; nunca en Kaggle ni en el bundle).

API:  RunRecorder(prob, out_dir, tag, cfg, steps, ...)  con on_step / on_refresh / after_step / finalize;  save_run_artifacts(model, prob, tag, out_dir, P=..., pre=..., step=...) para una instantanea suelta.
Contrato opcional del problema: grid_points(), reference_fields(), boundary_sets(xy), field_names, consistency(model, xy), residual_terms(model_fn, X), loss(model, P)->(total, terms), name/mode.
Solo lectura sobre el modelo: no modifica parametros, no consume RNG, no cambia la trayectoria (lo prueba tests/test_piezo_problem.py).
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

OCC_THRESHOLD = 0.1
WANDB_MAX_ARTIFACT_MB = 60


# ------------------------------------------------------------------ utilidades
def _sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _f32(x):
    return np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float32)


def run_meta(cfg, seed, files=()):
    v = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, platform=platform.platform())
    try:
        import wandb  # noqa: F401
        v["wandb"] = getattr(wandb, "__version__", "?")
    except Exception:  # noqa: BLE001
        v["wandb"] = None
    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent), timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return dict(seed=seed, versions=v, git_commit=commit, hashes={Path(f).name: _sha(f) for f in files}, cfg=cfg, cuda=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))


def _layers(model):
    return list(model.layers) if hasattr(model, "layers") else []


def effective_coefficients(model):
    out = []
    for layer in _layers(model):
        out.append((layer.pre_layout.apply(layer.pre_P, layer.coefficients) if hasattr(layer, "pre_P") else layer.coefficients).detach().double().cpu())
    return out


def state_arrays(model, prefix):
    return {f"{prefix}/state/{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()}


@torch.no_grad()
def field_arrays(model, xy, dtype=torch.float64):
    o = model(torch.as_tensor(xy, dtype=dtype, device=next(model.parameters()).device)).double().cpu().numpy()
    return _f32(o)


def _slice(X, lo, n):
    return {k: v[lo:lo + n] for k, v in X.items()}


def residual_arrays(prob, model, X, chunk=512):
    """{termino: residuo por punto (float32)} y {termino: peso}. X: dict de conjuntos de puntos (sin 'interface'). Trozos del mismo rango para todas las llaves."""
    out, weights = {}, {}
    n = max(v.shape[0] for v in X.values())
    for lo in range(0, n, chunk):
        for r, w, name in prob.residual_terms(lambda x: model(x), _slice(X, lo, chunk)):
            out.setdefault(name, []).append(r.detach().double().reshape(-1).cpu())
            weights[name] = w
    return {k: _f32(torch.cat(v)) for k, v in out.items()}, weights


@torch.no_grad()
def occupancy(model, X, thr=OCC_THRESHOLD):
    """Por capa RBF: fraccion de evaluaciones (puntos x pasadas de la red) en que la activacion de la base g sobre la entrada i supera `thr`. Devuelve [(I, G) float32 por capa]."""
    layers = _layers(model)
    counts, total, handles = [None] * len(layers), [0] * len(layers), []

    def make(k, layer):
        def hook(mod, inp, out):
            x = inp[0]
            act = torch.exp(-((x.unsqueeze(-1) - mod.centers) / mod.width).square()) > thr
            c = act.double().sum(0)
            counts[k] = c if counts[k] is None else counts[k] + c
            total[k] += x.shape[0]
        return hook

    for k, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make(k, layer)))
    try:
        model(X)
    finally:
        for h in handles:
            h.remove()
    return [_f32(counts[k] / max(total[k], 1)) for k in range(len(layers))]


def snapshot_arrays(model, prob, step, P=None, grid=None, pre=None, with_state=True, with_fields=True):
    """Todo lo que cuelga de un paso: estado, campos en la malla, residuos (pool y malla), ocupacion, consistencia. Claves `s<paso>/...`."""
    pfx = f"s{step:06d}"
    a = {}
    if with_state:
        a.update(state_arrays(model, pfx))
    if not with_fields:
        return a
    xy = grid if grid is not None else (prob.grid_points() if hasattr(prob, "grid_points") else None)
    if xy is not None:
        a[f"{pfx}/fields/xy"] = _f32(xy)
        a[f"{pfx}/fields/values"] = field_arrays(model, xy, next(model.parameters()).dtype)
        if hasattr(prob, "boundary_sets"):
            _p = next(model.parameters())
            Xg = {"interior": torch.as_tensor(xy, dtype=_p.dtype, device=_p.device)}
            Xg.update({k: torch.as_tensor(v, dtype=_p.dtype, device=_p.device) for k, v in prob.boundary_sets(xy).items()})
            r, _ = residual_arrays(prob, model, Xg)
            a.update({f"{pfx}/resid_grid/{k}": v for k, v in r.items()})
        if hasattr(prob, "consistency"):
            c = prob.consistency(model, xy)
            a[f"{pfx}/consistency_grid/discrepancy"] = c["discrepancy"]
            a[f"{pfx}/consistency_grid/rel_l2"] = np.asarray(c["rel_l2"], dtype=np.float64)
    if P is not None:
        X = {k: v for k, v in P.items() if k != "interface"}
        r, _ = residual_arrays(prob, model, X)
        a.update({f"{pfx}/resid_pool/{k}": v for k, v in r.items()})
        a[f"{pfx}/pool/interior_xy"] = _f32(P["interior"])
        occ = occupancy(model, P["interior"])
        for k, o in enumerate(occ):
            a[f"{pfx}/occupancy/layer{k}"] = o
        if hasattr(prob, "consistency"):
            c = prob.consistency(model, P["interior"].detach().cpu().numpy())
            a[f"{pfx}/consistency_pool/rel_l2"] = np.asarray(c["rel_l2"], dtype=np.float64)
    return a


# ------------------------------------------------------------------ recalculo del bloque: capturar B y medir
class _CaptureBlocks:
    """Envuelve gram_preconditioning.gn_blocks durante `pre.refresh` para quedarse con los bloques B (la clase no los devuelve)."""

    def __init__(self, gp):
        self.gp, self.blocks, self.structure = gp, None, None

    def __enter__(self):
        self.orig = self.gp.gn_blocks

        def wrapped(prob, model, X, structure, chunk=128):
            out = self.orig(prob, model, X, structure, chunk)
            self.blocks, self.structure = [b.detach().cpu() for b in out], structure
            return out
        self.gp.gn_blocks = wrapped
        return self

    def __exit__(self, *a):
        self.gp.gn_blocks = self.orig


def block_diagnostics(blocks, structure, eps_rel=1e-3):
    """Por capa: traza por arista/entrada de B, cond de cada bloque normalizado (B/m + eps I, m = traza/n), y valores singulares de la matriz de bloque de la traza."""
    res = []
    for B in blocks:
        if structure == "diag":                       # (O, I, G): diagonal de cada bloque de arista
            tr = B.sum(-1).reshape(-1)
            d = B.reshape(-1, B.shape[-1]) / (tr.reshape(-1, 1) / B.shape[-1]).clamp_min(1e-300)
            cond = (d + eps_rel).max(-1).values / (d + eps_rel).min(-1).values
        else:
            M = B.reshape(-1, B.shape[-2], B.shape[-1]).double()
            tr = M.diagonal(dim1=-2, dim2=-1).sum(-1)
            m = (tr / M.shape[-1]).clamp_min(1e-300)
            ev = torch.linalg.eigvalsh(0.5 * (M + M.transpose(-1, -2)) / m[:, None, None] + eps_rel * torch.eye(M.shape[-1], dtype=M.dtype))
            cond = ev[:, -1] / ev[:, 0].clamp_min(1e-300)
        res.append((_f32(tr), _f32(cond)))
    return res


def p_spectrum(model):
    out = []
    for layer in _layers(model):
        if hasattr(layer, "pre_P"):
            out.append(_f32(torch.linalg.svdvals(layer.pre_P.detach().double().cpu())))
    return out


# ------------------------------------------------------------------ grabador de la corrida
class RunRecorder:
    def __init__(self, prob, out_dir, tag, cfg, steps, seed, gp=None, files=(), state_every=2000, field_every=5000, hist_every=100, settle_every=1000,
                 wandb_offline=True, project="AF_PIKANS", group="EXP-0028-piezo", job_type=None, entity="man22611-uvg"):
        self.prob, self.out, self.tag, self.steps, self.seed, self.gp = prob, Path(out_dir), tag, steps, seed, gp
        self.state_every, self.field_every, self.hist_every, self.settle_every = state_every, field_every, hist_every, settle_every
        self.cfg = dict(cfg, state_every=state_every, field_every=field_every, hist_every=hist_every, settle_every=settle_every)
        self.meta = run_meta(self.cfg, seed, files)
        self.arrays, self.hist = {}, {"step": [], "loss": []}
        self.term_hist, self.grad_hist, self.score_hist, self.refresh_n = {}, {}, [], 0
        self.prev_coefs, self.wb, self.wb_status = None, None, "disabled"
        self.out.mkdir(parents=True, exist_ok=True)
        ref = getattr(prob, "reference_fields", None)
        if ref is not None:                                    # el FEM de referencia: una sola vez por problema
            fp = self.out / f"fem_{getattr(prob, 'mode', prob.name)}_reference.npz"
            if not fp.exists():
                np.savez_compressed(fp, **{k: np.asarray(v) for k, v in ref().items()})
        self.job_type = job_type or getattr(prob, "mode", None)
        if wandb_offline:
            self._init_wandb(project, group, entity)

    # -- W&B offline (nunca sube; no toca WANDB_API_KEY)
    def _init_wandb(self, project, group, entity):
        if sys.platform == "win32" and os.environ.get("X28_WANDB_FORCE") != "1":
            # el servicio de wandb puede cerrar el proceso con una violacion de acceso en Windows (no se puede capturar con try/except): el registro offline se verifica en Linux (Kaggle/Colab)
            self.wb, self.wb_status = None, "skipped: win32 (W&B offline se verifica en Linux; X28_WANDB_FORCE=1 lo fuerza)"
            return
        try:
            os.environ["WANDB_MODE"] = "offline"
            import wandb
            if not hasattr(wandb, "init"):
                raise ImportError("el paquete `wandb` resuelto no es la libreria (carpeta homonima en el cwd?)")
            self.wb = wandb.init(project=project, entity=os.environ.get("WANDB_ENTITY", entity), group=group, job_type=self.job_type, name=self.tag, config=self.cfg, dir=str(self.out), mode="offline", reinit="finish_previous", settings=wandb.Settings(console="off", silent=True))
            self.wb_status = "offline"
            self._wandb = wandb
        except Exception as e:  # noqa: BLE001
            self.wb, self.wb_status = None, f"unavailable: {type(e).__name__}: {e}"

    def _log(self, step, d):
        if self.wb is not None:
            try:
                self.wb.log(d, step=step)
            except Exception:  # noqa: BLE001
                pass

    # -- ganchos del lazo
    def start(self, model, P):
        self.arrays.update(snapshot_arrays(model, self.prob, 0, P, pre=None, with_state=True, with_fields=False))
        self.prev_coefs = effective_coefficients(model)

    def on_step(self, step, loss, terms, model):
        """Despues de backward y antes de opt.step: historia por termino y normas de gradiente (cada hist_every pasos, y el paso 1)."""
        if step != 1 and step % self.hist_every:
            return
        self.hist["step"].append(step)
        self.hist["loss"].append(float(loss.detach()))
        for k, v in terms.items():
            self.term_hist.setdefault(k, []).append(float(v.detach()))
        gn = {n: float(p.grad.detach().double().norm()) for n, p in model.named_parameters() if p.grad is not None}
        for n, v in gn.items():
            self.grad_hist.setdefault(n, []).append(v)
        self._log(step, {"loss": self.hist["loss"][-1], **{f"term/{k}": self.term_hist[k][-1] for k in terms}, **{f"gradnorm/{n}": v for n, v in gn.items()}})

    def capture(self):
        return _CaptureBlocks(self.gp)

    def on_refresh(self, step, cap, model):
        """Despues de `pre.refresh`: traza/cond de B por bloque (capturados) y espectro de P recien instalada."""
        pfx = f"refresh{self.refresh_n:02d}"
        self.arrays[f"{pfx}/step"] = np.asarray(step)
        if cap is not None and cap.blocks is not None:
            for l, (tr, cond) in enumerate(block_diagnostics(cap.blocks, cap.structure)):
                self.arrays[f"{pfx}/layer{l}/trace"], self.arrays[f"{pfx}/layer{l}/cond"] = tr, cond
        spec = p_spectrum(model)
        for l, sv in enumerate(spec):
            self.arrays[f"{pfx}/layer{l}/P_sv"] = sv
        if spec:
            self._log(step, {"P_cond_median": float(np.median(np.concatenate([(s[:, 0] / s[:, -1]) for s in spec])))})
        self.refresh_n += 1

    def after_step(self, step, model, P, score=None):
        if step % self.settle_every == 0:
            cur = effective_coefficients(model)
            if self.prev_coefs is not None:
                for l, (a, b) in enumerate(zip(cur, self.prev_coefs)):
                    rel = (a - b).flatten(2).norm(dim=-1) / b.flatten(2).norm(dim=-1).clamp_min(1e-300)          # (O, I) por arista
                    self.arrays[f"settle/step{step:06d}/layer{l}"] = _f32(rel)
            self.prev_coefs = cur
        if score is not None:
            self.score_hist.append(dict(step=step, **{k: v for k, v in score.items() if isinstance(v, float)}))
            self._log(step, {f"score/{k}": v for k, v in score.items() if isinstance(v, float)})
        heavy = step % self.field_every == 0 or step == self.steps
        st = step % self.state_every == 0 or step == self.steps
        if heavy or st:
            self.arrays.update(snapshot_arrays(model, self.prob, step, P if heavy else None, pre=None, with_state=st, with_fields=heavy))

    def finalize(self, model, P, extra=None):
        """Escribe `<tag>-artifacts.npz` (comprimido) y cierra W&B offline. Devuelve un resumen para el JSON de la corrida."""
        a = dict(self.arrays)
        a["hist/step"] = np.asarray(self.hist["step"])
        a["hist/loss"] = np.asarray(self.hist["loss"], dtype=np.float64)
        for k, v in self.term_hist.items():
            a[f"hist/term/{k}"] = np.asarray(v, dtype=np.float64)
        for k, v in self.grad_hist.items():
            a[f"hist/gradnorm/{k}"] = np.asarray(v, dtype=np.float64)
        a["meta_json"] = np.asarray(json.dumps(dict(self.meta, extra=extra or {}, wandb=self.wb_status, score_hist=self.score_hist), default=str))
        path = self.out / f"{self.tag}-artifacts.npz"
        np.savez_compressed(path, **a)
        mb = path.stat().st_size / 2 ** 20
        if self.wb is not None:
            try:
                if mb <= WANDB_MAX_ARTIFACT_MB:
                    art = self._wandb.Artifact(f"{self.tag}-artifacts", type="run-state")
                    art.add_file(str(path))
                    self.wb.log_artifact(art)
                    self.wb_status = "offline+artifact"
                else:
                    self.wb_status = f"offline (artefacto de {mb:.0f} MB omitido)"
                self.wb.finish()
            except Exception as e:  # noqa: BLE001
                self.wb_status = f"offline (cierre con error: {type(e).__name__})"
        return dict(path=str(path), megabytes=mb, n_arrays=len(a), wandb=self.wb_status)


def save_run_artifacts(model, prob, tag, out_dir=".", P=None, pre=None, step=0, extras=None, seed=None, files=()):
    """Instantanea suelta (sin lazo): estado, campos en la malla, residuos (pool y malla), ocupacion y consistencia -> `<tag>-artifacts.npz`. Devuelve la ruta."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = snapshot_arrays(model, prob, step, P, pre=pre)
    for l, sv in enumerate(p_spectrum(model)):
        a[f"s{step:06d}/layer{l}/P_sv"] = sv
    a["meta_json"] = np.asarray(json.dumps(dict(run_meta(extras or {}, seed, files), step=step), default=str))
    path = out / f"{tag}-artifacts.npz"
    np.savez_compressed(path, **a)
    return path


# ------------------------------------------------------------------ guarda: un fallo de artefactos nunca debe matar el entrenamiento (en GPU, 2026-10-25, un tensor en CPU mato dos corridas de ~30 min)
import functools  # noqa: E402
import traceback  # noqa: E402


def _guarded(fn, default=None):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except Exception as e:                                               # noqa: BLE001
            if not hasattr(self, "errors"):
                self.errors = []
            self.errors.append(f"{fn.__name__}: {type(e).__name__}: {e}"[:400])
            print("AVISO artefactos:", self.errors[-1], flush=True)
            return default() if callable(default) else default
    return wrapper


for _n in ("start", "on_refresh", "on_step", "after_step"):
    if hasattr(RunRecorder, _n):
        setattr(RunRecorder, _n, _guarded(getattr(RunRecorder, _n)))
if hasattr(RunRecorder, "finalize"):
    RunRecorder.finalize = _guarded(RunRecorder.finalize, default=lambda: dict(error="finalize fallo; ver AVISO en el registro"))
