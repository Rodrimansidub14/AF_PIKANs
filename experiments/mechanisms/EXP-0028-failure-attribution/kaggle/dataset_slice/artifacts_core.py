"""EXP-0028, rebanada con artefactos completos (Allen-Cahn, Burgers, Fisher-KPP, NS2d-CG, DFG). Modulo GENERICO, autocontenido: solo numpy/torch (+ wandb OFFLINE opcional).
Mismo esquema de claves que `piezo/artifacts.py` (otro agente), para que un solo analisis lea ambos; no importa nada de ese archivo.

NO modifica los ejecutores existentes ni el mecanismo: se engancha por envoltorios en tiempo de ejecucion (`prob.loss`, `dfg_loss`, `GramPreconditioner.refresh`, `gp.gn_blocks`, `sw.lbfgs_stage`).
Solo LEE el modelo: no cambia parametros, no consume RNG, no altera la trayectoria (lo comprueba `run_slice.py --check`: B con y sin registro identicos bit a bit).

Por corrida escribe `<tag>-artifacts.npz` (np.savez_compressed, float32) y, una vez por problema, `ref_<problema>.npz`. Contenido (claves):
  hist/step, hist/loss, hist/term/<termino>, hist/gradnorm/<param>        cada 100 pasos de Adam (y el paso 1): perdida total, por termino y norma de gradiente por tensor (layers.k.coefficients|base)
  <snap>/c_eff/layerK, <snap>/terms/<t>, settle/<snap>/layerK             INSTANTANEA LIVIANA cada `light_every` (1000) pasos: coeficientes efectivos, terminos en el subconjunto de evaluacion, cambio
                                                                         relativo de coeficientes por arista frente a la instantanea anterior (mapa de asentamiento)
  <snap>/state/<nombre>, <snap>/fields/values, <snap>/resid/<termino>,    INSTANTANEA COMPLETA en pasos del calendario (0, 2000, M/2, M, cada 5000, L-BFGS final): state_dict completo (coeficientes
  <snap>/occupancy/layerK, <snap>/P_sv/layerK, <snap>/P_cond/layerK,      c', base, centros, anchos, pre_P), campos (u,v,p / u) en la malla de evaluacion, residuo por punto y por termino sobre el
  <snap>/gradterm/values|names                                           pool, ocupacion de cada base RBF (fraccion de evaluaciones con activacion > 0.1), espectro y cond de P, norma de gradiente
                                                                         por termino x tensor
  refreshNN/{step,trace,cond,P_sv}                                       cada recalculo de J/E/I: paso, traza de B por arista (o entrada), cond de cada bloque normalizado, valores singulares de P instalada
  meta_json                                                              semilla, versiones, hash de codigo y datos, configuracion, calendario, score_hist (error vs paso en la malla de evaluacion)
`<snap>` = `a000000` (paso de Adam) o `l00500` (iteraciones de L-BFGS). Tope de tamano por corrida: SIZE_CAP_MB (declarado; si se acerca, se omiten campos/residuos de instantaneas intermedias).
Puntuar durante el entrenamiento (score_hist) es solo MONITOREO: nada del entrenamiento lee la referencia.
W&B: siempre OFFLINE (`WANDB_MODE=offline`, project AF_PIKANS, entity man22611-uvg, group EXP-0028-artifact-slice, job_type=<problema>, tags=[problema, brazo, seed, receta]); dir = salida de la corrida; no lee
ni escribe WANDB_API_KEY ni .env, no sube nada. Sincronizar es un acto local del investigador (wandb_tools/sync_wandb.py).
API publica: SliceRecorder (enganches), save_run_artifacts(model, prob, tag, step, ...) (instantanea suelta), make_adapter(problem, prob), make_lbfgs_stage(sw, recorder).
"""
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCHEMA = "slice-1"
SIZE_CAP_MB = 25.0                 # tope declarado por corrida (comprimido)
RAW_CAP_MB = 70.0                  # tope sin comprimir en memoria antes de omitir instantaneas intermedias
OCC_THRESHOLD = 0.1
WANDB_MAX_ARTIFACT_MB = 15.0       # solo para artifact_mode="file"
MAX_GRID = 40000
N_RES = 16384                      # maximo de puntos por clave para residuos por punto
N_TERMS = 2048                     # maximo de puntos por clave para terminos/gradientes por termino en el subconjunto de evaluacion
ENTITY, PROJECT, GROUP = "man22611-uvg", "AF_PIKANS", "EXP-0028-artifact-slice"


# ------------------------------------------------------------------ utilidades
def _f32(x):
    return np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float32)


def _sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


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
    return dict(schema=SCHEMA, seed=seed, versions=v, git_commit=commit, hashes={Path(f).name: _sha(f) for f in files}, cfg=cfg,
                cuda=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))


def _layers(model):
    return list(model.layers) if hasattr(model, "layers") else []


def effective_coefficients(model):
    out = []
    for layer in _layers(model):
        out.append((layer.pre_layout.apply(layer.pre_P, layer.coefficients) if hasattr(layer, "pre_P") else layer.coefficients).detach().double().cpu())
    return out


def state_arrays(model, prefix):
    return {f"{prefix}/state/{k}": _f32(v) for k, v in model.state_dict().items()}


def p_spectrum(model):
    out = []
    for layer in _layers(model):
        if hasattr(layer, "pre_P"):
            out.append(_f32(torch.linalg.svdvals(layer.pre_P.detach().double().cpu())))
    return out


@torch.no_grad()
def occupancy(model, X, thr=OCC_THRESHOLD):
    """Por capa RBF: fraccion de evaluaciones (puntos) en que la activacion de la base g sobre la entrada i supera `thr` -> [(I, G)]."""
    layers = _layers(model)
    counts, total, handles = [None] * len(layers), [0] * len(layers), []

    def make(k):
        def hook(mod, inp, out):
            x = inp[0]
            act = torch.exp(-((x.unsqueeze(-1) - mod.centers) / mod.width).square()) > thr
            c = act.double().sum(0)
            counts[k] = c if counts[k] is None else counts[k] + c
            total[k] += x.shape[0]
        return hook

    for k, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make(k)))
    try:
        model(X)
    finally:
        for h in handles:
            h.remove()
    return [_f32(counts[k] / max(total[k], 1)) for k in range(len(layers))]


def block_diagnostics(blocks, structure, eps_rel=1e-3):
    """Por capa: traza de B por arista/entrada y cond de cada bloque normalizado (B/m + eps I, m = traza/n)."""
    res = []
    for B in blocks:
        if structure == "diag":
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


class _CaptureBlocks:
    """Envuelve gp.gn_blocks durante `pre.refresh` para quedarse con los bloques B (la clase no los devuelve)."""

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


def _equi(n, k):
    return torch.linspace(0, n - 1, k).round().long()


def subset_pool(P, nmax):
    """Subconjunto determinista (indices equiespaciados) de cada clave del pool."""
    out = {}
    for key, v in P.items():
        out[key] = v.detach() if v.shape[0] <= nmax else v.detach()[_equi(v.shape[0], nmax).to(v.device)]
    return out


# ------------------------------------------------------------------ adaptadores por problema
class ScalarAdapter:
    """Allen-Cahn, Fisher-KPP, Burgers: una salida con transformacion dura, un termino (`pde`), pool P['residual']."""
    field_names = ("u",)
    term_names = ("pde",)

    def __init__(self, prob, ob):
        self.prob, self.ob = prob, ob
        self.name = prob.name
        sx = max(1, int(math.ceil(len(prob.x) * len(prob.t) / MAX_GRID)))
        self.x = np.asarray(prob.x)[::sx]
        self.t = np.asarray(prob.t)
        xx, tt = np.meshgrid(self.x, self.t, indexing="xy")
        self.xy = np.stack((xx.reshape(-1), tt.reshape(-1)), 1)
        self.shape = (len(self.t), len(self.x))
        self.ref = np.asarray(prob.u)[:, ::sx].reshape(-1, 1)

    def reference(self):
        return dict(xy=self.xy, values=self.ref, shape=np.asarray(self.shape), field_names=np.asarray(self.field_names))

    @torch.no_grad()
    def fields(self, model, dev, dt):
        xt = torch.as_tensor(self.xy, dtype=dt, device=dev)
        return _f32(self.prob.transform(xt, model(xt)))

    def residuals(self, model, P):
        X = P["residual"]
        return {"pde": _f32(self.ob.residual_at(self.prob, model, X.detach()))}

    def occ_points(self, P):
        return P["residual"].detach()


class VectorAdapter:
    """NS2d-CG y DFG: tres salidas (u, v, p), pool por clave, terminos de `prob.residual_terms`."""
    field_names = ("u", "v", "p")

    def __init__(self, prob, xy, ref, anchor=None, interior="domain"):
        self.prob, self.name, self.xy, self.ref, self.anchor, self.interior = prob, prob.name, np.asarray(xy, dtype=np.float64), np.asarray(ref, dtype=np.float64), anchor, interior
        self.shape = None

    def reference(self):
        return dict(xy=self.xy, values=self.ref, field_names=np.asarray(self.field_names), p_anchor_xy=np.asarray(self.anchor if self.anchor else (np.nan, np.nan)))

    @torch.no_grad()
    def fields(self, model, dev, dt):
        out = model(torch.as_tensor(self.xy, dtype=dt, device=dev)).double()
        if self.anchor is not None:                                    # DFG: presion con ancla, como la referencia (pressure_gauge_fixed)
            out[:, 2] -= model(torch.tensor([list(self.anchor)], dtype=dt, device=dev))[0, 2].double()
        return _f32(out)

    def residuals(self, model, P, chunk=512):
        X = P
        out, n = {}, max(v.shape[0] for v in X.values())
        for lo in range(0, n, chunk):
            Xs = {k: v[lo:lo + chunk] for k, v in X.items()}
            for r, w, name in self.prob.residual_terms(lambda x: model(x), Xs):
                out.setdefault(name, []).append(r.detach().double().reshape(-1).cpu())
        return {k: _f32(torch.cat(v)) for k, v in out.items()}

    def occ_points(self, P):
        return P[self.interior].detach()


def make_adapter(problem, prob, ob=None, dfg_ref=None, ns_ref=None):
    """problem: 'ac' | 'kpp' | 'burgers' | 'ns' | 'dfg'. `dfg_ref`: dict con xy, velocity, pressure_gauge_fixed; `ns_ref`: array (n, 5) x y u v p."""
    if problem in ("ac", "kpp", "burgers"):
        return ScalarAdapter(prob, ob)
    if problem == "ns":
        r = np.asarray(ns_ref)
        return VectorAdapter(prob, r[:, :2], r[:, 2:5], None, "domain")
    if problem == "dfg":
        ref = np.concatenate([dfg_ref["velocity"], np.asarray(dfg_ref["pressure_gauge_fixed"]).reshape(-1, 1)], 1)
        return VectorAdapter(prob, dfg_ref["xy"], ref, (2.2, 0.2), "interior")
    raise ValueError(problem)


# ------------------------------------------------------------------ grabador
class SliceRecorder:
    def __init__(self, problem, arm, seed, out_dir, tag, cfg, steps, full_steps, light_every=1000, hist_every=100, lbfgs_light=(100, 250), files=(), gp=None, wandb_on=True,
                 run_group=GROUP, recipe_label="medium", artifact_mode="reference"):
        self.problem, self.arm, self.seed, self.out, self.tag, self.steps = problem, arm, seed, Path(out_dir), tag, steps
        self.full_steps = set(int(s) for s in full_steps)
        self.light_every, self.hist_every, self.lbfgs_light = light_every, hist_every, tuple(lbfgs_light)
        self.cfg = dict(cfg, full_steps=sorted(self.full_steps), light_every=light_every, hist_every=hist_every, lbfgs_light=list(lbfgs_light))
        self.meta = run_meta(self.cfg, seed, files)
        self.gp, self.artifact_mode = gp, artifact_mode
        self.arrays, self.hist = {}, {"step": [], "loss": []}
        self.term_hist, self.grad_hist, self.score_hist, self.errors = {}, {}, [], []
        self.taken, self.calls, self.guard, self.refresh_n, self.raw_bytes = set(), 0, False, 0, 0
        self.prev_ceff, self.t_snap = None, 0.0
        self.prob = self.model = self.P = self.adapter = None
        self._restore = []
        self.out.mkdir(parents=True, exist_ok=True)
        self.wb, self.wb_status, self._wandb = None, "disabled", None
        if wandb_on:
            self._init_wandb(run_group, recipe_label)

    # -- W&B offline
    def _init_wandb(self, group, recipe_label):
        try:
            os.environ["WANDB_MODE"] = "offline"
            for var, sub in (("WANDB_DATA_DIR", "wandb_data"), ("WANDB_CACHE_DIR", "wandb_cache"), ("WANDB_CONFIG_DIR", "wandb_config")):
                os.environ.setdefault(var, str(self.out / sub))             # nada fuera de la salida del kernel (staging de artefactos, logs del nucleo)
            import wandb
            if not hasattr(wandb, "init"):
                raise ImportError("el paquete `wandb` resuelto no es la libreria (carpeta homonima en el cwd?)")
            self.wb = wandb.init(project=PROJECT, entity=os.environ.get("WANDB_ENTITY", ENTITY), group=group, job_type=self.problem, name=self.tag, id=self.tag.replace("/", "_"),
                                 tags=[self.problem, self.arm, f"seed{self.seed}", recipe_label, "artifact-slice"], config=self.cfg, dir=str(self.out), mode="offline",
                                 reinit="finish_previous", settings=wandb.Settings(console="off", silent=True))
            self._wandb, self.wb_status = wandb, "offline"
        except Exception as e:  # noqa: BLE001
            self.wb, self.wb_status = None, f"unavailable: {type(e).__name__}: {e}"

    def _log(self, step, d):
        if self.wb is not None:
            try:
                self.wb.log(d, step=int(step))
            except Exception:  # noqa: BLE001
                pass

    # -- instalacion de enganches
    def attach(self, prob, model, P, adapter, pre=None):
        self.prob, self.model, self.P, self.adapter = prob, model, P, adapter
        p0 = next(model.parameters())
        self.dev, self.dt = p0.device, p0.dtype
        self.Pres = subset_pool(P, N_RES)
        self.Pterm = subset_pool(P, N_TERMS)
        ref = adapter.reference()
        fp = self.out / f"ref_{self.problem}.npz"
        if not fp.exists():
            np.savez_compressed(fp, **{k: np.asarray(v) for k, v in ref.items()})
        for k, v in self.Pres.items():                                # puntos del pool de esta semilla (una vez)
            self.arrays[f"pool/{k}"] = _f32(v)
        self.pre = pre
        if pre is not None and pre.arm in ("J", "E", "I"):
            orig = pre.refresh
            rec = self

            def refresh_w(*a, **k):
                rec.maybe_snap(rec.calls)
                with _CaptureBlocks(rec.gp) as cap:
                    d = orig(*a, **k)
                rec.on_refresh(d, cap, a[0] if a else rec.model)
                return d
            pre.refresh = refresh_w
            self._restore.append(lambda: pre.__dict__.pop("refresh", None))

    def wrap_loss(self, orig):
        rec = self

        def w(*a, **k):
            if rec.guard:
                return orig(*a, **k)
            n = rec.calls
            rec.maybe_snap(n)
            total, terms = orig(*a, **k)
            rec.calls += 1
            if rec.calls == 1 or rec.calls % rec.hist_every == 0:
                rec.log_step(rec.calls, total, terms)
            return total, terms
        return w

    def end_adam(self):
        self.maybe_snap(self.calls)

    def close(self):
        for f in self._restore:
            f()
        self._restore = []

    # -- historia por paso
    def log_step(self, step, total, terms):
        self.hist["step"].append(step)
        self.hist["loss"].append(float(total.detach()))
        for k, v in terms.items():
            self.term_hist.setdefault(k, []).append(float(v.detach()))
        params = [(n, p) for n, p in self.model.named_parameters()]
        g = torch.autograd.grad(total, [p for _, p in params], retain_graph=True, allow_unused=True)
        gn = {n: (float(gi.detach().double().norm()) if gi is not None else 0.0) for (n, _), gi in zip(params, g)}
        for n, v in gn.items():
            self.grad_hist.setdefault(n, []).append(v)
        self._log(step, {"loss": self.hist["loss"][-1], **{f"term/{k}": self.term_hist[k][-1] for k in terms}, **{f"gradnorm/{n}": v for n, v in gn.items()}})

    # -- recalculo del bloque
    def on_refresh(self, d, cap, model):
        pfx = f"refresh{self.refresh_n:02d}"
        step = d.get("step", self.calls)
        self.arrays[f"{pfx}/step"] = np.asarray(self.calls)
        if cap is not None and cap.blocks is not None:
            for l, (tr, cond) in enumerate(block_diagnostics(cap.blocks, cap.structure)):
                self.arrays[f"{pfx}/layer{l}/trace"], self.arrays[f"{pfx}/layer{l}/cond"] = tr, cond
        spec = p_spectrum(model)
        for l, sv in enumerate(spec):
            self.arrays[f"{pfx}/layer{l}/P_sv"] = sv
        if spec:
            self._log(self.calls, {"P_cond_median": float(np.median(np.concatenate([(s[:, 0] / s[:, -1]) for s in spec]))), "refresh_seconds": float(d.get("seconds", 0.0))})
        self.refresh_n += 1

    # -- instantaneas
    def maybe_snap(self, n):
        if n in self.taken:
            return
        full = n in self.full_steps
        light = self.light_every and n % self.light_every == 0
        if full or light:
            self.taken.add(n)
            self.snapshot(f"a{n:06d}", n, full)

    def on_lbfgs(self, iters, final):
        """Llamado por el lazo de L-BFGS tras cada trozo aceptado (model ya tiene el estado aceptado)."""
        key = f"l{iters:05d}"
        if final:
            if key not in self.taken:
                self.taken.add(key)
                self.snapshot(key, self.steps + iters, True)
            return
        for thr in self.lbfgs_light:
            if iters >= thr and f"lthr{thr}" not in self.taken:
                self.taken.add(f"lthr{thr}")
                self.snapshot(key, self.steps + iters, False)            # liviana; si el final cae en la misma clave, la sobrescribe la completa

    def snapshot(self, key, step, full):
        t0 = time.perf_counter()
        self.guard = True
        a = {}
        try:
            with torch.enable_grad():
                a.update(self._snapshot(key, step, full))
        except Exception as e:  # noqa: BLE001 - un fallo de registro no mata la corrida; queda anotado (y `--check` lo exige vacio)
            self.errors.append(f"{key}: {type(e).__name__}: {e}")
        finally:
            self.guard = False
        for k, v in a.items():
            self.arrays[k] = v
            self.raw_bytes += np.asarray(v).nbytes
        self.t_snap += time.perf_counter() - t0

    def _snapshot(self, key, step, full):
        prob, model, ad = self.prob, self.model, self.adapter
        a = {}
        if full and self.raw_bytes / 2 ** 20 > RAW_CAP_MB and step != self.steps and not key.startswith("l"):
            full = False                                              # tope de tamano: se omite lo pesado de instantaneas intermedias
        score = prob.score(model, self.dev, self.dt)
        sc = {k: float(v) for k, v in score.items() if isinstance(v, (int, float))}
        self.score_hist.append(dict(key=key, step=int(step), **sc))
        ceff = effective_coefficients(model)
        for l, c in enumerate(ceff):
            a[f"{key}/c_eff/layer{l}"] = _f32(c)
            if self.prev_ceff is not None:
                rel = (c - self.prev_ceff[l]).flatten(2).norm(dim=-1) / self.prev_ceff[l].flatten(2).norm(dim=-1).clamp_min(1e-300)
                a[f"settle/{key}/layer{l}"] = _f32(rel)                  # (O, I) cambio relativo por arista desde la instantanea anterior
        self.prev_ceff = ceff
        total, terms = prob.loss(model, self.Pterm)
        tv = {k: float(v.detach()) for k, v in terms.items()}
        for k, v in tv.items():
            a[f"{key}/terms/{k}"] = np.asarray(v, dtype=np.float64)
        log = {f"score/{k}": v for k, v in sc.items()}
        log.update({f"eval_term/{k}": v for k, v in tv.items()})
        if full:
            a.update(state_arrays(model, key))
            a[f"{key}/fields/values"] = ad.fields(model, self.dev, self.dt)
            for k, v in ad.residuals(model, self.Pres).items():
                a[f"{key}/resid/{k}"] = v
            for l, o in enumerate(occupancy(model, ad.occ_points(self.Pres))):
                a[f"{key}/occupancy/layer{l}"] = o
            spec = p_spectrum(model)
            for l, sv in enumerate(spec):
                a[f"{key}/P_sv/layer{l}"] = sv
                a[f"{key}/P_cond/layer{l}"] = _f32(sv[:, 0] / sv[:, -1])
            params = [(n, p) for n, p in model.named_parameters()]
            names, rows = ["total"] + list(terms), []
            for t in [total] + list(terms.values()):
                g = torch.autograd.grad(t, [p for _, p in params], retain_graph=True, allow_unused=True)
                rows.append([float(gi.detach().double().norm()) if gi is not None else 0.0 for gi in g])
            a[f"{key}/gradterm/values"] = np.asarray(rows, dtype=np.float64)
            a[f"{key}/gradterm/names"] = np.asarray(json.dumps(dict(terms=names, params=[n for n, _ in params])))
        self._log(step, log)
        return a

    # -- cierre
    def finalize(self, extra=None):
        a = dict(self.arrays)
        a["hist/step"] = np.asarray(self.hist["step"])
        a["hist/loss"] = np.asarray(self.hist["loss"], dtype=np.float64)
        for k, v in self.term_hist.items():
            a[f"hist/term/{k}"] = np.asarray(v, dtype=np.float64)
        for k, v in self.grad_hist.items():
            a[f"hist/gradnorm/{k}"] = np.asarray(v, dtype=np.float64)
        a["meta_json"] = np.asarray(json.dumps(dict(self.meta, extra=extra or {}, wandb=self.wb_status, score_hist=self.score_hist, errors=self.errors, snapshot_seconds=self.t_snap), default=str))
        path = self.out / f"{self.tag}-artifacts.npz"
        np.savez_compressed(path, **a)
        mb = path.stat().st_size / 2 ** 20
        if self.wb is not None:
            try:
                art = self._wandb.Artifact(f"{self.tag}-artifacts".replace("/", "_"), type="run-state", metadata=dict(file=path.name, megabytes=mb, sha256=_sha(path), problem=self.problem, arm=self.arm, seed=self.seed))
                if self.artifact_mode == "file" and mb <= WANDB_MAX_ARTIFACT_MB:
                    art.add_file(str(path))                                   # copia el npz al staging (dentro de la salida); la subida de los bytes ocurre en el `wandb sync`
                    self.wb_status = "offline+artifact(file)"
                else:
                    art.add_reference(Path(path).resolve().as_uri(), name=path.name)   # solo URI + digest: el npz viaja en la salida del kernel; se adjunta en local con attach_artifacts_local.py
                    self.wb_status = "offline+artifact(reference)"
                self.wb.log_artifact(art)
                self.wb.summary.update({"artifact_mb": mb, "n_errors": len(self.errors)})
                self.wb.finish()
            except Exception as e:  # noqa: BLE001
                self.wb_status = f"offline (cierre con error: {type(e).__name__})"
        return dict(path=str(path), megabytes=mb, over_cap=bool(mb > SIZE_CAP_MB), n_arrays=len(a), n_snapshots=len(self.score_hist), n_errors=len(self.errors), errors=self.errors[:5],
                    wandb=self.wb_status, snapshot_seconds=self.t_snap)


def save_run_artifacts(model, prob, tag, step, out_dir=".", P=None, adapter=None, pre=None, seed=None, cfg=None, gp=None):
    """Instantanea suelta COMPLETA (sin lazo): estado, campos, residuos, ocupacion, espectro de P, gradientes por termino -> `<tag>-artifacts.npz`. Devuelve la ruta.
    Requiere `adapter` (make_adapter) y el pool `P`."""
    rec = SliceRecorder(getattr(prob, "name", "problem"), getattr(pre, "arm", "B") if pre is not None else "B", seed, out_dir, tag, cfg or {}, step, [step], light_every=0, wandb_on=False, gp=gp)
    rec.attach(prob, model, P, adapter, pre)
    rec.maybe_snap(step)
    rec.close()
    return Path(rec.finalize()["path"])


# ------------------------------------------------------------------ L-BFGS con enganche (copia del lazo de sweep.lbfgs_stage; mismo criterio de parada)
def make_lbfgs_stage(sw, rec):
    """Devuelve una funcion con la firma de `sw.lbfgs_stage` que ejecuta EL MISMO lazo y llama a `rec.on_lbfgs(iters, final)` tras cada trozo aceptado.
    Equivalencia con el original comprobada por `run_slice.py --check` (B con y sin registro, bit a bit en CPU)."""
    def lbfgs_stage(prob, model, P, seconds_cap, max_iter=15000, chunk=50, snapshot_every=30.0, sync=lambda: None, clock=time.perf_counter):
        opt2 = torch.optim.LBFGS(model.parameters(), max_iter=chunk, max_eval=chunk * 5, **sw.LBFGS_PARAMS)
        accepted = sw.clone_state(model)
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
            except Exception as exc:  # noqa: BLE001
                status = f"lbfgs_exception:{type(exc).__name__}"
                break
            after = opt2.state[opt2._params[0]].get("n_iter", 0)
            cur = float(prob.loss(model, P)[0].detach())
            if not math.isfinite(cur) or cur > last * 1.5:
                model.load_state_dict(accepted)
                status = "lbfgs_rejected_nonfinite_or_worse"
                break
            improved = (last - cur) / max(abs(last), 1e-30)
            accepted = sw.clone_state(model)
            last = cur
            iters += after - before
            elapsed = clock() - t1
            traj.append({"t": elapsed, "iters": iters, "loss": cur})
            if elapsed >= next_snap:
                snaps.append((elapsed, iters, accepted))
                next_snap = elapsed + snapshot_every
            rec.on_lbfgs(iters, False)
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
        rec.on_lbfgs(iters, True)
        return dict(status=status, iterations=iters, seconds=seconds, traj=traj, snaps=snaps, final_state=accepted)
    return lbfgs_stage
