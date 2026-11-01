"""EXP-0025 (v1 firmada 2026-10-21; GPU bloqueada, ver spec): observador con sondeo. Elige palanca durante el entrenamiento.

Cada `K` pasos de Adam: instantánea -> continuar `n_p` pasos con cada palanca desde la MISMA instantánea -> evaluar el residuo en puntos retenidos H (flujo propio)
-> restaurar exactamente -> aplicar la elegida los `K` pasos siguientes. Palancas: a0 (continuar), aR (muestreo ∝ |r|^2, `rad.sample_rad`), aG (corrección de Gram,
instancia de `GramPreconditioner` que entrega quien llama). Este módulo NO importa `evals`, no lee la solución de referencia y no modifica `rad.py` ni
`gram_preconditioning.py`.

Decisiones de implementación que el spec deja abiertas (todas configurables, ninguna ajustada con datos): K=500, n_p=25, delta=0.05 (mejora relativa mínima del residuo
medio), gamma=0.25 (empeoramiento relativo máximo por celda crítica), rejilla de 16x16 celdas, 5 % de celdas críticas, piso uniforme 0.5. El «residuo medio» es la media
de r^2 sobre H (coherente con la pérdida). Las celdas críticas se fijan con el residuo por celda de la instantánea; el empeoramiento de una celda se mide frente al
residuo de esa celda tras el sondeo de a0. a0 conserva el estado actual (pool y P persisten entre ventanas). Con aG, P queda instalado: no se «desinstala».
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

try:                                                     # importable como paquete o cargado por ruta (bundles de Kaggle)
    from . import rad as _rad
except ImportError:                                      # pragma: no cover
    import importlib.util
    import sys
    from pathlib import Path
    _p = Path(__file__).resolve().parent / "rad.py"
    _spec = importlib.util.spec_from_file_location("observer_v0_rad", str(_p))
    _rad = importlib.util.module_from_spec(_spec)
    sys.modules["observer_v0_rad"] = _rad
    _spec.loader.exec_module(_rad)

A0 = "a0"


@dataclass(frozen=True)
class ObserverConfig:
    K: int = 500
    n_p: int = 25
    delta: float = 0.05
    gamma: float = 0.25
    critical_frac: float = 0.05
    cells_per_axis: int = 16
    uniform_floor: float = 0.5
    inert_threshold: float = 0.90
    h_chunk: int = 2048


# ------------------------------------------------------------------ puntos retenidos H
def heldout_points(lo, hi, n=8192, seed=0):
    """H: flujo Sobol propio (semilla desplazada), independiente del pool de entrenamiento y de cualquier malla de evaluación."""
    from scipy.stats import qmc
    d = qmc.Sobol(d=len(lo), scramble=True, seed=10_000_019 + int(seed)).random_base2(int(round(math.log2(n))))
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return lo_a + d * (hi_a - lo_a)


class FreshHeldout:
    """H_d: un bloque Sobol NUEVO por decisión d, de un flujo propio (semilla = [constante, semilla del observador, d]), disjunto del entrenamiento y de los bloques
    de las demás decisiones. Reutilizar un H fijo infla la mejora estimada (el observador elegiría varias veces sobre el mismo conjunto)."""

    def __init__(self, lo, hi, n=8192, seed=0):
        self.lo, self.hi, self.n, self.seed = lo, hi, int(n), int(seed)

    def block(self, k):
        from scipy.stats import qmc
        rng = np.random.default_rng([10_000_019, self.seed, int(k)])
        d = qmc.Sobol(d=len(self.lo), scramble=True, seed=rng).random_base2(int(round(math.log2(self.n))))
        lo_a, hi_a = np.asarray(self.lo, dtype=np.float64), np.asarray(self.hi, dtype=np.float64)
        return lo_a + d * (hi_a - lo_a)


def cell_ids(points, lo, hi, cells_per_axis):
    lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    q = np.clip(((points - lo_a) / (hi_a - lo_a) * cells_per_axis).astype(int), 0, cells_per_axis - 1)
    return np.ravel_multi_index(q.T, (cells_per_axis,) * points.shape[1])


# ------------------------------------------------------------------ residuo puntual
def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]


def residual_at(prob, model, X, chunk=2048):
    """Residuo PDE puntual (float64, numpy) en X (n, 2). Solo usa la PDE del problema; nunca la referencia."""
    out = []
    for lo in range(0, X.shape[0], chunk):
        xt = X[lo:lo + chunk].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            u = prob.transform(xt, model(xt))
            g = _grad(u, xt)
            uxx = _grad(g[:, 0:1], xt)[:, 0:1]
            if hasattr(prob, "pde_residual"):
                r = prob.pde_residual(u, (g[:, 0:1], g[:, 1:2]), uxx)
            elif prob.name == "burgers_pinnacle_1d":
                r = g[:, 1:2] + u * g[:, 0:1] - prob.nu * uxx
            elif prob.name == "allen_cahn":
                r = g[:, 1:2] - 0.001 * uxx - 5.0 * (u - u.pow(3))
            else:
                raise ValueError(f"problema sin residuo conocido: {prob.name}")
        out.append(r.detach().double().cpu().numpy().reshape(-1))
    return np.concatenate(out)


# ------------------------------------------------------------------ instantánea exacta
def take_snapshot(model, opt, P):
    return dict(model={k: v.detach().clone() for k, v in model.state_dict().items()}, opt=copy.deepcopy(opt.state_dict()),
                P={k: v.detach().clone() for k, v in P.items()})


def restore_snapshot(snap, model, opt, P):
    model.load_state_dict(snap["model"])
    opt.load_state_dict(copy.deepcopy(snap["opt"]))
    for k, v in snap["P"].items():
        P[k] = v.clone()


def states_equal(snap_a, snap_b):
    """Igualdad bit a bit de θ (incluidos buffers), pool y estado del optimizador."""
    if snap_a["model"].keys() != snap_b["model"].keys():
        return False
    if not all(torch.equal(snap_a["model"][k], snap_b["model"][k]) for k in snap_a["model"]):
        return False
    if not all(torch.equal(snap_a["P"][k], snap_b["P"][k]) for k in snap_a["P"]):
        return False
    sa, sb = snap_a["opt"], snap_b["opt"]
    if sa["param_groups"] != sb["param_groups"] or sa["state"].keys() != sb["state"].keys():
        return False
    for i in sa["state"]:
        if sa["state"][i].keys() != sb["state"][i].keys():
            return False
        for name, va in sa["state"][i].items():
            vb = sb["state"][i][name]
            if torch.is_tensor(va):
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


# ------------------------------------------------------------------ palancas
class Lever:
    name = "lever"

    def apply(self, prob, model, P, opt, rng):          # pragma: no cover
        raise NotImplementedError


class RadLever(Lever):
    """aR: remuestrea el pool de entrenamiento ∝ |r|^2 con piso uniforme y presupuesto de puntos FIJO (el mismo número de puntos que el pool actual)."""
    name = "aR"

    def __init__(self, lo, hi, n_candidates=8192, uniform_floor=0.5, power=2.0):
        self.lo, self.hi, self.n_candidates, self.uniform_floor, self.power = np.asarray(lo, float), np.asarray(hi, float), int(n_candidates), uniform_floor, power

    def apply(self, prob, model, P, opt, rng):
        pool = P["residual"]
        budget = int(pool.shape[0])
        cand = self.lo + rng.random((self.n_candidates, len(self.lo))) * (self.hi - self.lo)
        r = residual_at(prob, model, torch.as_tensor(cand, dtype=pool.dtype, device=pool.device))
        s = _rad.sample_rad(cand, r, budget, rng, power=self.power, uniform_fraction=self.uniform_floor)
        P["residual"] = torch.as_tensor(s["points"], dtype=pool.dtype, device=pool.device)
        return dict(effective_candidates=s["effective_candidates"], unique_count=s["unique_count"])


class GramLever(Lever):
    """aG: corrección de Gram sobre un modelo ya `attach`-ado (P = I al inicio). Variante E: recalcula los bloques con el pool vigente.
    Variante F: instala el P fijo (Gram uniforme de las bases). Para F el modelo debe adjuntarse con arm='ID' (identidad), no con 'F'."""
    name = "aG"

    def __init__(self, pre, variant="E"):
        if variant not in ("E", "F"):
            raise ValueError("variant debe ser 'E' o 'F'")
        self.pre, self.variant = pre, variant

    def apply(self, prob, model, P, opt, rng):
        if self.variant == "E":
            return self.pre.refresh(model, prob, P["residual"], opt)
        layers = self.pre._layers
        self.pre._set(model, [self.pre._fixed_blocks(layer) for layer in layers], opt)
        return dict(variant="F")


# ------------------------------------------------------------------ observador
@dataclass
class Decision:
    step: int
    chosen: str
    mode: str
    improvement: dict = field(default_factory=dict)         # acción -> mejora relativa del residuo medio frente a a0 (None si no medible)
    worse_cells: dict = field(default_factory=dict)         # acción -> nº de celdas críticas que empeoran más de gamma
    null_reason: dict = field(default_factory=dict)         # acción -> razón si el sondeo devolvió null
    h_block: int | None = None                              # índice del bloque H_d usado (None si H es fijo)
    probe_seconds: float = 0.0
    seconds: float = 0.0


class Observer:
    """mode: 'probe' (O), 'random' (O-: acción al azar por decisión, sin sondeo), 'matched' (O-m: al azar con las frecuencias `probs` de O), 'fixed:<palanca>' (la palanca en cada decisión) o 'none' (solo a0)."""

    def __init__(self, levers, lo, hi, H, seed, config=ObserverConfig(), mode="probe", clock=time.perf_counter, probs=None):
        """H: arreglo fijo (solo pruebas) o `FreshHeldout` (el bloque H_d se genera en cada decisión d; es lo preregistrado)."""
        names = [lv.name for lv in levers]
        if len(set(names)) != len(names) or A0 in names:
            raise ValueError("nombres de palanca duplicados o reservados")
        if mode.startswith("fixed:") and mode[6:] not in names:
            raise ValueError(f"palanca fija desconocida: {mode}")
        if mode not in ("probe", "random", "none", "matched") and not mode.startswith("fixed:"):
            raise ValueError(f"modo desconocido: {mode}")
        if mode == "matched":                       # O-m: acciones al azar con la frecuencia que O mostro en ese problema (probs: accion -> probabilidad, suma 1)
            if not probs or set(probs) - {A0, *names} or abs(sum(probs.values()) - 1.0) > 1e-9:
                raise ValueError("matched requiere probs {accion: p} sobre {a0, palancas} que sume 1")
        self.probs = dict(probs) if probs else None
        self.levers = {lv.name: lv for lv in levers}
        self.lo, self.hi, self.seed, self.cfg, self.mode, self.clock = lo, hi, int(seed), config, mode, clock
        self.fresh = H if isinstance(H, FreshHeldout) else None
        self.n_cells = config.cells_per_axis ** len(lo)
        self._set_H(H if self.fresh is None else self.fresh.block(0))
        self.decisions: list[Decision] = []
        self.probe_seconds = 0.0
        self.extra_steps = 0

    def _set_H(self, H):
        self.H = torch.as_tensor(H)
        self.cells = cell_ids(np.asarray(H, dtype=np.float64) if not torch.is_tensor(H) else H.cpu().numpy().astype(np.float64), self.lo, self.hi, self.cfg.cells_per_axis)

    # -- rng derivada: la misma para el sondeo y para la aplicación real de una acción
    def _rng(self, k, name):
        code = sum(ord(c) * (i + 1) for i, c in enumerate(name))
        return np.random.default_rng([self.seed, int(k), int(code)])

    def _cell_residual(self, r):
        r2 = np.asarray(r) ** 2
        s = np.bincount(self.cells, weights=r2, minlength=self.n_cells)
        n = np.bincount(self.cells, minlength=self.n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan), float(r2.mean())

    def _measure(self, prob, model):
        r = residual_at(prob, model, self.H.to(next(model.parameters()).dtype).to(next(model.parameters()).device), self.cfg.h_chunk)
        if not np.isfinite(r).all():
            raise FloatingPointError("residuo no finito en H")
        return self._cell_residual(r)

    @staticmethod
    def _train(prob, model, P, opt, n, log=None, step0=0, log_every=100, clock=time.perf_counter, t0=0.0):
        for i in range(1, n + 1):
            opt.zero_grad(set_to_none=True)
            total, _ = prob.loss(model, P)
            total.backward()
            opt.step()
            val = float(total.detach())
            if not math.isfinite(val):
                return "nonfinite_loss"
            if log is not None and (step0 + i) % log_every == 0:
                log.append({"step": step0 + i, "loss": val, "seconds": clock() - t0})
        return "ok"

    def _apply(self, name, k, prob, model, P, opt):
        if name == A0:
            return {}
        return self.levers[name].apply(prob, model, P, opt, self._rng(k, name)) or {}

    # -- una decisión
    def decide(self, k, step, prob, model, P, opt):
        t_dec = self.clock()
        if self.fresh is not None:
            self._set_H(self.fresh.block(k))                 # H_d fresco para la decisión d = k
        actions = [A0] + list(self.levers)
        dec = Decision(step=step, chosen=A0, mode=self.mode, h_block=int(k) if self.fresh is not None else None)
        if self.mode == "none":
            chosen = A0
        elif self.mode.startswith("fixed:"):
            chosen = self.mode[6:]
        elif self.mode == "random":
            chosen = actions[int(self._rng(k, "random").integers(len(actions)))]
        elif self.mode == "matched":
            u, acc, chosen = float(self._rng(k, "matched").random()), 0.0, actions[-1]
            for a in actions:
                acc += self.probs.get(a, 0.0)
                if u < acc:
                    chosen = a
                    break
        else:
            chosen = self._probe(k, step, prob, model, P, opt, actions, dec)
        dec.chosen = chosen
        if chosen != A0:
            self._apply(chosen, k, prob, model, P, opt)
        dec.seconds = self.clock() - t_dec
        self.decisions.append(dec)
        return chosen

    def _probe(self, k, step, prob, model, P, opt, actions, dec):
        t0 = self.clock()
        snap = take_snapshot(model, opt, P)
        try:
            cells_snap, _ = self._measure(prob, model)
        except FloatingPointError as e:
            dec.null_reason[A0] = f"snapshot: {e}"
            restore_snapshot(snap, model, opt, P)
            dec.probe_seconds = self.clock() - t0
            self.probe_seconds += dec.probe_seconds
            return A0
        valid = np.where(np.isfinite(cells_snap))[0]
        n_crit = max(1, int(round(self.cfg.critical_frac * len(valid))))
        critical = valid[np.argsort(cells_snap[valid])[::-1][:n_crit]]
        results = {}
        for a in actions:
            try:
                self._apply(a, k, prob, model, P, opt)
                status = self._train(prob, model, P, opt, self.cfg.n_p)
                self.extra_steps += self.cfg.n_p
                if status != "ok":
                    raise FloatingPointError(status)
                results[a] = self._measure(prob, model)
            except Exception as e:                                           # una palanca que no se puede medir -> null con razón, nunca un número inventado
                dec.null_reason[a] = f"{type(e).__name__}: {e}"
                results[a] = None
            restore_snapshot(snap, model, opt, P)                            # restauración exacta antes de la acción siguiente
        base = results.get(A0)
        best, best_gain = A0, 0.0
        for a in actions[1:]:
            if base is None or results[a] is None:
                dec.improvement[a] = None
                continue
            gain = (base[1] - results[a][1]) / max(base[1], 1e-300)
            c0, ca = base[0][critical], results[a][0][critical]
            with np.errstate(invalid="ignore", divide="ignore"):
                worse = int(np.nansum((ca - c0) / np.maximum(c0, 1e-300) > self.cfg.gamma))
            dec.improvement[a], dec.worse_cells[a] = float(gain), worse
            if gain >= self.cfg.delta and worse == 0 and gain > best_gain:
                best, best_gain = a, gain
        dec.probe_seconds = self.clock() - t0
        self.probe_seconds += dec.probe_seconds
        return best

    # -- Adam observado (mismo lazo que sweep.adam_stage / adam_with_pre, con decisión al inicio de cada ventana)
    def run_adam(self, prob, model, P, steps, opt=None, log_every=100, lr=1e-3):
        opt = opt or torch.optim.Adam(model.parameters(), lr=lr)
        hist, status, done, t0 = [], "adam_complete", 0, self.clock()
        k = 0
        while done < steps:
            self.decide(k, done, prob, model, P, opt)
            n = min(self.cfg.K, steps - done)
            st = self._train(prob, model, P, opt, n, hist, done, log_every, self.clock, t0)
            done += n
            k += 1
            if st != "ok":
                status = st
                break
        return dict(hist=hist, steps=done, status=status, seconds=self.clock() - t0, probe_seconds=self.probe_seconds, extra_steps=self.extra_steps,
                    decisions=[vars(d) for d in self.decisions], inert=self.is_inert(), optimizer=opt)

    def is_inert(self):
        """Inerte: elige a0 en más del 90 % de las decisiones (se reporta, no se esconde)."""
        if not self.decisions or self.mode != "probe":
            return False
        return sum(d.chosen == A0 for d in self.decisions) / len(self.decisions) > self.cfg.inert_threshold
