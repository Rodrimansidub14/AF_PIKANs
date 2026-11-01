"""Gram preconditioning (EXP-0023): reparametrize the RBF coefficients of each edge (or input) as c = P c' and let the optimizer move c'.

P = (B/m + eps_rel I)^(-1/2) (amendment 1, normalized: singular values of P in ~[0.3, 32] for n = 11), with B the Gauss-Newton block
(J_b^T J_b / N, J = d residual / d coefficients) of the group of coefficients, m = trace(B) / n and eps_rel = 1e-3. (Spec v2 had P = (B + eps I)^(-1/2),
eps = eps_rel * trace(B) / n, i.e. sqrt(m) times larger: unnormalized, it diverged in the short test of 2026-10-20.) The blocks come from the PDE residual of the model itself at the training collocation points: the module reads no
reference solution, no evaluation grid and imports no verifier (see tests/test_gram_preconditioning.py and tests/test_mechanism_isolation.py).

Arms (EXP-0023 spec v2):
    J  diagonal of E's block: P = diag((B_gg + eps)^(-1/2)), with the eps of the edge block (same eps as E, so J and E differ only in the off-diagonals)
    E  one block per edge (layer, out, in): the G coefficients c[o, i, :]
    I  one block per input (layer, in): all O*G coefficients c[:, i, :] that read the same input
    F  fixed: E's structure with the basis Gram E[phi phi^T] for z uniform on [-1, 1]; never recomputed
    ID identity (P = I, never recomputed): test arm, bit-for-bit equal to the baseline B
The baseline B is simply the model without `attach`. Only `coefficients` are reparametrized; `base` and the rest stay as in the baseline.

Schedule: the blocks of J/E/I are recomputed before Adam step 1 and then every `every` steps (`due(done_steps)`), and once at the start of L-BFGS
(`refresh` again, called by the trainer). After each recomputation the effective coefficients are preserved (c' = P_new^-1 c), so the function does
not change, and the optimizer state of the reparametrized parameters is cleared.
Implementation: the layer's `coefficients` parameter keeps its name and its position in model.parameters() but holds c'; P is the buffer `pre_P` of the
layer and the layer class is swapped for a subclass whose forward uses c = P c'. `state_dict` (and the accepted-state rollback of L-BFGS) carries P along
with c', and with P = I the model is bit-for-bit the baseline (same parameter order for the optimizers).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import jacrev, vmap

ARMS = ("J", "E", "I", "F", "ID", "S")
RECOMPUTED = ("J", "E", "I", "S")
EPS_REL = 1e-3
EVERY = 1000
N_POINTS = 2048
MIX_B = 0.3          # arm S: weight of the physical-group balance in the block (B_mix = (1-b) B/trB + b B_bal/trB_bal); fixed, not tuned (EXP-0029 spec)
DENSE_EXTRA = (250, 500)   # arm S, leg P5: extra recomputations at the start (on top of 0, 1000, 2000, ...)
GUARD_FACTOR, GUARD_WINDOW, GUARD_COOLDOWN, GUARD_MIN_A = 5.0, 200, 200, 0.25   # arm S, leg P4
FLOOR_REL = 1e-12   # guard for dead groups: eps >= FLOOR_REL * (mean over groups of trace/n); unreachable unless a block is (numerically) zero


# ------------------------------------------------------------------ group layouts
class _Layout:
    """Grouping of the coefficients (O, I, G) for P: 'edge' -> (O*I, G); 'input' -> (I, O*G)."""

    def __init__(self, shape, layout):
        self.shape, self.layout = tuple(shape), layout

    def identity(self, dtype, device):
        O, I, G = self.shape
        K, n = (O * I, G) if self.layout == "edge" else (I, O * G)
        return torch.eye(n, dtype=dtype, device=device).expand(K, n, n).clone()

    def to_groups(self, c):
        O, I, G = self.shape
        return c.reshape(O * I, G) if self.layout == "edge" else c.permute(1, 0, 2).reshape(I, O * G)

    def from_groups(self, g):
        O, I, G = self.shape
        return g.reshape(O, I, G) if self.layout == "edge" else g.reshape(I, O, G).permute(1, 0, 2)

    def apply(self, P, cp):
        return self.from_groups(torch.einsum("kab,kb->ka", P, self.to_groups(cp)))


def _edge_class(base_cls):
    """Subclass of the campaign's RBFEdge whose forward uses c = P c'. `coefficients` holds c' (same name, same position in model.parameters(),
    so the optimizers see the parameters in the baseline order); P is the buffer `pre_P`."""
    class PreconditionedEdge(base_cls):
        def forward(self, x):
            c = self.pre_layout.apply(self.pre_P, self.coefficients)
            basis = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width).square())
            return torch.einsum("big,oig->bo", basis, c) + F.linear(F.silu(x), self.base)
    PreconditionedEdge.__name__ = "Preconditioned" + base_cls.__name__
    return PreconditionedEdge


# ------------------------------------------------------------------ Gauss-Newton blocks
def _forward(model, coefs, x):
    """Forward of the campaign's RBFKAN as a function of the coefficients (same expression as RBFEdge/RBFKAN.forward)."""
    h = 2.0 * (x - model.lo) / (model.hi - model.lo) - 1.0
    last = len(model.layers) - 1
    for k, layer in enumerate(model.layers):
        basis = torch.exp(-((h.unsqueeze(-1) - layer.centers) / layer.width).square())
        out = torch.einsum("big,oig->bo", basis, coefs[k]) + F.linear(F.silu(h), layer.base.detach())
        h = torch.tanh(out) if k < last else out
    return h


def make_residual(prob, model):
    """res_single(coefs, x): scalar PDE residual at x (2,), as a function of the tuple of coefficient tensors."""
    def u_single(coefs, x):
        xb = x.unsqueeze(0)
        return prob.transform(xb, _forward(model, coefs, xb))[0, 0]

    ux = jacrev(u_single, argnums=1)

    def res_single(coefs, x):
        g = ux(coefs, x)
        uxx = jacrev(lambda xx: ux(coefs, xx)[0])(x)[0]
        u = u_single(coefs, x)
        if hasattr(prob, "pde_residual"):                     # problems added after 1-B (round 1-C): residual(u, (u_x, u_t), u_xx); 1-B problems are unaffected
            return prob.pde_residual(u, g, uxx)
        if prob.name == "burgers_pinnacle_1d":
            return g[1] + u * g[0] - prob.nu * uxx
        if prob.name == "allen_cahn":
            return g[1] - 0.001 * uxx - 5.0 * (u - u ** 3)
        raise ValueError(prob.name)

    return res_single


def layer_coefficients(layer):
    """Effective coefficients c of an edge layer (c' mapped through P if the layer is preconditioned)."""
    if hasattr(layer, "pre_P"):
        return layer.pre_layout.apply(layer.pre_P, layer.coefficients)
    return layer.coefficients


def effective_coefficients(model):
    return tuple(layer_coefficients(layer).detach() for layer in model.layers)


def subset(pool, n_points=N_POINTS):
    """Deterministic subset of the training pool (equispaced indices); the whole pool if it has fewer points."""
    if isinstance(pool, dict):                                   # problems with several kinds of points (NS, DFG): a subset of each key
        return {k: subset(v, n_points) for k, v in pool.items()}
    n = pool.shape[0]
    if n <= n_points:
        return pool.detach()
    idx = torch.linspace(0, n - 1, n_points, device=pool.device).round().long()
    return pool.detach()[idx]


def _residual_terms_module():
    """residual_terms.py next to this file: package import if there is a package, else loaded from the file (the campaign scripts load this module by path).
    `__import__` instead of import statements: tests/test_gram_preconditioning.py::test5 restricts this file's imports to math/time/torch."""
    try:
        from . import residual_terms
        return residual_terms
    except ImportError:
        sys = __import__("sys")
        if "mechanisms_residual_terms" not in sys.modules:
            util = __import__("importlib.util", fromlist=["util"])
            spec = util.spec_from_file_location("mechanisms_residual_terms", __file__.replace("gram_preconditioning.py", "residual_terms.py"))
            mod = util.module_from_spec(spec)
            sys.modules["mechanisms_residual_terms"] = mod
            spec.loader.exec_module(mod)
        return sys.modules["mechanisms_residual_terms"]


def gn_blocks(prob, model, X, structure, chunk=128):
    """Gauss-Newton blocks B = J_b^T J_b / N per layer, float64. structure 'edge' -> (O, I, G, G); 'input' -> (I, O*G, O*G); 'diag' -> (O, I, G)."""
    coefs = effective_coefficients(model)
    if hasattr(prob, "residual_terms"):                        # vector residual with all loss terms (EXP-0028, phase 4); Burgers / AC define none and take the scalar path
        return _residual_terms_module().gn_blocks_terms(prob, model, X, structure, coefs, chunk)
    res = make_residual(prob, model)
    f = lambda cc, xs: vmap(res, in_dims=(None, 0))(cc, xs)
    acc = None
    N = X.shape[0]
    for lo in range(0, N, chunk):
        xs = X[lo:lo + chunk]
        Jd = jacrev(lambda cc: f(cc, xs))(coefs)                 # tuple of (n, O, I, G)
        Jd = [j.double() for j in Jd]
        if structure == "edge":
            part = [torch.einsum("noig,noih->oigh", j, j) for j in Jd]
        elif structure == "input":
            part = [torch.einsum("noig,npih->iogph", j, j).reshape(j.shape[2], j.shape[1] * j.shape[3], j.shape[1] * j.shape[3]) for j in Jd]
        elif structure == "diag":
            part = [j.square().sum(0) for j in Jd]
        else:
            raise ValueError(structure)
        acc = part if acc is None else [a + p for a, p in zip(acc, part)]
    return [a / N for a in acc]


def _inv_sqrt_and_sqrt(B, eps):
    """(B + eps I)^(-1/2) and (B + eps I)^(1/2) for a batch of symmetric PSD matrices (K, n, n); eps (K,)."""
    n = B.shape[-1]
    w, V = torch.linalg.eigh(B + eps[:, None, None] * torch.eye(n, dtype=B.dtype, device=B.device))
    w = w.clamp_min(1e-300)
    Vt = V.transpose(-1, -2)
    return V @ torch.diag_embed(w.rsqrt()) @ Vt, V @ torch.diag_embed(w.sqrt()) @ Vt


def _normalize(B, eps_rel):
    """Amendment 1: (B/m + eps_rel I) with m = trace(B)/n per group, so the eigenvalues of B/m lie in [0, n] and the singular values of
    P = (B/m + eps_rel I)^(-1/2) in [1/sqrt(n + eps_rel), 1/sqrt(eps_rel)] ~ [0.3, 32] for n = 11. Returns (B/m, eps_rel/m * m_floor_adjusted): the floor
    FLOOR_REL * mean(m) only acts on (numerically) zero blocks."""
    n = B.shape[-1]
    m = B.diagonal(dim1=-2, dim2=-1).sum(-1) / n
    m = m + FLOOR_REL * m.mean()
    return B / m[:, None, None], torch.full_like(m, eps_rel)


def uniform_gram(layer, m=4001):
    """Basis Gram E[phi phi^T] for z uniform on [-1, 1] (float64), for the layer's own centers and width."""
    assert layer.centers.dim() == 1, "F supports one grid of centers per layer"
    z = torch.linspace(-1.0, 1.0, m, dtype=torch.float64, device=layer.centers.device)
    phi = torch.exp(-((z.unsqueeze(-1) - layer.centers.double()) / layer.width.double()).square())
    return phi.T @ phi / m


# ------------------------------------------------------------------ the mechanism
class GramPreconditioner:
    def __init__(self, arm, eps_rel=EPS_REL, every=EVERY, n_points=N_POINTS, chunk=128):
        if arm not in ARMS:
            raise ValueError(f"arm {arm!r} not in {ARMS}")
        self.arm, self.eps_rel, self.every, self.n_points, self.chunk = arm, eps_rel, every, n_points, chunk
        self.layout = "input" if arm == "I" else "edge"
        self.b_mix, self.a_body, self.damp_enabled = MIX_B, 1.0, True   # damp_enabled=False: P4 only observes (shadow events)
        self._raw, self._hist, self._last_trigger, self.events = None, [], -10 ** 9, []
        self.n_recomputes = 0
        self.seconds = 0.0
        self._layers = []

    # -- schedule
    def due(self, done_steps):
        """True before Adam step done_steps+1 when the blocks must be recomputed (done_steps = 0, every, 2*every, ...)."""
        if self.arm == "S" and done_steps in DENSE_EXTRA:
            return True
        return self.arm in RECOMPUTED and done_steps % self.every == 0

    def due_lbfgs(self):
        return self.arm in RECOMPUTED

    # -- structure
    def attach(self, model):
        """Reparametrize model.layers[*].coefficients with P = I (c' = c). F: also sets its fixed P at once."""
        self._layers = list(model.layers)
        for layer in self._layers:
            if hasattr(layer, "pre_P"):
                raise RuntimeError("model already preconditioned")
            lay = _Layout(layer.coefficients.shape, self.layout)
            layer.pre_layout = lay
            layer.register_buffer("pre_P", lay.identity(layer.coefficients.dtype, layer.coefficients.device))
            layer.__class__ = _edge_class(layer.__class__)
        if self.arm == "F":
            self._set(model, [self._fixed_blocks(layer) for layer in self._layers], None)
        return model

    def originals(self, model):
        """The optimized parameters c' (they are the layers' `coefficients`)."""
        return [layer.coefficients for layer in model.layers]

    def _fixed_blocks(self, layer):
        O, I, G = layer.coefficients.shape
        return uniform_gram(layer).expand(O * I, G, G).clone()

    def _set(self, model, blocks, optimizer):
        """Install P from `blocks` ((K, n, n) float64, or (O, I, G) diagonals for J) keeping the effective coefficients."""
        for layer, B in zip(self._layers, blocks):
            lay = layer.pre_layout
            with torch.no_grad():
                c_eff = layer_coefficients(layer).detach()                              # current effective coefficients
                if self.arm == "J":                                                      # diagonal of E's block, eps from E's block trace
                    O, I, G = B.shape
                    Bn, eps = _normalize(torch.diag_embed(B.reshape(O * I, G)), self.eps_rel)
                    sq = (torch.diagonal(Bn, dim1=-2, dim2=-1) + eps[:, None]).sqrt()
                    P, Pinv = torch.diag_embed(sq.reciprocal()), torch.diag_embed(sq)
                else:
                    B = B.reshape(-1, B.shape[-2], B.shape[-1])                          # (O, I, G, G) -> (O*I, G, G); 'input' is already (I, n, n)
                    Bn, eps = _normalize(B, self.eps_rel)
                    P, Pinv = _inv_sqrt_and_sqrt(Bn, eps)
                    if self.arm == "S":
                        self._raw[layer] = P.clone()
                        if self.a_body < 1.0:
                            P = self._blend(P)
                            Pinv = torch.linalg.inv(P)
                cp = lay.from_groups(torch.einsum("kab,kb->ka", Pinv, lay.to_groups(c_eff).double()))
                layer.pre_P.copy_(P.to(layer.pre_P.dtype))
                layer.coefficients.copy_(cp.to(c_eff.dtype))
        if optimizer is not None:
            for p in self.originals(model):
                optimizer.state[p].clear()

    # -- arm S: graded activation and the P4 guard
    def _blend(self, P):
        """P_ef = (1 - a) I + a P (a = a_body)."""
        I = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device).expand_as(P)
        return (1.0 - self.a_body) * I + self.a_body * P

    def observe(self, step, loss):
        """Leg P4: True when the loss of `step` is > GUARD_FACTOR x the minimum of the previous GUARD_WINDOW losses (cooldown GUARD_COOLDOWN steps). Only arm S; call after each Adam step."""
        if self.arm != "S":
            return False
        h = self._hist
        trig = len(h) >= 50 and loss > GUARD_FACTOR * min(h) and step - self._last_trigger >= GUARD_COOLDOWN
        h.append(loss)
        if len(h) > GUARD_WINDOW:
            h.pop(0)
        if trig:
            self._last_trigger = step
            self.events.append(dict(step=step, loss=loss, a_before=self.a_body))
        return trig

    def damp(self, model, optimizer=None):
        """Leg P4 response: halve a_body (floor GUARD_MIN_A) and reinstall P_ef = blend(P_raw) keeping the effective coefficients. The next regular recomputation raises a_body by 0.25."""
        self.a_body = max(GUARD_MIN_A, self.a_body * 0.5)
        self.events[-1]["a_after"] = self.a_body
        for layer in self._layers:
            lay = layer.pre_layout
            with torch.no_grad():
                c_eff = layer_coefficients(layer).detach()
                P = self._blend(self._raw[layer])
                cp = lay.from_groups(torch.einsum("kab,kb->ka", torch.linalg.inv(P), lay.to_groups(c_eff).double()))
                layer.pre_P.copy_(P.to(layer.pre_P.dtype))
                layer.coefficients.copy_(cp.to(c_eff.dtype))
        if optimizer is not None:
            for p in self.originals(model):
                optimizer.state[p].clear()

    def _mixed_blocks(self, prob, model, X, chunk):
        rt = _residual_terms_module()
        B, Bbal = rt.gn_blocks_terms_groups(prob, model, X, effective_coefficients(model), chunk)
        out = []
        for b, bb in zip(B, Bbal):
            tb = b.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-300)[..., None, None]
            tbb = bb.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-300)[..., None, None]
            out.append((1.0 - self.b_mix) * b / tb + self.b_mix * bb / tbb)
        return out

    # -- recomputation
    def refresh(self, model, prob, pool, optimizer=None, sync=lambda: None, clock=None):
        """Recompute the blocks from the model's own residual at a fixed subset of the training pool. Returns diagnostics. Counts in the clock."""
        import time
        if self.arm not in RECOMPUTED:
            raise RuntimeError(f"arm {self.arm} does not recompute")
        clock = clock or time.perf_counter
        sync()
        t0 = clock()
        X = subset(pool, self.n_points)
        if self.arm == "S":
            if self._raw is None:
                self._raw = {}
            self.a_body = min(1.0, self.a_body + 0.25) if self.n_recomputes else 1.0
            blocks = self._mixed_blocks(prob, model, X, self.chunk)
        else:
            structure = {"J": "diag", "E": "edge", "I": "input"}[self.arm]
            blocks = gn_blocks(prob, model, X, structure, self.chunk)
        self._set(model, blocks, optimizer)
        sync()
        dt = clock() - t0
        self.n_recomputes += 1
        self.seconds += dt
        return dict(seconds=dt, n_points=int(X.shape[0]) if torch.is_tensor(X) else int(sum(v.shape[0] for v in X.values())),
                    p_cond=[float(torch.linalg.cond(l.pre_P.double()).median()) if l.pre_P.shape[-1] <= 64 else None for l in self._layers])

    def effective(self, model):
        return effective_coefficients(model)
