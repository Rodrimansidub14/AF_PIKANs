#!/usr/bin/env python3
"""EXP-0028 Fase 2 (ampliacion): propiedades a la inicializacion con `residual_terms` (PDE + BORDE + pesos), para NS2d-CG (repeticion de `ns`, ahora con terminos de borde)
y DFG 2D-1 Re20 (adaptador `DFGTerms`, modelo DFGRBFPIKAN con output_scales). CPU, float32, sin entrenar, sin leer referencias.

Uso:  python measure_properties_terms.py --problem {ns_terms,dfg,all} [--seeds 4001-4010] [--threads 4]
Escribe fase2/properties_terms.csv (tidy: problema,semilla,propiedad,valor) reemplazando las filas del problema pedido. NO toca properties.csv.

Mismo modelo, semilla, pools y puntos del GN que el entrenamiento de la fase 1c (run_phase1c.py / run_phase1c_dfg.py importados sin modificar):
GN = gram_preconditioning.subset(pool, 2048) por clave; bloques por arista B = sum_t (w_t/N_t) J_t^T J_t sobre TODOS los terminos (misma matematica que gn_blocks_terms,
acumulada por termino para poder separar PDE de borde).

Propiedades (todas por semilla):
 cond*: como measure_properties.prop_cond, sobre B de TODOS los terminos (condB_*, P_*), y sobre B solo-PDE (pde_condB_*), para comparar con la fase 2 original (solo PDE).
 gn_trace_bc_over_pde: traza total de los bloques GN de los terminos de borde / de PDE (cuanto del precondicionador lo dicta el borde).
 resid_conc_top10 / _sq: fraccion de la masa de sqrt(sum_k r_k^2) (o r^2) en el 10 % de puntos de PDE con mayor valor, sobre el pool completo de interior (terminos PDE).
 resid_conc_all_sq_top10: igual, pero con la contribucion a la perdida w_t r_i^2 / N_t de TODOS los puntos de todos los terminos (el 10 % de los puntos con mayor contribucion).
 loss0, loss0_pde, loss0_bc, bc_over_pde (perdida por terminos con pesos, pool completo), loss0_<termino>.
 term_loss_ratio_max_min: max/min de las perdidas por termino (disparidad de escalas entre terminos).
 terms_vs_loss_relerr: |sum_t w_t mean r_t^2 - prob.loss| / prob.loss (prueba de que el adaptador reproduce la perdida del entrenamiento; pool completo).
 grad_norm0 / grad_norm0_coef: del gradiente de prob.loss.
 output_scale_ratio: max/min de output_scales del modelo (DFG: U_MEAN/PRESSURE_SCALE; NS: 1).
 peclet (= Re), n_outputs, n_coupled_eqs, n_loss_terms, model_width.
"""
import argparse, csv, math, os, sys, time
sys.dont_write_bytecode = True
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[3]))
os.environ.pop("BUNDLE_DIR", None)                      # usar el repo (src/mechanisms), no la copia del bundle plano
sys.path.insert(0, str(EXP))
import torch
from torch.func import jacrev

import run_phase1c as rns            # noqa: E402  (NS2d-CG: make_problem, pools del arnes)
import run_phase1c_dfg as rdfg       # noqa: E402

gp, rt = rns.gp, rns.rt
EPS = gp.EPS_REL
N_PER_KEY, CHUNK = 2048, 16
PROBLEMS = ["ns_terms", "dfg", "piezo_standin"]
PDE_NAMES = ("momentum_x", "momentum_y", "continuity")
PIEZO_PDE = ("sx", "sz", "tau", "Dx", "Dy", "div1", "div2", "divD")      # terminos de interior del piezo (constitutivos + equilibrio); el resto son de borde
PIEZO_DIR = ROOT / "experiments/piezo/EXP-0009-piezo-indirect-rbf"
OUT = HERE / "properties_terms.csv"


def piezo_terms(pp, model_fn, X):
    """STAND-IN del piezo (piezo_pinn.py, reimplementacion propia, NO el codigo de Daniel): mismos 8 terminos de interior y los de borde de loss_groups, peso 1.
    Derivadas en x con torch.func (solo primer orden: constitutivas y equilibrio usan derivadas primeras de las 8 salidas). Entradas xi = x/L, eta = y/H en [0,1]."""
    terms = []
    if rt._has(X, "col"):
        xi = X["col"]
        y, dy, _ = rt.point_derivs(lambda x: model_fn(x.unsqueeze(0))[0], xi, 1)          # (n,8), (n,8,2)
        sx, sz, tau, Dx, Dy = y[:, 3], y[:, 4], y[:, 5], y[:, 6], y[:, 7]
        exx = pp.SC_U / pp.L * dy[:, 0, 0]
        eyy = pp.SC_V / pp.H * dy[:, 1, 1]
        exy = 0.5 * (pp.SC_U / pp.H * dy[:, 0, 1] + pp.SC_V / pp.L * dy[:, 1, 0])
        Ex, Ey = -pp.V / pp.L * dy[:, 2, 0], -pp.V / pp.H * dy[:, 2, 1]
        sg = torch.where(xi[:, 1] < 0.5, 1.0, -1.0)
        e31, e33 = sg * pp.E31, sg * pp.E33
        S, Dc = pp.SC_S, pp.SC_D
        terms += [((S * sx - (pp.C11 * exx + pp.C12 * eyy - e31 * Ey)) / S, 1.0, "sx"),
                  ((S * sz - (pp.C12 * exx + pp.C22 * eyy - e33 * Ey)) / S, 1.0, "sz"),
                  ((S * tau - pp.G * exy) / S, 1.0, "tau"),
                  ((Dc * Dx - pp.EPS * Ex) / Dc, 1.0, "Dx"),
                  ((Dc * Dy - (e31 * exx + e33 * eyy + pp.EPS * Ey)) / Dc, 1.0, "Dy"),
                  ((dy[:, 3, 0] / pp.L + dy[:, 5, 1] / pp.H) * pp.L, 1.0, "div1"),
                  ((dy[:, 5, 0] / pp.L + dy[:, 4, 1] / pp.H) * pp.L, 1.0, "div2"),
                  ((dy[:, 6, 0] / pp.L + dy[:, 7, 1] / pp.H) * pp.L, 1.0, "divD")]
    for key, cols in (("right", (3, 5, 6)), ("left", (6,)), ("bot", (4, 5)), ("top", (4, 5))):
        if rt._has(X, key):
            o = model_fn(X[key])
            names = {3: "sx", 4: "sz", 5: "tau", 6: "Dx"}
            terms += [(o[:, c], 1.0, f"{key}_{names[c]}") for c in cols]
    return terms


class PiezoStandIn:
    """prob para gn_blocks_terms: residual_terms + loss (= piezo_pinn.loss_fn, pesos 1)."""
    name = "piezo_standin"

    def __init__(self, pp):
        self.pp = pp

    def residual_terms(self, model_fn, X):
        return piezo_terms(self.pp, model_fn, X)

    def loss(self, model, P):
        total, g = self.pp.loss_fn(model, P)
        return total, g


def build(problem, seed):
    if problem == "piezo_standin":
        sys.path.insert(0, str(PIEZO_DIR))
        import piezo_pinn as pp
        pp.DT, pp.DEV = torch.float64, torch.device("cpu")
        for k in pp.WEIGHTS:
            pp.WEIGHTS[k] = 1.0
        prob = PiezoStandIn(pp)
        model = pp.build("rbf", 16, seed)                                    # RBFKAN(2, 16, 8), 11 bases, float64, como la receta del paper
        P = pp.make_pools(seed, 2048, 256)                                   # 2048 interiores + 256 por borde (receta del paper)
        return dict(prob=prob, model=model, P=P, pde_key="col", problem=problem)
    torch.manual_seed(seed)
    dev, dt = torch.device("cpu"), torch.float32
    if problem == "ns_terms":
        import pre_trainer as pt
        prob = rt.install(rns.pc.NSProblem.__new__(rns.pc.NSProblem))        # sin leer la referencia (solo pools/loss)
        model = prob.build("rbf", rns.WIDTH, rns.BASES, seed).to(device=dev, dtype=dt)
        P = prob.pools(seed, dev, dt)
        pde_key = "domain"
    else:
        prob = rdfg.make_problem()
        model = rdfg.build(prob, seed, dev, dt)
        P = rdfg.make_pools(seed, dev, dt)
        pde_key = "interior"
    return dict(prob=prob, model=model, P=P, pde_key=pde_key, problem=problem)


# ------------------------------------------------------------------ bloques GN por termino
def gn_blocks_by_term(prob, model, X, coefs, chunk=CHUNK):
    """{nombre: (peso, N_t, [bloque (O,I,G,G) por capa, float64, SIN dividir por N_t ni ponderar])}. Misma acumulacion que residual_terms.gn_blocks_terms."""
    acc, count, weight = {}, {}, {}
    for lo in range(0, rt._length(X), chunk):
        xs = rt._slice(X, lo, chunk)
        meta = {}

        def f(cc):
            terms = prob.residual_terms(rt.make_model_fn(model, cc), xs)
            meta["names"], meta["w"] = [n for _, _, n in terms], [w for _, w, _ in terms]
            return tuple(r.reshape(-1) for r, _, _ in terms)

        J = jacrev(f)(tuple(coefs))
        for t, (name, w) in enumerate(zip(meta["names"], meta["w"])):
            Jt = [a.detach().double() for a in J[t]]
            part = rt._part(Jt, "edge")
            acc[name] = part if name not in acc else [a + p for a, p in zip(acc[name], part)]
            count[name] = count.get(name, 0) + Jt[0].shape[0]
            weight[name] = w
    return {n: (weight[n], count[n], acc[n]) for n in acc}


def total_blocks(byterm, names=None):
    out = None
    for n, (w, N, parts) in byterm.items():
        if names is not None and n not in names:
            continue
        sc = [p * (w / N) for p in parts]
        out = sc if out is None else [a + s for a, s in zip(out, sc)]
    return out


def trace_of(blocks):
    return sum(float(B.reshape(-1, B.shape[-2], B.shape[-1]).diagonal(dim1=-2, dim2=-1).sum()) for B in blocks)


def prop_cond(blocks, prefix=""):
    out, allc, smin, smax, allPc, dead, traces = {}, [], [], [], [], 0, []
    for B in blocks:
        B = B.reshape(-1, B.shape[-2], B.shape[-1]); traces.append(B.diagonal(dim1=-2, dim2=-1).sum(-1))
    tmean = torch.cat(traces).mean()
    for l, B in enumerate(blocks):
        B = B.reshape(-1, B.shape[-2], B.shape[-1])
        Bn, eps = gp._normalize(B, EPS)
        ev = torch.linalg.eigvalsh(Bn + eps[:, None, None] * torch.eye(B.shape[-1], dtype=B.dtype))
        cond = ev[:, -1] / ev[:, 0].clamp_min(1e-300)
        sv = torch.linalg.svdvals(gp._inv_sqrt_and_sqrt(Bn, eps)[0])
        allc.append(cond); smin.append(sv[:, -1]); smax.append(sv[:, 0]); allPc.append(sv[:, 0] / sv[:, -1])
        dead += int((traces[l] <= 1e-12 * tmean).sum())
        out[f"{prefix}condB_l{l}_median"] = float(cond.median())
    c = torch.cat(allc)
    out.update({f"{prefix}condB_min": float(c.min()), f"{prefix}condB_median": float(c.median()), f"{prefix}condB_max": float(c.max()),
                f"{prefix}P_sv_min": float(torch.cat(smin).min()), f"{prefix}P_sv_max": float(torch.cat(smax).max()),
                f"{prefix}P_cond_median": float(torch.cat(allPc).median()), f"{prefix}n_edges": int(c.numel()), f"{prefix}n_edges_dead": dead})
    return out


def edge_trace_stats(blocks):
    """Escala de los bloques entre aristas (lo que J reescala): traza de cada bloque B_e; p95/p5 y max/min entre aristas (log10), fraccion de la traza total en el 10 % de aristas
    con mayor traza, y razon de medianas de traza entre capas (max/min)."""
    tr = [B.reshape(-1, B.shape[-2], B.shape[-1]).diagonal(dim1=-2, dim2=-1).sum(-1) for B in blocks]
    allt = torch.cat(tr).clamp_min(1e-300)
    srt = torch.sort(allt, descending=True).values
    k = max(1, int(round(0.1 * srt.numel())))
    med = torch.stack([t.median() for t in tr])
    q = torch.quantile(allt, torch.tensor([0.05, 0.95], dtype=allt.dtype))
    return dict(edge_trace_log10_p95_p5=float(torch.log10(q[1] / q[0])), edge_trace_log10_max_min=float(torch.log10(srt[0] / srt[-1])),
                edge_trace_top10=float(srt[:k].sum() / srt.sum()), layer_trace_log10_ratio=float(torch.log10(med.max() / med.min())))


def prop_gram(model):
    G = gp.uniform_gram(rt._layers_path(model)[1][0])
    Gn, eps = gp._normalize(G.unsqueeze(0), EPS)
    ev = torch.linalg.eigvalsh(Gn[0] + eps[0] * torch.eye(G.shape[0], dtype=G.dtype))
    return dict(gram_cond_raw=float(torch.linalg.cond(G)), gram_cond_reg=float(ev[-1] / ev[0]))


def top_frac(v, frac=0.10):
    k = max(1, int(round(frac * v.numel())))
    return float(torch.sort(v, descending=True).values[:k].sum() / v.sum())


def pool_terms(prob, model, P, chunk=1024):
    """Residuos por termino sobre el POOL COMPLETO, con los coeficientes tal cual (sin precondicionar: init) -> {nombre: (r (N_t,), w)}."""
    fn = lambda x: model(x)
    out, W = {}, {}
    n = max(v.shape[0] for v in P.values())
    for lo in range(0, n, chunk):
        xs = rt._slice(P, lo, chunk)
        for r, w, name in prob.residual_terms(fn, xs):
            out.setdefault(name, []).append(r.detach().double()); W[name] = w
    return {k: (torch.cat(v), W[k]) for k, v in out.items()}


def measure(problem, seed):
    b = build(problem, seed)
    prob, model, P = b["prob"], b["model"], b["P"]
    p = {}
    coefs = tuple(gp.layer_coefficients(l).detach() for l in rt._layers_path(model)[1])
    X = gp.subset(P, N_PER_KEY)
    byterm = gn_blocks_by_term(prob, model, X, coefs)
    allB = total_blocks(byterm)
    p.update(prop_cond(allB))
    p.update(edge_trace_stats(allB))
    PDE = PIEZO_PDE if problem == "piezo_standin" else PDE_NAMES
    pde_names = [n for n in byterm if n in PDE]
    p.update({k: v for k, v in prop_cond(total_blocks(byterm, pde_names), "pde_").items() if k.startswith("pde_condB_") and "_l" not in k or k in ("pde_P_cond_median",)})
    tr_pde, tr_bc = trace_of(total_blocks(byterm, pde_names)), trace_of(total_blocks(byterm, [n for n in byterm if n not in PDE]))
    p.update(gn_trace_bc_over_pde=tr_bc / tr_pde, gn_trace_pde=tr_pde, gn_trace_bc=tr_bc)
    p.update(prop_gram(model))
    # residuos sobre el pool completo
    PT = pool_terms(prob, model, P)
    pde = [(PT[n][0], PT[n][1]) for n in PDE if n in PT]
    rn = torch.sqrt(sum(r.square() for r, _ in pde))
    p.update(resid_conc_top10=top_frac(rn), resid_conc_top10_sq=top_frac(rn.square()), resid_rms0=float(rn.square().mean().sqrt()), n_pool_points=int(rn.numel()))
    contrib = torch.cat([w * r.square() / r.numel() for r, w in (PT[n] for n in PT)])
    p.update(resid_conc_all_sq_top10=top_frac(contrib))
    lt = {n: w * float(r.square().mean()) for n, (r, w) in PT.items()}
    l_pde, l_bc = sum(v for n, v in lt.items() if n in PDE), sum(v for n, v in lt.items() if n not in PDE)
    p.update(loss0_terms=l_pde + l_bc, loss0_pde=l_pde, loss0_bc=l_bc, bc_over_pde=l_bc / l_pde, n_loss_terms=len(lt),
             term_loss_ratio_max_min=max(lt.values()) / max(min(lt.values()), 1e-300))
    for n, v in lt.items():
        p[f"loss0_{n}"] = v
    # perdida/gradiente reales del entrenamiento
    model.zero_grad(set_to_none=True)
    total, _ = prob.loss(model, P)
    total.backward()
    p.update(loss0=float(total.detach()), terms_vs_loss_relerr=abs(l_pde + l_bc - float(total.detach())) / float(total.detach()),
             grad_norm0=math.sqrt(sum(float(q.grad.double().pow(2).sum()) for q in model.parameters() if q.grad is not None)),
             grad_norm0_coef=math.sqrt(sum(float(l.coefficients.grad.double().pow(2).sum()) for l in rt._layers_path(model)[1])))
    model.zero_grad(set_to_none=True)
    # estaticas
    sc = getattr(model, "output_scales", None)
    if problem == "piezo_standin":
        pp = prob.pp
        scales = [pp.SC_U, pp.SC_V, pp.V, pp.SC_S, pp.SC_S, pp.SC_S, pp.SC_D, pp.SC_D]          # u, v, phi, sx, sz, tau, Dx, Dy (unidades mezcladas)
        p.update(model_width=16, n_outputs=8, n_coupled_eqs=8, output_scale_ratio=max(scales) / min(scales), output_scale_ratio_uv_phi=pp.V / pp.SC_U, rbf_width_x=0.35)
    else:
        p.update(model_width=16, n_outputs=3, n_coupled_eqs=3, output_scale_ratio=float(sc.max() / sc.min()) if sc is not None else 1.0,
                 peclet=(1.0 / rt.NS_NU) if problem == "ns_terms" else rdfg.dfg.U_MEAN * rdfg.dfg.CYLINDER_RADIUS * 2 / rdfg.dfg.NU, rbf_width_x=0.35)
    return p


def parse_seeds(s):
    if "-" in s and "," not in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=PROBLEMS + ["all"])
    ap.add_argument("--seeds", default="4001-4010")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    probs = PROBLEMS if a.problem == "all" else [a.problem]
    csvp = Path(a.out)
    rows = []
    if csvp.exists():
        rows = [r for r in csv.DictReader(open(csvp, newline="")) if r["problema"] not in probs]
    for pr in probs:
        for s in parse_seeds(a.seeds):
            t0 = time.time(); p = measure(pr, s)
            for k, v in p.items():
                rows.append(dict(problema=pr, semilla=s, propiedad=k, valor=repr(float(v))))
            print(pr, s, f"{time.time() - t0:.1f}s", {k: round(p[k], 4) for k in ("condB_median", "resid_conc_top10", "bc_over_pde", "gn_trace_bc_over_pde", "terms_vs_loss_relerr")}, flush=True)
            with open(csvp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["problema", "semilla", "propiedad", "valor"]); w.writeheader(); w.writerows(rows)
    print("escrito", csvp)


if __name__ == "__main__":
    main()
