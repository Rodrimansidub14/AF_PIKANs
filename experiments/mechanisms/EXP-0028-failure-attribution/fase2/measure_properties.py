#!/usr/bin/env python3
"""EXP-0028 Fase 2: bateria de propiedades de PDE medidas a la INICIALIZACION (sin entrenar, CPU).

Uso:  python measure_properties.py --problem {ac_d1e-3,ac_d2e-4,burgers,kpp,ns,all} [--seeds 4001-4010] [--out DIR] [--ns-width 16]
Escribe/actualiza fase2/properties.csv (tidy: problema,semilla,propiedad,valor) reemplazando las filas del problema pedido.
Mismo modelo, semilla y puntos que el entrenamiento de EXP-0023: se usan las clases reales (pikan_campaign.py, round1C) y
gram_preconditioning.gn_blocks / _normalize / _inv_sqrt_and_sqrt importados sin modificar. No se lee ninguna referencia (salvo cargar el objeto
del problema); no se entrena; float32 en CPU como la receta.

Propiedades (todas por semilla; las de parametros del PDE son constantes):
 (1) condB_{min,median,max}: numero de condicion de los bloques por arista, B_n + eps_rel*I con B_n = B/m (m = traza/11), eps_rel=1e-3, es decir
     el bloque que realmente factoriza P (Enmienda 1). condB_l{0,1,2}_median por capa. El B crudo (11x11, J^T J / N, N=2048 puntos del pool) es
     casi singular (rango efectivo << 11): su cond sin regularizar no esta bien definida, se reporta la regularizada. P=(B_n+eps I)^(-1/2).
     P_sv_min / P_sv_max: minimo y maximo de los valores singulares de P sobre todas las aristas; P_cond_median = mediana de sv_max/sv_min por arista.
     n_edges_dead: aristas con traza(B) <= 1e-12 * media de trazas (bloque ~0).
 (2) gram_cond_raw / gram_cond_reg: cond del Gram uniforme E[phi phi^T], z~U[-1,1], de las 11 bases (raw) y de (G/m + eps_rel I) (reg, la que usa F).
 (3) resid_conc_top10: fraccion de la masa sum|r| en el 10 % de puntos del pool de entrenamiento con mayor |r| (0.10 = uniforme).
     resid_conc_top10_sq: igual con r^2 (energia, lo que entra a la perdida). |r| = valor absoluto (escalares) o norma 2 de las 3 ecuaciones (NS).
 (4) Numeros adimensionales de los parametros del PDE (L = semi-longitud 1 del dominio espacial, T = 1; ver FORMULAS abajo).
 (5) bc_over_pde: perdida inicial de terminos de borde / terminos PDE. Solo donde hay terminos de borde en la perdida (NS); AC/Burgers/KPP imponen
     IC y BC de forma dura (u0 + t(1-x^2)NN) y no tienen ese termino: no se reportan (celda '-').
 (6) n_outputs, n_coupled_eqs (ecuaciones PDE acopladas), n_loss_terms.
 (7) loss0 (perdida total inicial), grad_norm0 (norma 2 del gradiente de la perdida sobre todos los parametros), grad_norm0_coef (solo coeficientes).

FORMULAS de (4)  [L=1 semiancho espacial; T=1 horizonte; parametros leidos del codigo de cada problema]
 Allen-Cahn  u_t = d u_xx + a(u-u^3):   damkohler = a L^2 / d (reaccion/difusion);  front_width = sqrt(d/a);  front_over_rbf = front_width / ancho_RBF_en_x
 Burgers     u_t + u u_x = nu u_xx:     peclet = U L / nu con U = max|u0| = 1 (nu = 0.01/pi);  front_width = nu/U (espesor del choque);  front_over_rbf
 Fisher-KPP  u_t = D u_xx + r u(1-u):   damkohler = r L^2 / D;  peclet = c L / D con c = 5 sqrt(r D / 6) (velocidad de la onda exacta);  front_width = 1/k, k = sqrt(r/(6D))
 NS2d-CG     Re = U L / nu = 1*1/0.01 = 100 (U = max u_inlet = 1, L = altura del inlet/escalon = 1) -> peclet (= Re, sin reaccion)
 ancho_RBF_en_x = width*(hi-lo)/2 en la primera capa (entradas mapeadas a [-1,1]; width = 1.75*2/10 = 0.35): para x en [-1,1] vale 0.35.
 Para NS no se define front_over_rbf (sin frente prescrito).
"""
import argparse, csv, math, os, sys, time
sys.dont_write_bytecode = True
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[3]))
EXP23 = ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning"
os.environ.setdefault("BUNDLE_DIR", str(EXP23 / "kaggle/c1dataset"))
sys.path.insert(0, str(EXP23 / "round1C"))
import numpy as np
import torch
from torch.func import jacrev, vmap
import arnes_round1c as a1c          # noqa: E402  (importa gram_preconditioning, pikan_campaign y los problemas 1-C reales, sin modificarlos)

gp, pc, pt = a1c.gp, a1c.pc, a1c.pt
PROBLEMS = ["ac_d1e-3", "ac_d2e-4", "burgers", "kpp", "ns"]
HOOKS_PENDIENTES = {
    "dfg": "Sin problema listo para el mecanismo: DFGRBFPIKAN (src/research_pipeline/sciml/dfg.py) usa output_scales y minibatches, gn_blocks._forward no lo replica; falta adaptador (prob.residual_terms con terminos de PDE+borde y pesos) y pools del DFG.",
    "piezo": "No existe PiezoProblem ni modelo RBF en src/ ni en la campana; falta portar el codigo de Gonzalez (8 salidas, bordes duros por levantamiento, formulacion mixta) y un forward/residual vectorial.",
}
EPS = gp.EPS_REL


# ------------------------------------------------------------------ construccion (misma que el entrenamiento)
def build(problem, seed, ns_width=16):
    """Devuelve dict(prob, model, P, pool, width, problem). float32, CPU, igual que pre_trainer/arnes_round1c."""
    dev, dt = torch.device("cpu"), torch.float32
    if problem == "ac_d1e-3":
        cfg = pt.PROBLEMS["ac"]; prob = pc.make_problem("ac", str(cfg["ref"])); width = cfg["width"]
    elif problem == "burgers":
        cfg = pt.PROBLEMS["burgers"]; prob = pc.make_problem("burgers", str(cfg["ref"])); width = cfg["width"]   # RBF48
    elif problem == "ac_d2e-4":
        prob = a1c.make_problem_1c("ac_d2e-4"); width = a1c.WIDTH["ac_d2e-4"]
    elif problem == "kpp":
        prob = a1c.make_problem_1c("kpp"); width = a1c.WIDTH["kpp"]
    elif problem == "ns":
        prob = pc.NSProblem.__new__(pc.NSProblem)     # __init__ solo carga la referencia COMSOL (para puntuar); no se necesita aqui
        width = ns_width
    else:
        raise ValueError(problem)
    model = prob.build("rbf", width, 11, seed).to(device=dev, dtype=dt)
    P = prob.pools(seed, dev, dt)
    pool = P["domain"] if problem == "ns" else P["residual"]
    return dict(prob=prob, model=model, P=P, pool=pool, width=width, problem=problem)


# ------------------------------------------------------------------ bloques Gauss-Newton
def make_res_ns(model, nu=pc.NU):
    """Residual vectorial (3,) de NS2d-CG en un punto x (2,) como funcion de los coeficientes. Local (gn_blocks solo soporta residuo escalar)."""
    def out_fn(coefs, x):
        return gp._forward(model, coefs, x.unsqueeze(0))[0]                     # (3,) = (u, v, p)
    J1 = jacrev(out_fn, argnums=1)                                              # (3, 2)
    J2 = jacrev(lambda c, x: J1(c, x), argnums=1)                               # (3, 2, 2)

    def res(coefs, x):
        o = out_fn(coefs, x); g = J1(coefs, x); h = J2(coefs, x)
        u, v = o[0], o[1]
        mx = u * g[0, 0] + v * g[0, 1] + g[2, 0] - nu * (h[0, 0, 0] + h[0, 1, 1])
        my = u * g[1, 0] + v * g[1, 1] + g[2, 1] - nu * (h[1, 0, 0] + h[1, 1, 1])
        return torch.stack((mx, my, g[0, 0] + g[1, 1]))
    return res


def gn_blocks_vec(model, res_vec, X, chunk=32):
    """Como gp.gn_blocks(..., 'edge') pero con residuo vectorial: B = sum_n sum_k J_k^T J_k / N. Lista de (O, I, G, G) float64."""
    coefs = gp.effective_coefficients(model)
    f = lambda cc, xs: vmap(res_vec, in_dims=(None, 0))(cc, xs)                 # (n, 3)
    acc, N = None, X.shape[0]
    for lo in range(0, N, chunk):
        xs = X[lo:lo + chunk]
        Jd = jacrev(lambda cc: f(cc, xs))(coefs)                                # tuple of (n, 3, O, I, G)
        part = [torch.einsum("nkoig,nkoih->oigh", j.double(), j.double()) for j in Jd]
        acc = part if acc is None else [p + q for p, q in zip(acc, part)]
    return [x / N for x in acc]


def edge_blocks(b):
    X = gp.subset(b["pool"], gp.N_POINTS)
    if b["problem"] == "ns":
        return gn_blocks_vec(b["model"], make_res_ns(b["model"]), X)
    return gp.gn_blocks(b["prob"], b["model"], X, "edge", 32)


# ------------------------------------------------------------------ propiedades
def prop_cond(blocks):
    out, allc, allsv_min, allsv_max, allPc, dead = {}, [], [], [], [], 0
    traces = []
    for B in blocks:
        B = B.reshape(-1, B.shape[-2], B.shape[-1]); traces.append(B.diagonal(dim1=-2, dim2=-1).sum(-1))
    tmean = torch.cat(traces).mean()
    for l, B in enumerate(blocks):
        B = B.reshape(-1, B.shape[-2], B.shape[-1])
        Bn, eps = gp._normalize(B, EPS)
        ev = torch.linalg.eigvalsh(Bn + eps[:, None, None] * torch.eye(B.shape[-1], dtype=B.dtype))
        cond = ev[:, -1] / ev[:, 0].clamp_min(1e-300)
        sv = torch.linalg.svdvals(gp._inv_sqrt_and_sqrt(Bn, eps)[0])
        allc.append(cond); allsv_min.append(sv[:, -1]); allsv_max.append(sv[:, 0]); allPc.append(sv[:, 0] / sv[:, -1])
        dead += int((traces[l] <= 1e-12 * tmean).sum())
        out[f"condB_l{l}_median"] = float(cond.median())
    c, pcnd = torch.cat(allc), torch.cat(allPc)
    out.update(condB_min=float(c.min()), condB_median=float(c.median()), condB_max=float(c.max()),
               P_sv_min=float(torch.cat(allsv_min).min()), P_sv_max=float(torch.cat(allsv_max).max()), P_cond_median=float(pcnd.median()),
               n_edges=int(c.numel()), n_edges_dead=dead)
    return out


def prop_gram(model):
    G = gp.uniform_gram(model.layers[0])
    out = dict(gram_cond_raw=float(torch.linalg.cond(G)))
    Gn, eps = gp._normalize(G.unsqueeze(0), EPS)
    ev = torch.linalg.eigvalsh(Gn[0] + eps[0] * torch.eye(G.shape[0], dtype=G.dtype))
    out["gram_cond_reg"] = float(ev[-1] / ev[0])
    return out


def pointwise_residual(b):
    """|r| en todos los puntos del pool de entrenamiento (los de la PDE), con las mismas expresiones que prob.loss."""
    prob, model = b["prob"], b["model"]
    xy = b["pool"].detach().requires_grad_(True)
    if b["problem"] == "ns":
        out = model(xy); u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        gu, gv, gpr = pc.grad(u, xy), pc.grad(v, xy), pc.grad(p, xy)
        uxx = pc.grad(gu[:, 0:1], xy)[:, 0:1]; uyy = pc.grad(gu[:, 1:2], xy)[:, 1:2]
        vxx = pc.grad(gv[:, 0:1], xy)[:, 0:1]; vyy = pc.grad(gv[:, 1:2], xy)[:, 1:2]
        mx = u * gu[:, 0:1] + v * gu[:, 1:2] + gpr[:, 0:1] - pc.NU * (uxx + uyy)
        my = u * gv[:, 0:1] + v * gv[:, 1:2] + gpr[:, 1:2] - pc.NU * (vxx + vyy)
        ct = gu[:, 0:1] + gv[:, 1:2]
        return torch.sqrt(mx ** 2 + my ** 2 + ct ** 2).squeeze(1).detach().double()
    u = prob.transform(xy, model(xy)); g = pc.grad(u, xy); uxx = pc.grad(g[:, 0:1], xy)[:, 0:1]
    if b["problem"] in ("kpp", "ac_d2e-4"):
        r = prob._res(u, g, uxx)
    elif b["problem"] == "burgers":
        r = g[:, 1:2] + u * g[:, 0:1] - prob.nu * uxx
    else:
        r = g[:, 1:2] - 0.001 * uxx - 5.0 * (u - u.pow(3))
    return r.abs().squeeze(1).detach().double()


def top_frac(v, frac=0.10):
    k = max(1, int(round(frac * v.numel())))
    s = torch.sort(v, descending=True).values
    return float(s[:k].sum() / s.sum())


def prop_residual(b):
    r = pointwise_residual(b)
    return dict(resid_conc_top10=top_frac(r), resid_conc_top10_sq=top_frac(r ** 2), resid_rms0=float(r.pow(2).mean().sqrt()), n_pool_points=int(r.numel()))


def prop_loss_grad(b):
    model, prob, P = b["model"], b["prob"], b["P"]
    model.zero_grad(set_to_none=True)
    total, terms = prob.loss(model, P)
    total.backward()
    gall = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in model.parameters() if p.grad is not None))
    gcoef = math.sqrt(sum(float(l.coefficients.grad.double().pow(2).sum()) for l in model.layers))
    out = dict(loss0=float(total.detach()), grad_norm0=gall, grad_norm0_coef=gcoef, n_loss_terms=len(terms))
    if b["problem"] == "ns":
        pdek = ("momentum_x", "momentum_y", "continuity")
        pde = sum(float(terms[k]) for k in pdek)
        bc = sum(float(v) for k, v in terms.items() if k not in pdek)
        out.update(loss0_pde=pde, loss0_bc=bc, bc_over_pde=bc / pde)
        for k, v in terms.items():
            out[f"loss0_{k}"] = float(v)
    model.zero_grad(set_to_none=True)
    return out


def prop_static(b):
    """Propiedades de la definicion del problema (constantes entre semillas)."""
    pr = b["problem"]
    rbf_x = b["model"].layers[0].width.item() * (b["model"].hi[0] - b["model"].lo[0]).item() / 2.0   # ancho RBF en unidades de x
    out = dict(rbf_width_x=rbf_x, model_width=b["width"])
    if pr in ("ac_d1e-3", "ac_d2e-4"):
        d = 1e-3 if pr == "ac_d1e-3" else a1c.p1c.AC_D_DOSE; a = a1c.p1c.AC_A
        fw = math.sqrt(d / a)
        out.update(damkohler=a * 1.0 / d, front_width=fw, front_over_rbf=fw / rbf_x, n_outputs=1, n_coupled_eqs=1)
    elif pr == "burgers":
        nu = b["prob"].nu; U = 1.0
        out.update(peclet=U * 1.0 / nu, front_width=nu / U, front_over_rbf=(nu / U) / rbf_x, n_outputs=1, n_coupled_eqs=1)
    elif pr == "kpp":
        D, r = a1c.p1c.KPP_D, a1c.p1c.KPP_R; c = a1c.p1c.kpp_c(); fw = 1.0 / a1c.p1c.kpp_k()
        out.update(damkohler=r * 1.0 / D, peclet=c * 1.0 / D, front_width=fw, front_over_rbf=fw / rbf_x, n_outputs=1, n_coupled_eqs=1)
    elif pr == "ns":
        out.update(peclet=1.0 * 1.0 / pc.NU, n_outputs=3, n_coupled_eqs=3)       # Re = 100
    return out


def selftest_ns(b):
    """El residuo vectorial local debe coincidir con el de prob.loss (terminos momentum/continuidad)."""
    X = gp.subset(b["pool"], 256)
    res = make_res_ns(b["model"]); coefs = gp.effective_coefficients(b["model"])
    r = vmap(res, in_dims=(None, 0))(coefs, X).detach()
    _, terms = b["prob"].loss(b["model"], {**b["P"], "domain": X})
    rm = [float(r[:, k].square().mean()) for k in range(3)]
    ref = [float(terms[k]) for k in ("momentum_x", "momentum_y", "continuity")]
    return max(abs(x - y) / (abs(y) + 1e-30) for x, y in zip(rm, ref)), rm, ref


def measure(problem, seed, ns_width=16):
    torch.manual_seed(seed)
    b = build(problem, seed, ns_width)
    p = {}
    p.update(prop_cond(edge_blocks(b)))
    p.update(prop_gram(b["model"]))
    p.update(prop_residual(b))
    p.update(prop_loss_grad(b))
    p.update(prop_static(b))
    return p


def parse_seeds(s):
    if "-" in s and "," not in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=PROBLEMS + ["all"])
    ap.add_argument("--seeds", default="4001-4010")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--ns-width", type=int, default=16, help="ancho oculto del RBF-KAN para NS2d-CG (no fijado por el spec; 16 como DFG)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    probs = PROBLEMS if a.problem == "all" else [a.problem]
    csvp = Path(a.out) / "properties.csv"
    rows = []
    if csvp.exists():
        rows = [r for r in csv.DictReader(open(csvp, newline="")) if r["problema"] not in probs]
    for pr in probs:
        if a.selftest and pr == "ns":
            err, rm, ref = selftest_ns(build("ns", parse_seeds(a.seeds)[0], a.ns_width)); print("selftest NS: max rel err", err, rm, ref)
        for s in parse_seeds(a.seeds):
            t0 = time.time(); p = measure(pr, s, a.ns_width)
            for k, v in p.items():
                rows.append(dict(problema=pr, semilla=s, propiedad=k, valor=repr(float(v))))
            print(pr, s, f"{time.time() - t0:.1f}s", {k: round(p[k], 4) for k in ("condB_median", "condB_max", "resid_conc_top10", "loss0", "grad_norm0")}, flush=True)
    order = {p: i for i, p in enumerate(PROBLEMS)}
    rows.sort(key=lambda r: (order.get(r["problema"], 99), int(r["semilla"])))
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problema", "semilla", "propiedad", "valor"]); w.writeheader(); w.writerows(rows)
    print("escrito", csvp, len(rows), "filas")


if __name__ == "__main__":
    main()
