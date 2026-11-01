#!/usr/bin/env python3
"""EXP-0028 Fase 2 (piezo real): propiedades a la inicializacion con el PROBLEMA DEL PAPER (PiezoProblem + `piezo_residual_terms`), CONVERSO y DIRECTO, CPU, float64, sin entrenar y sin leer la
referencia FEM. Plantilla: fase2/measure_properties_terms.py (mismas definiciones de cada propiedad, para poder compararlas con el banco y con el stand-in anterior).

Modelo RBF-PIKAN 4-16-16-8 (11 bases) con las entradas/salidas del paper, semilla s, pools del paper (2 x 2048 interiores + 256 por frontera/interfaz, lote 0), GN con 2048 puntos por clave
(gram_preconditioning.subset) y bloques por arista B = sum_t (w_t / N_t) J_t^T J_t sobre todos los terminos del adaptador (PDE = 8 de interior; el resto, borde).

Uso: python measure_properties_piezo.py --problems converse,direct --seeds 4001-4005 [--threads 6] [--out properties_piezo.csv]; luego --report para escribir properties_piezo.md
"""
import argparse, csv, math, os, sys, time
sys.dont_write_bytecode = True
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[3]))
os.environ.pop("BUNDLE_DIR", None)
sys.path.insert(0, str(HERE))
import torch
from torch.func import jacrev
import run_piezo as rp          # noqa: E402  (gp, make_problem, build)
import piezo_problem as pp      # noqa: E402

gp = rp.gp
rt = gp._residual_terms_module()
EPS = gp.EPS_REL
N_PER_KEY, CHUNK = 2048, 16
PDE_NAMES = ("constitutive_sigma_xx", "constitutive_sigma_yy", "constitutive_tau_xy", "constitutive_D_x", "constitutive_D_y", "equilibrium_x", "equilibrium_y", "gauss")
OUT = HERE / "properties_piezo.csv"


def gn_blocks_by_term(prob, model, X, coefs, chunk=CHUNK):
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
    tr = [B.reshape(-1, B.shape[-2], B.shape[-1]).diagonal(dim1=-2, dim2=-1).sum(-1) for B in blocks]
    allt = torch.cat(tr).clamp_min(1e-300)
    srt = torch.sort(allt, descending=True).values
    k = max(1, int(round(0.1 * srt.numel())))
    med = torch.stack([t.median() for t in tr])
    q = torch.quantile(allt, torch.tensor([0.05, 0.95], dtype=allt.dtype))
    return dict(edge_trace_log10_p95_p5=float(torch.log10(q[1] / q[0])), edge_trace_log10_max_min=float(torch.log10(srt[0] / srt[-1])),
                edge_trace_top10=float(srt[:k].sum() / srt.sum()), layer_trace_log10_ratio=float(torch.log10(med.max() / med.min())))


def top_frac(v, frac=0.10):
    k = max(1, int(round(frac * v.numel())))
    return float(torch.sort(v, descending=True).values[:k].sum() / v.sum())


def pool_terms(prob, model, P, chunk=512):
    out, W = {}, {}
    n = max(v.shape[0] for v in P.values())
    for lo in range(0, n, chunk):
        xs = rt._slice({k: v for k, v in P.items() if k != "interface"}, lo, chunk)
        for r, w, name in prob.residual_terms(lambda x: model(x), xs):
            out.setdefault(name, []).append(r.detach().double()); W[name] = w
    return {k: (torch.cat(v), W[k]) for k, v in out.items()}


def measure(mode, seed):
    prob = rp.make_problem(mode)
    model = prob.build("rbf", rp.WIDTH, rp.BASES, seed)
    P = prob.pools(seed)
    p = {}
    coefs = tuple(gp.layer_coefficients(l).detach() for l in model.layers)
    X = gp.subset({k: v for k, v in P.items() if k != "interface"}, N_PER_KEY)
    byterm = gn_blocks_by_term(prob, model, X, coefs)
    allB = total_blocks(byterm)
    p.update(prop_cond(allB))
    p.update(edge_trace_stats(allB))
    pde_names = [n for n in byterm if n in PDE_NAMES]
    p.update({k: v for k, v in prop_cond(total_blocks(byterm, pde_names), "pde_").items() if k.startswith("pde_condB_") and "_l" not in k or k in ("pde_P_cond_median",)})
    tr_pde, tr_bc = trace_of(total_blocks(byterm, pde_names)), trace_of(total_blocks(byterm, [n for n in byterm if n not in PDE_NAMES]))
    p.update(gn_trace_bc_over_pde=tr_bc / tr_pde, gn_trace_pde=tr_pde, gn_trace_bc=tr_bc)
    PT = pool_terms(prob, model, P)
    pde = [(PT[n][0], PT[n][1]) for n in PDE_NAMES if n in PT]
    rn = torch.sqrt(sum(r.square() for r, _ in pde))
    p.update(resid_conc_top10=top_frac(rn), resid_conc_top10_sq=top_frac(rn.square()), resid_rms0=float(rn.square().mean().sqrt()), n_pool_points=int(rn.numel()))
    contrib = torch.cat([w * r.square() / r.numel() for r, w in (PT[n] for n in PT)])
    p.update(resid_conc_all_sq_top10=top_frac(contrib))
    lt = {n: w * float(r.square().mean()) for n, (r, w) in PT.items()}
    l_pde, l_bc = sum(v for n, v in lt.items() if n in PDE_NAMES), sum(v for n, v in lt.items() if n not in PDE_NAMES)
    p.update(loss0_terms=l_pde + l_bc, loss0_pde=l_pde, loss0_bc=l_bc, bc_over_pde=l_bc / l_pde, n_loss_terms=len(lt),
             term_loss_ratio_max_min=max(lt.values()) / max(min(lt.values()), 1e-300))
    nz = {n: v for n, v in lt.items() if v > 0}
    p.update(term_loss_ratio_max_min_nonzero=max(nz.values()) / min(nz.values()), n_loss_terms_nonzero=len(nz))
    for n, v in lt.items():
        p[f"loss0_{n}"] = v
    model.zero_grad(set_to_none=True)
    total, _ = prob.loss(model, P)
    total.backward()
    p.update(loss0=float(total.detach()), terms_vs_loss_relerr=abs(l_pde + l_bc - float(total.detach())) / float(total.detach()),
             grad_norm0=math.sqrt(sum(float(q.grad.double().pow(2).sum()) for q in model.parameters() if q.grad is not None)),
             grad_norm0_coef=math.sqrt(sum(float(l.coefficients.grad.double().pow(2).sum()) for l in model.layers)))
    model.zero_grad(set_to_none=True)
    s = prob.scales
    scales = [s["u"], s["v"], s["phi"], s["stress_x"], s["stress_y"], s["shear"], s["electric_displacement"], s["electric_displacement"]]
    p.update(model_width=16, n_outputs=8, n_coupled_eqs=8, output_scale_ratio=max(scales) / min(scales), output_scale_ratio_uv_phi=s["phi"] / s["u"], rbf_width_x=0.35)
    return p


def parse_seeds(s):
    if "-" in s and "," not in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def report(csvp):
    import statistics
    rows = list(csv.DictReader(open(csvp, newline="")))
    by = {}
    for r in rows:
        by.setdefault((r["problema"], r["propiedad"]), []).append(float(r["valor"]))
    std = {}
    for r in csv.DictReader(open(HERE.parent / "fase2/properties_terms.csv", newline="")):
        if r["problema"] == "piezo_standin":
            std.setdefault(r["propiedad"], []).append(float(r["valor"]))
    bank = {}
    for r in csv.DictReader(open(HERE.parent / "cruce/propiedades_por_problema.csv", newline="")):
        vals = []
        for c in ("ac_d1e-3", "ac_d2e-4", "burgers", "kpp", "ns", "dfg"):
            try:
                vals.append(float(r[c]))
            except (ValueError, KeyError):
                pass
        if vals:
            bank[r["propiedad"]] = (min(vals), max(vals))
    keys = ["n_outputs", "n_loss_terms", "n_loss_terms_nonzero", "resid_conc_top10_sq", "resid_conc_top10", "resid_rms0", "bc_over_pde", "gn_trace_bc_over_pde", "term_loss_ratio_max_min",
            "term_loss_ratio_max_min_nonzero", "output_scale_ratio", "output_scale_ratio_uv_phi", "condB_median", "condB_min", "edge_trace_log10_p95_p5", "edge_trace_log10_max_min", "layer_trace_log10_ratio",
            "loss0", "terms_vs_loss_relerr"]
    fmt = lambda v: f"{v:.3g}"
    med = lambda xs: statistics.median(xs)
    cell = lambda xs: f"{fmt(med(xs))} [{fmt(min(xs))}, {fmt(max(xs))}]"
    probs = sorted({r["problema"] for r in rows})
    n = len({(r["problema"], r["semilla"]) for r in rows}) // max(1, len(probs))
    lines = ["# EXP-0028 Fase 2 (piezo real) - propiedades a la inicializacion con el problema del paper, sin entrenar, CPU float64", "",
             f"Semillas {sorted({r['semilla'] for r in rows})[0]}-{sorted({r['semilla'] for r in rows})[-1]} (n = {n} por problema). Celda = mediana [min, max]. Medido con `measure_properties_piezo.py` (PiezoProblem real del paper + `piezo_residual_terms`; "
             "RBF-PIKAN 4-16-16-8 de 11 bases con las entradas/salidas del paper; pools del paper: 2 x 2048 interiores + 256 por frontera; GN con 2048 por clave; pesos fijos del paper; enrutamiento en el converso). "
             "Columna `stand-in`: el piezo propio anterior (`piezo_standin` de fase2/properties_terms.md: 2 entradas, escalas SC_*, pesos 1, sin fase ni enrutamiento). Columna `banco`: rango de los 6 problemas del banco (AC x2, Burgers, KPP, NS, DFG).", "",
             "| propiedad | converso | directo | stand-in anterior | banco (6 problemas) |", "|---|---|---|---|---|"]
    for k in keys:
        c = [cell(by[(pr, k)]) if (pr, k) in by else "-" for pr in ("converse", "direct")]
        st = cell(std[k]) if k in std else "-"
        bk = f"{fmt(bank[k][0])} a {fmt(bank[k][1])}" if k in bank else "-"
        lines.append(f"| `{k}` | {c[0]} | {c[1]} | {st} | {bk} |")
    lines += ["", "Perdida inicial por termino (mediana, converso | directo):", ""]
    names = sorted({k[1] for k in by if k[1].startswith("loss0_") and k[1] not in ("loss0_terms", "loss0_pde", "loss0_bc")})
    for nme in names:
        c = [fmt(med(by[(pr, nme)])) if (pr, nme) in by else "-" for pr in ("converse", "direct")]
        lines.append(f"- `{nme}`: {c[0]} | {c[1]}")
    (HERE / "properties_piezo_tabla.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="converse,direct")
    ap.add_argument("--seeds", default="4001-4005")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    csvp = Path(a.out)
    if a.report:
        report(csvp)
        return
    torch.set_num_threads(a.threads)
    probs = a.problems.split(",")
    rows = []
    if csvp.exists():
        rows = [r for r in csv.DictReader(open(csvp, newline="")) if r["problema"] not in probs]
    for pr in probs:
        for s in parse_seeds(a.seeds):
            t0 = time.time(); p = measure(pr, s)
            for k, v in p.items():
                rows.append(dict(problema=pr, semilla=s, propiedad=k, valor=repr(float(v))))
            print(pr, s, f"{time.time() - t0:.1f}s", {k: round(p[k], 4) for k in ("condB_median", "resid_conc_top10_sq", "bc_over_pde", "gn_trace_bc_over_pde", "terms_vs_loss_relerr")}, flush=True)
            with open(csvp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["problema", "semilla", "propiedad", "valor"]); w.writeheader(); w.writerows(rows)
    print("escrito", csvp)


if __name__ == "__main__":
    main()
