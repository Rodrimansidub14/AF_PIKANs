"""EXP-0030, bloque H (post hoc, exploratorio): conflicto y alineacion de gradientes entre grupos fisicos de terminos, y dispersion de la escala del gradiente entre aristas, en B, E y M.
Marco (Wang, Bhartari, Li y Perdikaris, arXiv:2502.00604, lectura de resumen): conflicto tipo I = gradientes de direccion parecida con magnitudes muy distintas; tipo II = magnitudes parecidas con direcciones opuestas.
Se calcula en los estados guardados s000000, s002000, s004000, s010000, s020000 (CPU, float64, 128 puntos por clave del pool de la semilla), gradiente de la perdida ponderada de cada grupo respecto a los coeficientes efectivos
(grupos: mec, elec, equil, borde; borde incluye la carga neta). Para E y M se reporta ademas el gradiente en las coordenadas que ve Adam, c' (grad_c' = P^T grad_c por arista). Solo converso y directo, semillas 28-30."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import common  # noqa: E402

PZ = common.EXP.parent / "EXP-0028-failure-attribution" / "piezo"
sys.path.insert(0, str(PZ))
import run_piezo as rp  # noqa: E402

rt = rp.gp._residual_terms_module()
dev, dt = torch.device("cpu"), torch.float64
torch.set_num_threads(4)
STEPS = (0, 2000, 4000, 10000, 20000)
SEEDS = (28, 29, 30)
NPTS = 128
GROUPS = {"mec": rt.GROUPS["mec"], "elec": rt.GROUPS["elec"], "equil": rt.GROUPS["equil"]}


def group_of(n):
    for g, names in GROUPS.items():
        if n in names:
            return g
    return "borde"


def load(prob, mode, arm, seed, step, npz):
    m = rp.build(prob, "B", 20260700 + seed, dt).to(dev)
    if arm != "B":
        rp.gp.GramPreconditioner(arm, every=10 ** 9, n_points=64, chunk=2).attach(m)
    z = np.load(npz, allow_pickle=True)
    key = f"s{step:06d}/state/"
    sd = {k[len(key):]: torch.tensor(z[k]) for k in z.keys() if k.startswith(key)}
    if not sd:
        return None
    r = m.load_state_dict(sd, strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    return m


def group_grads(prob, model, X):
    coefs = tuple(c.detach().clone().requires_grad_(True) for c in rp.gp.effective_coefficients(model))
    terms = prob.residual_terms(rt.make_model_fn(model, coefs), X)
    loss_g = {}
    for r, w, n in terms:
        v = w * r.square().mean()
        g = group_of(n)
        loss_g[g] = loss_g[g] + v if g in loss_g else v
    out = {}
    for g, v in loss_g.items():
        gr = torch.autograd.grad(v, coefs, retain_graph=True, allow_unused=True)
        out[g] = [torch.zeros_like(c) if x is None else x.detach() for x, c in zip(gr, coefs)]
    return out


def to_prime(model, grads):
    """grad_c' = P^T grad_c por arista (identidad si no hay P)."""
    res = []
    for li, layer in enumerate(model.layers):
        g = grads[li]
        if hasattr(layer, "pre_P"):
            lay = layer.pre_layout
            gp_ = torch.einsum("kba,kb->ka", layer.pre_P.double(), lay.to_groups(g).double())
            res.append(lay.from_groups(gp_))
        else:
            res.append(g)
    return res


def flat(gl):
    return torch.cat([x.reshape(-1) for x in gl])


def edge_norms(gl):
    return torch.cat([x.reshape(-1, x.shape[-1]).norm(dim=-1) for x in gl])


def metrics(gg):
    names = [g for g in ("mec", "equil", "borde", "elec") if g in gg]
    v = {g: flat(gg[g]) for g in names}
    nrm = {g: float(v[g].norm()) for g in names}
    cos = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(v[a].norm() * v[b].norm())
            cos[f"{a}-{b}"] = float((v[a] @ v[b]) / d) if d > 0 else float("nan")
    main = [g for g in ("mec", "equil", "borde") if g in v]
    pair = [cos[f"{a}-{b}"] for i, a in enumerate(main) for b in main[i + 1:] if f"{a}-{b}" in cos]
    big = max(nrm[g] for g in main)
    small = min(nrm[g] for g in main if nrm[g] > 0)
    tot = sum(v[g] for g in names)
    en = edge_norms([sum(gg[g][li] for g in names) for li in range(len(gg[names[0]]))])
    live = en > 1e-12 * en.max()
    q = torch.quantile(en[live], torch.tensor([0.05, 0.95], dtype=en.dtype))
    return dict(norm=nrm, cos=cos, align=float(np.nanmean(pair)) if pair else float("nan"), ratio_max_min=big / small if small > 0 else float("inf"),
                edge_disp_decades=float(torch.log10(q[1] / q[0].clamp_min(1e-300))), total_norm=float(tot.norm()))


def main():
    R = common.cargar()
    rows = []
    for mode in ("converse", "direct"):
        prob = rt.install(rp.make_problem(mode, 0))
        for arm in ("B", "E", "M"):
            for sd in SEEDS:
                if (mode, arm, sd) not in R:
                    continue
                X = rp.gp.subset(prob.pools(20260700 + sd, dev, dt, 0), NPTS)
                for st in STEPS:
                    m = load(prob, mode, arm, sd, st, R[(mode, arm, sd)]["npz"])
                    if m is None:
                        continue
                    gg = group_grads(prob, m, X)
                    r = dict(problem=mode, arm=arm, seed=sd, step=st, hw=R[(mode, arm, sd)]["hw"], c=metrics(gg))
                    if arm != "B":
                        r["cprime"] = metrics({g: to_prime(m, v) for g, v in gg.items()})
                    rows.append(r)
                    print(mode, arm, sd, st, f"align {r['c']['align']:.2f} disp {r['c']['edge_disp_decades']:.2f}", flush=True)
    json.dump(rows, open(common.SALIDA / "H_alineacion.json", "w"), indent=1)
    md = ["# H. Alineación de gradientes por grupo físico y dispersión por arista (post hoc, exploratorio)\n",
          "Marco: conflicto tipo I (magnitudes distintas) y tipo II (direcciones opuestas), Wang et al. (arXiv:2502.00604, resumen). Mediana entre semillas 28-30 de: `align` = coseno medio por pares entre mec, equil y borde; `razón` = norma mayor / menor entre esos tres grupos; `disp` = décadas entre el percentil 5 y 95 de la norma del gradiente por arista. Entre paréntesis, en c' (lo que ve Adam). 128 puntos por clave; estados guardados.\n",
          "| problema | paso | brazo | align (c) | razón (c) | disp (c) | disp (c′) | cos mec-equil (c) |", "|---|---|---|---|---|---|---|---|"]
    for mode in ("converse", "direct"):
        for st in STEPS:
            for arm in ("B", "E", "M"):
                s = [r for r in rows if r["problem"] == mode and r["arm"] == arm and r["step"] == st]
                if not s:
                    continue
                med = lambda f: float(np.nanmedian([f(r) for r in s]))
                cp = f"{med(lambda r: r['cprime']['edge_disp_decades']):.2f}" if arm != "B" else "-"
                md.append(f"| {mode} | {st} | {arm} (n={len(s)}) | {med(lambda r: r['c']['align']):.2f} | {med(lambda r: r['c']['ratio_max_min']):.1f} | {med(lambda r: r['c']['edge_disp_decades']):.2f} | {cp} | {med(lambda r: r['c']['cos'].get('mec-equil', float('nan'))):.2f} |")
    (common.SALIDA / "H_alineacion.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
