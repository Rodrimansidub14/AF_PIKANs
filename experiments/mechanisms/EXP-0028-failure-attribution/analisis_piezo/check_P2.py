"""Comprobacion de P2 sin entrenar: ¿cuanto aporta cada grupo fisico de terminos a la traza del bloque Gauss-Newton de cada arista, y cambia la DIRECCION del bloque si se balancean los grupos?
Estados finales guardados (paso 20 000) de B y E, converso y directo, semillas 20260728-30, CPU float64. Bloques por arista (O, I, G, G) con J respecto a los coeficientes efectivos.
B_bal = sum_g B_g / tr(B_g) (cada grupo con la misma traza; la normalizacion por traza de la arista de E ya borra la escala global, asi que P2 solo puede cambiar la FORMA del bloque).
Umbral del diseño (DISENO_EJECUCION.html): pasa si el grupo electrico (D constitutivas + Gauss) aporta < 5 % de la traza en >= 70 % de las aristas Y el P balanceado cambia la direccion."""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piezo"))
import run_piezo as rp  # noqa: E402

rt = rp.gp._residual_terms_module()
RES = HERE.parent / "results"
DIRS = ("piezo-s1a", "piezo-s1b", "piezo-modal-run")
SEEDS = (20260728, 20260729, 20260730)
NPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 256
torch.set_num_threads(4)
dev, dt = torch.device("cpu"), torch.float64
GROUPS = {"mec": ("constitutive_sigma_xx", "constitutive_sigma_yy", "constitutive_tau_xy"),
          "elec": ("constitutive_D_x", "constitutive_D_y", "gauss"),
          "equil": ("equilibrium_x", "equilibrium_y")}


def group_of(name):
    for g, names in GROUPS.items():
        if name in names:
            return g
    return "carga" if name == "bc_top_zero_net_charge" else "borde"


def art(prob_mode, arm, seed):
    for dd in DIRS:
        for f in glob.glob(str(RES / dd / "**" / f"piezo-{prob_mode}-{arm}-seed{seed}-artifacts.npz"), recursive=True):
            return np.load(f, allow_pickle=True)


def load_model(prob, mode, arm, seed):
    m = rp.build(prob, "B", seed, dt).to(dev)
    z = art(mode, arm, seed)
    if arm != "B":
        rp.gp.GramPreconditioner(arm, every=10 ** 9, n_points=64, chunk=2).attach(m)
    sd = {k.split("/state/")[1]: torch.tensor(z[k]) for k in z.keys() if k.startswith("s020000/state/")}
    r = m.load_state_dict(sd, strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    return m


def term_blocks(prob, model, X, chunk=16):
    coefs = tuple(c.detach() for c in rp.gp.effective_coefficients(model))
    acc, cnt = {}, {}
    for lo in range(0, rt._length(X), chunk):
        xs = rt._slice(X, lo, chunk)
        meta = {}

        def f(cc):
            terms = prob.residual_terms(rt.make_model_fn(model, cc), xs)
            meta["n"], meta["w"] = [n for _, _, n in terms], [w for _, w, _ in terms]
            return tuple(r.reshape(-1) for r, _, _ in terms)

        J = torch.func.jacrev(f)(coefs)
        for t, (nm, w) in enumerate(zip(meta["n"], meta["w"])):
            part = rt._part([a.detach().double() for a in J[t]], "edge")
            acc[nm] = part if nm not in acc else [a + p for a, p in zip(acc[nm], part)]
            cnt[nm] = cnt.get(nm, 0) + J[t][0].shape[0]
    out = {}
    for nm, parts in acc.items():
        out[nm] = [p * (1.0 / cnt[nm]) for p in parts]          # peso del termino aparte: se aplica abajo
    return out


def main():
    rows, lines = [], []
    for mode in ("converse", "direct"):
        prob = rt.install(rp.make_problem(mode, 0))
        wts = {}
        for seed in SEEDS:
            P = prob.pools(seed, dev, dt, 0)
            X = rp.gp.subset(P, NPTS)
            for arm in ("B", "E"):
                t0 = time.time()
                m = load_model(prob, mode, arm, seed)
                tb = term_blocks(prob, m, X)
                names = list(tb)
                w = {n: next(wt for _, wt, nn in prob.residual_terms(rt.make_model_fn(m, tuple(c.detach() for c in rp.gp.effective_coefficients(m))), {k: v[:2] for k, v in X.items()}) if nn == n) for n in names}
                for li in range(len(tb[names[0]])):
                    Bg = {}
                    for n in names:
                        b = tb[n][li] * w[n]
                        g = group_of(n)
                        Bg[g] = Bg[g] + b if g in Bg else b
                    O, I, G, _ = Bg["mec"].shape if "mec" in Bg else next(iter(Bg.values())).shape
                    tot = sum(Bg.values())
                    tr = {g: b.diagonal(dim1=-2, dim2=-1).sum(-1) for g, b in Bg.items()}          # (O, I)
                    ttot = tot.diagonal(dim1=-2, dim2=-1).sum(-1)
                    share = {g: (tr[g] / ttot.clamp_min(1e-300)) for g in tr}
                    live = ttot > 1e-12 * ttot.max()
                    bal = sum(b / tr[g].clamp_min(1e-300)[..., None, None] for g, b in Bg.items() if g not in ("borde", "carga"))
                    # direccion dominante y P regularizado (como E: B/m + 1e-3 I)
                    def norm(B):
                        n = B.shape[-1]
                        m_ = B.diagonal(dim1=-2, dim2=-1).sum(-1) / n
                        return B / m_.clamp_min(1e-300)[..., None, None]
                    Bn, Bb = norm(tot), norm(bal)
                    wv, V = torch.linalg.eigh(Bn)
                    wb, Vb = torch.linalg.eigh(Bb)
                    cos = (V[..., -1] * Vb[..., -1]).sum(-1).abs()
                    def pinv_sqrt(B):
                        w_, V_ = torch.linalg.eigh(B + 1e-3 * torch.eye(B.shape[-1], dtype=B.dtype))
                        return V_ @ torch.diag_embed(w_.clamp_min(1e-300).rsqrt()) @ V_.transpose(-1, -2)
                    Pn, Pb = pinv_sqrt(Bn), pinv_sqrt(Bb)
                    reld = (Pn - Pb).flatten(-2).norm(dim=-1) / Pn.flatten(-2).norm(dim=-1)
                    # cuanto cambia el condicionamiento efectivo de la tarea real: cond( P_b B_n P_b ) (regularizado) frente a cond( P_n B_n P_n )
                    def eff_cond(Pm):
                        M = Pm @ (Bn + 1e-3 * torch.eye(Bn.shape[-1], dtype=Bn.dtype)) @ Pm          # regularizado como P: con P de E da cond 1
                        e = torch.linalg.eigvalsh(M)
                        return (e[..., -1] / e[..., 0].clamp_min(1e-300))
                    cE, cB = eff_cond(Pn), eff_cond(Pb)
                    r = dict(problem=mode, arm=arm, seed=seed % 100, layer=li, edges=int(live.sum()),
                             **{f"share_{g}_med": float(share[g][live].median()) for g in share},
                             elec_lt5=float((share.get("elec", torch.zeros_like(ttot))[live] < 0.05).double().mean()),
                             mec_gt90=float((share.get("mec", torch.zeros_like(ttot))[live] > 0.90).double().mean()),
                             cos_med=float(cos[live].median()), cos_lt09=float((cos[live] < 0.9).double().mean()),
                             dP_med=float(reld[live].median()), dP_p90=float(reld[live].quantile(0.9)),
                             effcond_E_med=float(cE[live].median()), effcond_bal_med=float(cB[live].median()))
                    rows.append(r)
                print(mode, arm, seed % 100, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(HERE / "check_P2.json", "w"), indent=1)
    # resumen
    def agg(sel, key):
        v = [r[key] for r in sel]
        return np.median(v) if v else float("nan")
    out = [f"# Comprobación de P2 (balance por grupo físico), estados finales guardados, CPU, {NPTS} puntos por clave\n",
           "Grupos: mec = constitutivas mecánicas (σxx, σyy, τ); elec = constitutivas D + Gauss; equil = equilibrio; borde = condiciones de borde; carga = carga neta (directo).",
           "Cada celda: mediana sobre semillas y brazos de la mediana por aristas. `cos` = coseno del autovector dominante de B frente al de B balanceado; `dP` = diferencia relativa entre P y P balanceado.\n",
           "| problema | capa | cuota mec | cuota elec | cuota equil | cuota borde | aristas con elec<5 % | aristas con mec>90 % | cos med | aristas cos<0.9 | dP med | cond efectiva E → balanceado |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode in ("converse", "direct"):
        for li in sorted({r["layer"] for r in rows}):
            s = [r for r in rows if r["problem"] == mode and r["layer"] == li]
            out.append(f"| {mode} | {li} | {agg(s,'share_mec_med'):.2f} | {agg(s,'share_elec_med'):.3f} | {agg(s,'share_equil_med'):.3f} | {agg(s,'share_borde_med'):.3f} | {agg(s,'elec_lt5'):.2f} | {agg(s,'mec_gt90'):.2f} | {agg(s,'cos_med'):.2f} | {agg(s,'cos_lt09'):.2f} | {agg(s,'dP_med'):.2f} | {agg(s,'effcond_E_med'):.3g} → {agg(s,'effcond_bal_med'):.3g} |")
    (HERE / "check_P2.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
