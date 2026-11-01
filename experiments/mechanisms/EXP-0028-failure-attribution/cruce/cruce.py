#!/usr/bin/env python3
"""EXP-0028: cruce efecto (fase 1/1c, scored.json) x propiedades (fase 2). DESCRIPTIVO: n = 6 problemas, sin p-values como evidencia.

Lee: results/phase1_scored.json (AC 1e-3, AC 2e-4, Burgers, KPP), results/phase1c_scored.json (NS, l2re_flat), results/phase1c_dfg_scored.json (DFG, l2re_flat),
      fase2/properties.csv (AC, Burgers, KPP; PDE sin borde: estos tres no tienen terminos de borde, BC duras) y fase2/properties_terms.csv (NS y DFG con residual_terms).
Escribe en cruce/: tabla_problema_brazo.csv, tabla_wide_{final,adam}.csv, propiedades_por_problema.csv, spearman_efecto_propiedad.csv, separacion_clase.csv,
                   heatmap_efecto.{png,pdf}, heatmap_spearman.{png,pdf}, dispersion_efecto_propiedad.{png,pdf}.
"""
import csv, json, math, statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROBS = ["ac_d1e-3", "ac_d2e-4", "burgers", "kpp", "ns", "dfg"]
LABEL = {"ac_d1e-3": "AC d=1e-3", "ac_d2e-4": "AC d=2e-4", "burgers": "Burgers", "kpp": "Fisher-KPP", "ns": "NS2d-CG", "dfg": "DFG"}
ARMS = ["F", "J", "E", "I", "S", "R", "C", "W"]
HEAT_ARMS = ["F", "J", "E", "I", "S", "R", "C"]
CORE = ["F", "J", "E", "I"]


# ------------------------------------------------------------------ efectos
def load_effects():
    rows = []
    p1 = json.load(open(EXP / "results/phase1_scored.json"))
    src = {"ac_d1e-3": p1["ac"], "ac_d2e-4": p1["ac2e4"], "burgers": p1["burgers"], "kpp": p1["kpp"],
           "ns": json.load(open(EXP / "results/phase1c_scored.json"))["l2re_flat"],
           "dfg": json.load(open(EXP / "results/phase1c_dfg_scored.json"))["l2re_flat"]}
    for pr in PROBS:
        for arm, d in src[pr]["brazos"].items():
            for stage in ("final", "adam"):
                s = d[stage]
                rows.append(dict(problema=pr, brazo=arm, etapa=stage, n=s["n"], mediana_razon=s["mediana"], log10_mediana=math.log10(s["mediana"]),
                                 mejora_k=s["mejora_k"], razon_max=s["razon_max"], razon_min=s["razon_min"], clase=s["clase"], fase=d.get("fase", "")))
    return rows


# ------------------------------------------------------------------ propiedades
def load_props():
    vals = defaultdict(lambda: defaultdict(list))
    for fn, keep in (("fase2/properties.csv", {"ac_d1e-3", "ac_d2e-4", "burgers", "kpp"}), ("fase2/properties_terms.csv", {"ns_terms", "dfg", "piezo_standin"}), ("fase2/properties_edge_scale.csv", {"ac_d1e-3", "ac_d2e-4", "burgers", "kpp", "ns_terms", "dfg", "piezo_standin"})):
        for r in csv.DictReader(open(EXP / fn, newline="")):
            pr = "ns" if r["problema"] == "ns_terms" else r["problema"]
            if r["problema"] in keep and (not fn.endswith("edge_scale.csv") or r["propiedad"].startswith(("edge_trace", "layer_trace"))):
                vals[pr][r["propiedad"]].append(float(r["valor"]))
    med = {pr: {k: statistics.median(v) for k, v in d.items()} for pr, d in vals.items()}
    # propiedades derivadas / estructurales
    for pr in PROBS + ["piezo_standin"]:
        if pr not in med:
            continue
        m = med[pr]
        hard = pr in ("ac_d1e-3", "ac_d2e-4", "burgers", "kpp")                 # IC y BC duras: sin terminos de borde en la perdida
        if hard:
            m.setdefault("bc_over_pde", 0.0); m.setdefault("gn_trace_bc_over_pde", 0.0); m.setdefault("n_loss_terms", 1.0)
            m.setdefault("output_scale_ratio", 1.0)
        m["log10_bc_over_pde"] = math.log10(1 + m["bc_over_pde"])
        m["log10_damkohler"] = math.log10(m["damkohler"]) if "damkohler" in m else float("nan")
        m["log10_peclet"] = math.log10(m["peclet"]) if "peclet" in m else float("nan")
        m["log10_front_over_rbf"] = math.log10(m["front_over_rbf"]) if "front_over_rbf" in m else float("nan")
        m["log10_condB_median"] = math.log10(m["condB_median"])
    return med


PROPS = [  # (clave, descripcion corta)
    ("resid_conc_top10", "concentracion |r| (top 10 %)"),
    ("resid_conc_top10_sq", "concentracion r^2 (top 10 %)"),
    ("resid_rms0", "RMS del residuo inicial"),
    ("condB_median", "cond(B) mediana"),
    ("condB_min", "cond(B) minima"),
    ("log10_bc_over_pde", "log10(1+borde/PDE) en la perdida"),
    ("gn_trace_bc_over_pde", "traza GN borde/PDE"),
    ("n_outputs", "n salidas"),
    ("n_loss_terms", "n terminos de la perdida"),
    ("output_scale_ratio", "razon de escalas de salida"),
    ("log10_peclet", "log10 Peclet (Re)"),
    ("log10_damkohler", "log10 Damkohler"),
    ("log10_front_over_rbf", "log10 ancho de frente / ancho RBF"),
    ("model_width", "ancho del modelo"),
    ("grad_norm0", "norma del gradiente inicial"),
    ("edge_trace_log10_p95_p5", "log10 p95/p5 de traza(B) entre aristas (3 semillas)"),
    ("edge_trace_top10", "traza de B en el 10 % de aristas (3 semillas)"),
    ("layer_trace_log10_ratio", "log10 max/min de traza mediana entre capas (3 semillas)"),
]
PLOT_PROPS = ["resid_conc_top10_sq", "resid_rms0", "condB_median", "condB_min", "log10_bc_over_pde", "n_outputs", "n_loss_terms", "output_scale_ratio", "log10_peclet", "log10_front_over_rbf",
              "model_width", "grad_norm0", "edge_trace_log10_p95_p5", "edge_trace_top10", "layer_trace_log10_ratio"]


def boot_spearman(x, y, B=4000, seed=0):
    ok = [i for i in range(len(x)) if not (math.isnan(x[i]) or math.isnan(y[i]))]
    xs, ys = np.array([x[i] for i in ok]), np.array([y[i] for i in ok])
    n = len(ok)
    if n < 4 or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return n, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    rho = spearmanr(xs, ys).statistic
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if np.ptp(xs[idx]) == 0 or np.ptp(ys[idx]) == 0:
            continue
        bs.append(spearmanr(xs[idx], ys[idx]).statistic)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 100 else (float("nan"), float("nan")))
    loo = [spearmanr(np.delete(xs, i), np.delete(ys, i)).statistic for i in range(n)]
    loo = [v for v in loo if not math.isnan(v)]
    return n, rho, lo, hi, (min(loo) if loo else float("nan")), (max(loo) if loo else float("nan"))


def auc(pos, neg):
    """P(x_pos > x_neg) + 0.5 empates (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    s = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def main():
    eff = load_effects()
    props = load_props()
    # ---- tablas de efectos
    with open(HERE / "tabla_problema_brazo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eff[0])); w.writeheader(); w.writerows(eff)
    E = {(r["problema"], r["brazo"], r["etapa"]): r for r in eff}
    for stage in ("final", "adam"):
        with open(HERE / f"tabla_wide_{stage}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["problema"] + [f"{a}_{c}" for a in ARMS for c in ("mediana", "k_n", "max", "clase")])
            for pr in PROBS:
                row = [pr]
                for a in ARMS:
                    r = E.get((pr, a, stage))
                    row += [f"{r['mediana_razon']:.4g}", f"{r['mejora_k']}/{r['n']}", f"{r['razon_max']:.3g}", r["clase"]] if r else ["", "", "", "sin datos"]
                w.writerow(row)
    with open(HERE / "propiedades_por_problema.csv", "w", newline="") as f:
        keys = sorted({k for p in props.values() for k in p})
        cols = PROBS + (["piezo_standin"] if "piezo_standin" in props else [])
        w = csv.writer(f); w.writerow(["propiedad"] + cols)
        for k in keys:
            w.writerow([k] + [("" if k not in props[pr] else f"{props[pr][k]:.6g}") for pr in cols])

    # ---- Spearman (efecto vs propiedad), por brazo y etapa
    out = []
    for arm in CORE:
        for stage in ("final", "adam"):
            y = [E[(pr, arm, stage)]["log10_mediana"] for pr in PROBS]
            for key, desc in PROPS:
                x = [props[pr].get(key, float("nan")) for pr in PROBS]
                n, rho, lo, hi, lmin, lmax = boot_spearman(x, y)
                out.append(dict(brazo=arm, etapa=stage, propiedad=key, descripcion=desc, n=n, rho=rho, ic95_lo=lo, ic95_hi=hi, loo_min=lmin, loo_max=lmax))
    with open(HERE / "spearman_efecto_propiedad.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    # ---- separacion por clase (sirve vs el resto / dana vs el resto), final
    sep = []
    for arm in CORE:
        cls = {pr: E[(pr, arm, "final")]["clase"] for pr in PROBS}
        for tgt in ("sirve", "daña"):
            P_, N_ = [pr for pr in PROBS if cls[pr] == tgt], [pr for pr in PROBS if cls[pr] != tgt]
            if not P_ or not N_:
                continue
            for key, desc in PROPS:
                xp = [props[pr][key] for pr in P_ if key in props[pr] and not math.isnan(props[pr][key])]
                xn = [props[pr][key] for pr in N_ if key in props[pr] and not math.isnan(props[pr][key])]
                a = auc(xp, xn)
                sep.append(dict(brazo=arm, objetivo=tgt, problemas_objetivo="+".join(P_), problemas_resto="+".join(N_), propiedad=key, n_obj=len(xp), n_resto=len(xn), auc=a,
                                separa_perfecto=bool(not math.isnan(a) and (a in (0.0, 1.0))), direccion=("mayor en objetivo" if a == 1.0 else "menor en objetivo" if a == 0.0 else "")))
    with open(HERE / "separacion_clase.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sep[0])); w.writeheader(); w.writerows(sep)

    # ---- figura 1: heatmap de efecto problema x brazo
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, stage, title in zip(axes, ("final", "adam"), ("Final (tras L-BFGS)", "Final de Adam")):
        M = np.full((len(PROBS), len(HEAT_ARMS)), np.nan)
        for i, pr in enumerate(PROBS):
            for j, a in enumerate(HEAT_ARMS):
                r = E.get((pr, a, stage))
                if r:
                    M[i, j] = r["log10_mediana"]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(HEAT_ARMS))); ax.set_xticklabels(HEAT_ARMS); ax.set_yticks(range(len(PROBS))); ax.set_yticklabels([LABEL[p] for p in PROBS])
        for i, pr in enumerate(PROBS):
            for j, a in enumerate(HEAT_ARMS):
                r = E.get((pr, a, stage))
                if r:
                    ax.text(j, i, f"{r['mediana_razon']:.2f}\n{r['mejora_k']}/{r['n']}", ha="center", va="center", fontsize=7, color="black")
                else:
                    ax.text(j, i, "s/d", ha="center", va="center", fontsize=7, color="gray")
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="log10 mediana e_X / e_B (azul = mejora)")
    fig.suptitle("EXP-0028: efecto por problema y brazo (pareado con B; celda = razon mediana, mejoras k/n; s/d = sin datos)", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"heatmap_efecto.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- figura 2: heatmap de Spearman (solo rho; los IC y los rangos leave-one-out estan en el CSV y en cruce.md)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, stage in zip(axes, ("final", "adam")):
        M = np.full((len(CORE), len(PLOT_PROPS)), np.nan)
        for i, a in enumerate(CORE):
            for j, k in enumerate(PLOT_PROPS):
                M[i, j] = next(o for o in out if o["brazo"] == a and o["etapa"] == stage and o["propiedad"] == k)["rho"]
        im = ax.imshow(M, cmap="PuOr_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(PLOT_PROPS))); ax.set_xticklabels(PLOT_PROPS, rotation=75, ha="right", fontsize=7)
        ax.set_yticks(range(len(CORE))); ax.set_yticklabels(CORE)
        for i, a in enumerate(CORE):
            for j, k in enumerate(PLOT_PROPS):
                r = next(o for o in out if o["brazo"] == a and o["etapa"] == stage and o["propiedad"] == k)
                ax.text(j, i, "-" if math.isnan(r["rho"]) else f"{r['rho']:.2f}" + ("*" if r["n"] < 6 else ""), ha="center", va="center", fontsize=7)
        ax.set_title("Final (L-BFGS)" if stage == "final" else "Final de Adam", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman rho (efecto por problema vs propiedad)")
    fig.suptitle("DESCRIPTIVO: n = 6 problemas (* = n < 6: propiedad no definida en todos); sin p-values; IC bootstrap en spearman_efecto_propiedad.csv", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"heatmap_spearman.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- figura 3: dispersiones
    sel = [("resid_conc_top10_sq", True), ("condB_median", True), ("log10_bc_over_pde", False), ("n_outputs", False), ("log10_peclet", False), ("log10_damkohler", False)]
    colors = {"F": "tab:blue", "J": "tab:red", "E": "tab:green", "I": "tab:purple"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    desc = dict(PROPS)
    for ax, (k, _) in zip(axes.ravel(), sel):
        for a in CORE:
            xs = [props[pr].get(k, float("nan")) for pr in PROBS]
            ys = [E[(pr, a, "final")]["log10_mediana"] for pr in PROBS]
            ax.scatter(xs, ys, color=colors[a], s=26, label=a, alpha=0.85)
        for pr in PROBS:
            xv = props[pr].get(k, float("nan"))
            if not math.isnan(xv):
                ax.annotate(LABEL[pr], (xv, E[(pr, "E", "final")]["log10_mediana"]), fontsize=5.5, xytext=(2, 3), textcoords="offset points")
        pz = props.get("piezo_standin", {}).get(k, float("nan"))
        if not math.isnan(pz):
            ax.axvline(pz, color="k", ls="--", lw=0.8); ax.text(pz, ax.get_ylim()[1], " piezo (stand-in)", fontsize=5.5, va="top", rotation=90)
        ax.axhline(0, color="gray", lw=0.6); ax.set_xlabel(desc[k], fontsize=8); ax.set_ylabel("log10 e_X/e_B (final)", fontsize=8); ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Efecto final por brazo vs propiedad (6 puntos por serie; etiqueta = problema, sobre la serie E)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"dispersion_efecto_propiedad.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
