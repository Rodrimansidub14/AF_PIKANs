#!/usr/bin/env python3
"""Regenerable figures for the baseline-v2 campaigns (reads only experiments/*/results, writes experiments/*/figures).

Per-seed points (seeds are the unit of dispersion) + mean; log y-axis for errors; the two optimisation stages are never mixed.
Run: py -3.13 experiments/make_figures.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
E8 = "EXP-0008-baseline-v2-competitive"
COL = {"mlp": "#d95f02", "rbf": "#1b9e77", "rbf@16": "#1b9e77", "rbf@48": "#7570b3", "rbf@56": "#7570b3"}


def rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def strip(ax, groups, ylabel, title, hlines=()):
    """groups: list of (label, color, [values])"""
    for i, (lab, col, vals) in enumerate(groups):
        ax.scatter([i] * len(vals), vals, color=col, alpha=0.75, s=28, zorder=3)
        ax.hlines(st.mean(vals), i - 0.3, i + 0.3, color="black", lw=2, zorder=4)
    for y, lab in hlines:
        ax.axhline(y, ls="--", color="grey", lw=1)
        ax.text(len(groups) - 0.5, y, lab, ha="right", va="bottom", fontsize=8, color="grey")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, which="both")


def save(fig, folder, name):
    out = ROOT / folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=140)
    plt.close(fig)
    print("wrote", out / name)


def fig_ac():
    r = rows(ROOT / f"allen-cahn/{E8}/results/ac-rerun-20260930/ac-rerun-per-seed.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, key, ttl in ((axs[0], "l2_adam", "fin de Adam"), (axs[1], "l2_lbfgs", "tras L-BFGS")):
        g = [(a, COL.get(a, "grey"), [float(x[key]) for x in r if x["arm"] == a]) for a in ("mlp", "rbf")]
        strip(ax, g, "L2 relativo vs Allen_Cahn.mat", f"Allen–Cahn · {ttl} (n=5)")
    save(fig, f"allen-cahn/{E8}", "ac-l2-by-stage.png")


def fig_ns():
    r = rows(ROOT / f"navier-stokes/{E8}/results/ns2d-cg-20260930/ns2dcg-per-seed.csv")
    arms = sorted({x["arm"] + ("@" + x.get("width", "") if x.get("width") else "") for x in r})
    fig, ax = plt.subplots(figsize=(6, 4))
    g = []
    for a in sorted({x["arm"] for x in r}):
        for params in sorted({x["params"] for x in r if x["arm"] == a}, key=int):
            g.append((f"{a} ({int(params):,} par.)", COL.get(a, "grey"), [float(x["flat"]) for x in r if x["arm"] == a and x["params"] == params]))
    strip(ax, g, "L2RE plano vs COMSOL", "NS2d-CG · fin de Adam", hlines=((0.119, "PINNacle vanilla 0.119"), (0.0824, "LAAF 0.0824")))
    save(fig, f"navier-stokes/{E8}", "ns2dcg-flat-l2re.png")


def fig_dfg_long():
    r = rows(ROOT / f"navier-stokes/{E8}/results/dfg-long-20261001/dfg-long-per-seed.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ttl in ((axs[0], "u", "error de campo u (L2 rel. vs FEM L3)"), (axs[1], "Cd_err", "error relativo de Cd")):
        g = []
        for a in ("mlp", "rbf"):
            for stage, lab in (("adam_final", "Adam"), ("lbfgs_final", "+L-BFGS")):
                g.append((f"{a} {lab}", COL[a], [float(x[key]) for x in r if x["arm"] == a and x["stage"] == stage]))
        strip(ax, g, key, f"DFG receta larga · {ttl}")
    save(fig, f"navier-stokes/{E8}", "dfg-long-by-stage.png")


def fig_heat():
    d = json.load(open(ROOT / f"heat/{E8}/results/heat-lbfgs-redo-20260930/results-summary.json", encoding="utf-8"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    g = [("fin de Adam", COL["rbf"], [x["adam_final_rel_l2"] for x in d]), ("+ L-BFGS completo", COL["rbf"], [x["rel_l2_512x256"] for x in d])]
    strip(axs[0], g, "L2 relativo 512×256", "Heat · control RBF-PIKAN (n=5)")
    for x in d:
        c = x["curve"]
        axs[1].plot([p["iters"] + 1 for p in c], [p["rel_l2"] for p in c], marker="o", ms=3, label=str(x["seed"]))
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("iteraciones L-BFGS")
    axs[1].set_ylabel("L2 relativo")
    axs[1].set_title("Heat · L-BFGS alcanza su piso en ~50 iteraciones", fontsize=10)
    axs[1].legend(title="semilla", fontsize=7)
    axs[1].grid(alpha=0.25, which="both")
    save(fig, f"heat/{E8}", "heat-lbfgs-stages-and-curves.png")


def fig_burgers():
    p = ROOT / f"burgers/{E8}/results/burgers-close-20261003"
    runs = sorted(p.glob("*/run.json")) + sorted(p.glob("*/*/run.json"))
    if not runs:
        return
    data = {}
    for f in runs:
        r = json.loads(f.read_text(encoding="utf-8"))
        key = r["arm"] + (f"@{r['width']}" if r["arm"] == "rbf" else "")
        data.setdefault(key, []).append((r["score_adam_final"]["l2re_coleHopf"], r.get("score_lbfgs_final", {}).get("l2re_coleHopf")))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, idx, ttl in ((axs[0], 0, "fin de Adam"), (axs[1], 1, "tras L-BFGS")):
        g = [(k, COL.get(k, "grey"), [v[idx] for v in vs if v[idx] is not None]) for k, vs in sorted(data.items())]
        strip(ax, g, "L2 relativo vs Cole–Hopf", f"Burgers · {ttl}", hlines=((1.45e-2, "PINNacle vanilla 1.45e-2"), (1.33e-2, "Adam+L-BFGS pub. 1.33e-2")))
    save(fig, f"burgers/{E8}", "burgers-l2-by-stage.png")


if __name__ == "__main__":
    for fn in (fig_ac, fig_ns, fig_dfg_long, fig_heat, fig_burgers):
        try:
            fn()
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)
