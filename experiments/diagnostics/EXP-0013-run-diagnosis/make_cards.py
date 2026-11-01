#!/usr/bin/env python3
"""EXP-0013 — fichas de diagnóstico por corrida (cards/<id>.md) y reglas de modo de falla.

Modo de falla = etiquetas heurísticas FIJADAS AQUÍ (no causales). Se asignan solo con salidas ya calculadas:
  null_state          : error relativo L2 >= 0.8 en todas las componentes principales (campo casi nulo/trivial)
  budget_limited      : L-BFGS terminó por tope de tiempo o de iteraciones (no por plateau/convergencia)
  concentrated_error  : masa del error en el 5 % de puntos >= 0.5, o concentración del error en la región steep5 >= 5
  residual_misaligned : Spearman(residuo,error) < 0.2 y solape del 5 % superior < 0.1
  residual_aligned    : Spearman(residuo,error) >= 0.5
  dead_bases          : fracción de bases muertas (capa 0) >= 0.2
  low_jacobian_rank_per_param : rango efectivo del jacobiano por parámetro < 0.002 (umbral no calibrado; casi todos los modelos grandes lo activan)
  plateau_adam        : ganancia relativa de la pérdida en el último 20 % de Adam < 0.01
Uso: py -3.13 make_cards.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import HERE, RESULTS, log

CARDS = HERE / "cards"
CARDS.mkdir(exist_ok=True)


def f(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:.{nd}g}"


def modes(r: pd.Series) -> list:
    tags = []
    comps = [c for c in ("err/u_rel_l2", "err/v_rel_l2") if c in r.index and pd.notna(r[c])]
    if comps and all(float(r[c]) >= 0.8 for c in comps):
        tags.append("null_state")
    st = str(r.get("train/meta_lbfgs_status", r.get("train/meta_lbfgs_stop_reason", r.get("train/meta_stop_reason", ""))))
    if st in ("lbfgs_time_cap", "lbfgs_iter_cap", "time_cap", "max_iterations"):
        tags.append("budget_limited")
    if (pd.notna(r.get("err/top5_mass")) and r["err/top5_mass"] >= 0.5) or (pd.notna(r.get("region/steep5/err_concentration")) and r["region/steep5/err_concentration"] >= 5):
        tags.append("concentrated_error")
    sp, ov = r.get("audit/spearman_res_err"), r.get("audit/top5_overlap")
    if pd.notna(sp) and pd.notna(ov) and sp < 0.2 and ov < 0.1:
        tags.append("residual_misaligned")
    if pd.notna(sp) and sp >= 0.5:
        tags.append("residual_aligned")
    if pd.notna(r.get("basis/l0/dead_fraction")) and r["basis/l0/dead_fraction"] >= 0.2:
        tags.append("dead_bases")
    if pd.notna(r.get("jac/effective_rank_per_param")) and r["jac/effective_rank_per_param"] < 0.002:
        tags.append("low_jacobian_rank_per_param")
    if pd.notna(r.get("train/adam_rel_gain_last20pct")) and r["train/adam_rel_gain_last20pct"] < 0.01 and r["stage"] == "adam_final":
        tags.append("plateau_adam")
    return tags


def main():
    s = pd.read_csv(RESULTS / "signals_wide.csv")
    l = pd.read_csv(RESULTS / "labels_wide.csv")
    d = s.merge(l.drop(columns=["problem", "campaign", "arm", "width", "seed", "stage"]), on="id")
    d["modes"] = d.apply(modes, axis=1)
    d[["id", "problem", "campaign", "arm", "width", "seed", "stage", "modes"]].assign(modes=d["modes"].apply(lambda m: ";".join(m))).to_csv(RESULTS / "failure_modes.csv", index=False)
    nulls = pd.read_csv(RESULTS / "nulls.csv")
    for _, r in d.iterrows():
        nl = nulls[nulls["id"] == r["id"]]
        regs = sorted({c.split("/")[1] for c in d.columns if c.startswith("region/") and pd.notna(r.get(c))})
        L = [f"# Ficha de diagnóstico — {r['id']}", "",
             f"**Problema:** {r['problem']} · **Campaña:** {r['campaign']} · **Brazo:** {r['arm']}{' (ancho ' + str(int(r['width'])) + ')' if (pd.notna(r['width']) and r['width']) else ''} · **Semilla:** {int(r['seed'])} · **Etapa:** {r['stage']}",
             f"**Checkpoint:** `{r['path']}` · **Familia de modelo:** {r['family']} ({r['kind']}) · **Parámetros:** {int(r['params'])} · **Precisión nativa:** {r['dtype_native']} (señales en float64)", "",
             "> Exploratorio. `scientific_accepted=false`. Las señales no usan la referencia; las etiquetas sí y solo describen el resultado.", "",
             "## 1. Qué hizo el modelo (resultado, etiquetas)", "",
             f"- Error relativo L2 por componente: " + ", ".join(f"{c.split('/')[1].replace('_rel_l2', '')} = {f(r[c])}" for c in d.columns if c.startswith("err/") and c.endswith("_rel_l2") and pd.notna(r[c])),
             f"- Error normalizado (RMS, todas las componentes): {f(r['err/all_rms_normalized'])}; plano: {f(r['err/flat_rel_l2'])} (la medida plana mezcla unidades en piezo)",
             f"- Concentración del error: Gini {f(r['err/gini'])}; masa en el 5 % superior {f(r['err/top5_mass'])}; entropía normalizada {f(r['err/entropy_norm'])}", "",
             "## 2. Dónde falla (regiones, definidas con la referencia)", "", "| Región | fracción de puntos | masa de error | concentración de error | masa de residuo | concentración de residuo |", "|---|---|---|---|---|---|"]
        for g in regs:
            L.append(f"| {g} | {f(r[f'region/{g}/point_fraction'])} | {f(r[f'region/{g}/err_mass'])} | {f(r[f'region/{g}/err_concentration'])} | {f(r[f'region/{g}/res_mass'])} | {f(r[f'region/{g}/res_concentration'])} |")
        L += ["", "## 3. Qué ve el residuo (señales sin referencia)", "",
              f"- Residuo RMS {f(r['res/all/rms'])}; p95 {f(r['res/all/p95'])}; máx {f(r['res/all/max'])}",
              f"- Concentración: Gini {f(r['res/all/gini'])}; masa en el 1 % / 5 % superior {f(r['res/all/top1_mass'])} / {f(r['res/all/top5_mass'])}; entropía normalizada {f(r['res/all/entropy_norm'])}",
              f"- Alineación del residuo con la estructura del modelo: Spearman(|r|,|∇u|) {f(r['align/spearman_res_gradnorm'])}; con |Δu| {f(r['align/spearman_res_laplacian'])}", "",
              "## 4. Alineación residuo–error (auditoría con etiquetas)", "",
              f"- Spearman(|r|, error) {f(r['audit/spearman_res_err'])}; solape del 5 % superior {f(r['audit/top5_overlap'])}; masa de error en el 5 % superior del residuo {f(r['audit/err_mass_in_top5res'])}; masa de residuo en el 5 % superior del error {f(r['audit/res_mass_in_top5err'])}",
              "- Lectura: si el residuo no se alinea con el error, un muestreo guiado por residuo no puede dirigirse a donde está el error en este punto del entrenamiento.", "",
              "## 5. Capacidad y optimización", ""]
        L += [f"- Rango efectivo del jacobiano {f(r['jac/effective_rank'])} (por parámetro {f(r['jac/effective_rank_per_param'])}); número de condición {f(r['jac/cond'])}"]
        if pd.notna(r.get("basis/l0/eff_bases")):
            L += [f"- Bases (capa 0): efectivas {f(r['basis/l0/eff_bases'])}; muertas {f(r['basis/l0/dead_fraction'])}; aristas pequeñas {f(r['basis/l0/small_edge_fraction'])}"]
        ratios = [c for c in d.columns if c.startswith("scale/ratio_struct_to_spacing") and pd.notna(r[c])]
        for c in ratios:
            L += [f"- {c.replace('scale/', '')} = {f(r[c])} (< 1 significa que la estructura del modelo es más fina que el espaciado de las bases de la capa 0)"]
        tr = [c for c in d.columns if c.startswith("train/") and not c.startswith("train/meta") and pd.notna(r[c])]
        if tr:
            L += ["", "## 6. Señales de entrenamiento registradas", ""] + [f"- {c.replace('train/', '')}: {f(r[c])}" for c in tr]
        meta = [c for c in d.columns if c.startswith("train/meta") and pd.notna(r[c])]
        if meta:
            L += [f"- {c.replace('train/', '')}: {r[c]}" for c in meta]
        L += ["", "## 7. Modo(s) de falla asignado(s) (reglas fijas, no causales)", "", "- " + (", ".join(r["modes"]) if r["modes"] else "ninguno de los marcadores se activa"), "",
              "## 8. Límites y datos ausentes", ""]
        L += [f"- `{x.signal_group}`: {x.reason}" for x in nl.itertuples()] or ["- (ninguno)"]
        (CARDS / f"{r['id']}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("cards", n=len(d))
    print(len(d), "fichas")


if __name__ == "__main__":
    main()
