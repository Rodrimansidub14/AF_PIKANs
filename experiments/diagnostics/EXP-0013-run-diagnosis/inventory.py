#!/usr/bin/env python3
"""EXP-0013, paso 0: inventario de checkpoints del universo de análisis (solo lectura; CPU).

Recorre las ubicaciones de los checkpoints que respaldan los baselines v1/v2, sus brazos pareados, EXP-0011/0012 y piezo; calcula tamaño y sha256,
clasifica (problema, campaña, brazo, semilla, etapa) y prueba si el checkpoint se carga con la clase de modelo correspondiente (state_dict estricto).
No usa referencias ni etiquetas. Salidas: results/inventory.csv y results/inventory-summary.md.
Run: py -3.13 experiments/diagnostics/EXP-0013-run-diagnosis/inventory.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BL = REPO / "var/agent_runs/baseline-competitive-2026-09-27"
EX = REPO / "experiments"

# (problema, campaña, raíz, patrón de archivo)
SOURCES = [
    ("heat_1d", "S2-control-adam", BL / "kaggle-outputs/closure-s2-20260928", "heat-stepmatched-s2/runs/tier2/seed-*/rbf/checkpoint-adam-final.pt"),
    ("heat_1d", "heat-lbfgs-redo", EX / "heat/EXP-0008-baseline-v2-competitive/checkpoints", "heat-lbfgs-redo-20260930/results/seed-*/checkpoint-lbfgs-redo.pt"),
    ("heat_1d", "EXP-0011", EX / "heat/EXP-0011-heat-rad-adam/results/run-20261004/results", "*/seed-*/checkpoint-*.pt"),
    ("heat_1d", "EXP-0012", EX / "heat/EXP-0012-heat-rad-uniform-lbfgs/results/run-20261004/results", "seed-*/checkpoint-lbfgs-uniform.pt"),
    ("heat_1d", "heat-v4-isocost", BL / "kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs", "tier2/seed-*/rbf/checkpoint-final.pt"),
    ("heat_1d", "heat-v4-isocost", BL / "kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/runs", "tier2/seed-*/fnn/checkpoint-final.pt"),
    ("allen_cahn_1d", "ac-rerun-20260930", EX / "allen-cahn/EXP-0008-baseline-v2-competitive/checkpoints", "ac-rerun-20260930/ac/*/checkpoint-*.pt"),
    ("burgers_1d", "burgers-close-20261003", EX / "burgers/EXP-0008-baseline-v2-competitive/checkpoints", "burgers-close-20261003/*/checkpoint-*.pt"),
    ("burgers_1d", "burgers-f64-lbfgs-20261003", EX / "burgers/EXP-0008-baseline-v2-competitive/checkpoints", "burgers-f64-lbfgs-20261003/*/checkpoint-*.pt"),
    ("burgers_1d", "burgers-big-20261003", EX / "burgers/EXP-0008-baseline-v2-competitive/checkpoints", "burgers-big-20261003/*/checkpoint-*.pt"),
    ("ns2d_cg", "ns2d-cg-20260930", EX / "navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints", "ns2d-cg-20260930/ns/*/checkpoint-*.pt"),
    ("dfg_2d", "dfg-daniel-cfg-20260930", EX / "navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints", "dfg-daniel-cfg-20260930/*/checkpoint-*.pt"),
    ("dfg_2d", "dfg-long-20261001", EX / "navier-stokes/EXP-0008-baseline-v2-competitive/checkpoints", "dfg-long-20261001/*/checkpoint-*.pt"),
    ("dfg_2d", "cohort-A-r3r4-adam", REPO / "var/agent_runs/baseline-close-2026-09-27", "dfg-re20-gpu-r*/download-attempts/*/pikan-*/runs/seed-*/dfg_rbf_pikan.pt"),
    ("dfg_2d", "cohort-A-lbfgs", BL / "kaggle-outputs/dfg-rbf-current-cohort-refinement", "*/run-*/training/seed-*/checkpoint-lbfgs.pt"),
    ("dfg_2d", "cohort-B-pilot-adam", BL / "kaggle-cli-fetch-competitive-pilots/dfg", "dfg-re20-t4x2/campaign/*/seed-*/checkpoint-final.pt"),
    ("dfg_2d", "cohort-B-lbfgs", BL / "kaggle-outputs/dfg-lbfgs-refinement", "*/run-*/training/*/seed-*/checkpoint-lbfgs.pt"),
    ("piezo_2d", "piezo-daniel-recipe", EX / "piezo/EXP-0009-piezo-indirect-rbf/results", "daniel-*/results/*/checkpoint-*-seed-*.pt"),
]

_mods: dict[str, object] = {}


def load_module(name: str, path: Path, extra_path: Path | None = None):
    if name in _mods:
        return _mods[name]
    if extra_path:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _mods[name] = mod
    return mod


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def classify(problem: str, campaign: str, rel: str):
    seed = re.search(r"seed-(\d+)", rel)
    seed = int(seed.group(1)) if seed else None
    name = Path(rel).name
    low = rel.lower()
    if "mlp" in low or "tanh-mlp" in low or "tanh_mlp" in low or "/fnn/" in low.replace("\\", "/"):
        arm = "mlp"
    elif "rbf" in low:
        arm = "rbf"
    else:
        arm = "rbf" if problem == "heat_1d" or campaign.startswith("cohort-A") else "?"
    m = re.search(r"rbf(\d+)", low)
    width = int(m.group(1)) if m else None
    if campaign == "EXP-0011":
        arm = "rbf+rad" if rel.replace("\\", "/").startswith("rad/") else "rbf(ctl-replicate)"
    if campaign == "EXP-0012":
        arm = "rbf+rad(adam)->uniform-lbfgs"
    if "adam-final" in name or name in ("dfg_rbf_pikan.pt",) or "pilot-adam" in campaign or campaign in ("S2-control-adam", "heat-v4-isocost"):
        stage = "adam_final"
    elif "lbfgs" in name or "lbfgs" in campaign:
        stage = "lbfgs_final"
    elif name.startswith("checkpoint-") and campaign.startswith("piezo"):
        stage = "adam_final"
    else:
        stage = "?"
    return arm, width, seed, stage


def state_of(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if isinstance(sd, dict) and sd and all(k.startswith("net.") for k in sd):
        sd = {k[4:]: v for k, v in sd.items()}
    return sd, obj


def try_load(problem: str, campaign: str, path: Path):
    """Devuelve (estado, nota). Carga estricta en la clase correspondiente con formas inferidas del state_dict."""
    try:
        sd, obj = state_of(path)
        if problem == "piezo_2d":
            return "not_tested", "requiere el codigo upstream de Daniel (daniel_arm.py); se prueba en el paso de señales"
        if problem == "heat_1d" and campaign == "heat-v4-isocost" and "/fnn/" in str(path).replace("\\", "/"):
            pay = BL / "kaggle-outputs/heat-tier2-t4x2-v4-20260928/heat-tier2/payload"
            mod = load_module("heat_comparison_runner", pay / "heat_comparison_runner.py", pay)
            m = mod.TutorialFNN()
            m.load_state_dict(sd, strict=True)
            return "ok", "heat_comparison_runner.TutorialFNN [2,20,20,20,1] tanh, hard trace"
        if problem == "heat_1d":
            mod = load_module("af_sciml_model", BL / "kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/payload/af_sciml_model.py")
            m = mod.build_model("heat", {"id": "rbf_fixed"})
            m.load_state_dict(sd, strict=True)
            return "ok", "af_sciml_model.build_model('heat', rbf_fixed)"
        if campaign.startswith("cohort-A"):
            mod = load_module("repo_dfg", REPO / "src/sciml/dfg.py", REPO)
            m = mod.DFGRBFPIKAN()
            m.load_state_dict(sd, strict=True)
            return "ok", "src.sciml.dfg.DFGRBFPIKAN"
        if campaign.startswith("cohort-"):
            src = BL / "kaggle-outputs/dfg-lbfgs-refinement/dfg-re20-lbfgs-refinement/run-1790581871213606372/run_campaign_source.py"
            mod = load_module("dfg_runner_source", src)
            arm = "DFG-Tanh-MLP" if "layers.0.weight" in sd else "DFG-RBF-PIKAN"
            m = mod.build_model(arm, 0)
            m.load_state_dict(sd, strict=True)
            return "ok", f"run_campaign_source.build_model({arm})"
        pc = load_module("pikan_campaign", BL / "burgers-close-20261003/pikan_campaign.py")
        if "layers.0.coefficients" in sd:
            w, n_in, bases = sd["layers.0.coefficients"].shape
            n_out = sd["layers.2.coefficients"].shape[0]
            lo = {"ns2d_cg": [0.0, 0.0], "dfg_2d": [0.0, 0.0], "allen_cahn_1d": [-1.0, 0.0], "burgers_1d": [-1.0, 0.0]}[problem]
            hi = {"ns2d_cg": [4.0, 2.0], "dfg_2d": [2.2, 0.41], "allen_cahn_1d": [1.0, 1.0], "burgers_1d": [1.0, 1.0]}[problem]
            m = pc.RBFKAN(n_in, w, n_out, bases, lo, hi)
            cls = f"RBFKAN(width={w}, bases={bases})"
        else:
            ws = [sd[f"layers.{i}.weight"].shape for i in range(100) if f"layers.{i}.weight" in sd]
            m = pc.MLP([ws[0][1]] + [s[0] for s in ws])
            cls = f"MLP{[ws[0][1]] + [s[0] for s in ws]}"
        m.load_state_dict(sd, strict=True)
        return "ok", f"pikan_campaign.{cls}"
    except Exception as e:  # noqa: BLE001
        return "error", f"{type(e).__name__}: {str(e)[:140]}"


def main():
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for problem, campaign, root, pat in SOURCES:
        files = sorted(root.glob(pat)) if root.exists() else []
        if not files:
            rows.append(dict(problem=problem, campaign=campaign, arm="", width="", seed="", stage="", path="(sin archivos)", bytes=0, sha256="", load="missing", note=f"raiz o patron sin coincidencias: {pat}"))
        seen_dup = set()
        for f in files:
            rel = str(f.relative_to(root))
            dup_key = (re.sub(r"attempt-\d+", "attempt-N", rel))
            if dup_key in seen_dup:
                continue  # intentos de descarga repetidos del mismo checkpoint
            seen_dup.add(dup_key)
            arm, width, seed, stage = classify(problem, campaign, rel)
            st, note = try_load(problem, campaign, f)
            rows.append(dict(problem=problem, campaign=campaign, arm=arm, width=width or "", seed=seed if seed is not None else "", stage=stage,
                             path=str(f.relative_to(REPO)).replace("\\", "/"), bytes=f.stat().st_size, sha256=sha(f), load=st, note=note))
    with open(out / "inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cells = defaultdict(lambda: {"n": 0, "ok": 0, "seeds": set()})
    for r in rows:
        k = (r["problem"], r["campaign"], r["arm"], r["stage"])
        cells[k]["n"] += 1
        cells[k]["ok"] += r["load"] == "ok"
        if r["seed"] != "":
            cells[k]["seeds"].add(r["seed"])
    lines = ["# EXP-0013 — inventario (paso 0)", "", f"Archivos listados: {sum(1 for r in rows if r['load'] != 'missing')}; cargables: {sum(r['load'] == 'ok' for r in rows)}; con error: {sum(r['load'] == 'error' for r in rows)}; no probados: {sum(r['load'] == 'not_tested' for r in rows)}; faltantes: {sum(r['load'] == 'missing' for r in rows)}.", "",
             "| Problema | Campaña | Brazo | Etapa | Checkpoints | Cargables | Semillas |", "|---|---|---|---|---|---|---|"]
    for (p, c, a, s), v in sorted(cells.items()):
        lines.append(f"| {p} | {c} | {a} | {s} | {v['n']} | {v['ok']} | {','.join(map(str, sorted(v['seeds'])))} |")
    errs = [r for r in rows if r["load"] == "error"]
    if errs:
        lines += ["", "## Errores de carga", ""] + [f"- `{r['path']}`: {r['note']}" for r in errs[:20]]
    (out / "inventory-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    print(len(cells), "celdas")


if __name__ == "__main__":
    main()
