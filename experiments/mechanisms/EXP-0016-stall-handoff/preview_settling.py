"""Tier 1 (CPU, sin referencia): vista previa del mapa interno de no asentamiento sobre checkpoints de Adam YA archivados.

Solo hay trayectorias parciales archivadas: Heat S2 (RBF16, f64) guarda la capa 0 en 4 instantes de Adam por semilla (pasos ~4 300, ~10 800,
~17 900 y 20 000; checkpoint-120s/300s/500s/adam-final). Burgers y Allen-Cahn solo guardan el final, asi que no hay nada que ver ahi: sus trayectorias
completas (cada 500 pasos) existiran solo tras la etapa A.

Esto NO es la senal preregistrada: los instantes no estan separados por 500 pasos y no hay una ventana de referencia 2 000-4 000. Se usa la tasa de
cambio relativa por 500 pasos de cada base entre instantes consecutivos, normalizada por la mediana ENTRE BASES del mismo intervalo. Sirve solo
para comprobar dos cosas antes de gastar GPU: (1) que el mapa m(x) no es constante, (2) que su forma se repite entre semillas (no es ruido).
No lee ningun error contra la referencia ni puntua nada.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from src.mechanisms.settling_focus import unsettled_map  # noqa: E402

RUNS = ROOT / "var/agent_runs/baseline-competitive-2026-09-27/kaggle-outputs/closure-s2-20260928/heat-stepmatched-s2/runs/tier2"
NAMES = ["checkpoint-120s.pt", "checkpoint-300s.pt", "checkpoint-500s.pt", "checkpoint-adam-final.pt"]
SEEDS = [901, 902, 903, 904, 905]
LO, HI = np.array([0.0, 0.0]), np.array([1.0, 1.0])  # dominio de Heat (x, t)


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


grid = np.stack(np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41), indexing="ij"), -1).reshape(-1, 2)
out, maps = {}, {}
for seed in SEEDS:
    snaps = []
    for n in NAMES:
        f = RUNS / f"seed-{seed}/rbf/{n}"
        if not f.exists():
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        snaps.append((int(ck["adam_step"]), sd["layers.0.coefficients"].double().numpy(), sd["layers.0.centers"].double().numpy(),
                      float(sd["layers.0.width"])))
    if len(snaps) < 3:
        print(seed, "sin suficientes instantes", len(snaps))
        continue
    centers, width = snaps[0][2], snaps[0][3]
    per_interval = []
    for (s0, c0, _, _), (s1, c1, _, _) in zip(snaps[:-1], snaps[1:]):
        rate = np.linalg.norm(c1 - c0, axis=0) / np.maximum(np.linalg.norm(c1, axis=0), 1e-300) * 500.0 / (s1 - s0)  # (D, G)
        med = np.median(rate)
        per_interval.append(dict(steps=(s0, s1), rho=rate / med if med > 0 else np.zeros_like(rate)))
    last = per_interval[-1]["rho"]
    m = unsettled_map(grid, last, centers, width)
    maps[seed] = m
    out[seed] = dict(
        steps=[s[0] for s in snaps],
        rho_last_min=float(last.min()), rho_last_max=float(last.max()), rho_last_cv=float(last.std() / last.mean()),
        most_unsettled_bases_last=[(int(d), int(g)) for d, g in zip(*np.unravel_index(np.argsort(last, axis=None)[::-1][:3], last.shape))],
        map_mean=float(m.mean()), map_ptp_over_mean=float(np.ptp(m) / m.mean()), map_cv=float(m.std() / m.mean()),
        median_rate_trend=[float(np.median(iv["rho"])) for iv in per_interval])
    print(seed, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[seed].items()}), flush=True)

seeds = sorted(maps)
pairs = [spearman(maps[a], maps[b]) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
out["map_spearman_between_seeds"] = dict(median=float(np.median(pairs)), min=float(np.min(pairs)), max=float(np.max(pairs)), n_pairs=len(pairs))
print("Spearman del mapa entre semillas:", out["map_spearman_between_seeds"])
(Path(__file__).parent / "preview_settling.json").write_text(json.dumps(out, indent=1, default=str))
