#!/usr/bin/env python3
"""EXP-0013 — criterio (d): costo de cada grupo de señales frente a un paso de entrenamiento (CPU, un checkpoint representativo por problema y tipo).

Paso de referencia = una evaluación de pérdida de residuo con backward sobre 4 096 puntos (tamaño del lote completo de Heat; Burgers/AC usaron 8 192).
Sobrecosto relativo con cadencia K = tiempo_señal / (K · tiempo_paso). El criterio (d) se evalúa con K = 1 000 (cadencia de refresco de EXP-0011) y se reporta también K = 100.
Uso: py -3.13 cost_bench.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import torch

from adapters import load_bundle, _to_t
from common import RESULTS, log
from grids import get_grid
import signals as S


def med(fn, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    rows = [r for r in csv.DictReader(open(RESULTS / "inventory.csv", encoding="utf-8")) if r["load"] in ("ok", "not_tested")]
    seen, pick = set(), []
    for r in rows:
        if r["stage"] != "adam_final":
            continue
        b0 = (r["problem"], "mlp" if r["arm"].startswith("mlp") else "rbf")
        if b0 not in seen:
            seen.add(b0)
            pick.append(r)
    out = []
    for r in pick:
        g = get_grid(r["problem"])
        pts = g["points"]
        b = load_bundle(r)
        rng = np.random.default_rng(0)
        batch = pts[rng.choice(len(pts), 4096, replace=len(pts) < 4096)]

        def step():
            comps = b.res_tensors(batch)
            loss = sum((v ** 2).mean() for v in comps.values())
            for p in b.model.parameters():
                p.grad = None
            loss.backward()

        t_step = med(step)
        timers = {
            "residual_stats(grid)": lambda: S.residual_stats("x", np.sqrt(sum(v ** 2 for v in b.residuals(pts).values()))),
            "structure_fields(grid)": lambda: S.structure_fields(b, pts),
            "basis_signals": lambda: S.basis_signals(b, pts, {}),
            "jacobian_256": lambda: S.jacobian_signals(b, pts),
            "gradterm_1024": lambda: S.grad_term_signals(b, pts),
            "param_signals": lambda: S.param_signals(b),
        }
        for name, fn in timers.items():
            try:
                t = med(fn, 2)
            except Exception as e:  # noqa: BLE001
                out.append({"problem": r["problem"], "kind": b.kind, "group": name, "seconds": None, "step_seconds": t_step, "error": f"{type(e).__name__}: {e}"[:120]})
                continue
            out.append({"problem": r["problem"], "kind": b.kind, "group": name, "grid_points": len(pts), "seconds": t, "step_seconds": t_step, "ratio_to_step": t / t_step,
                        "overhead_K100": t / (100 * t_step), "overhead_K1000": t / (1000 * t_step), "meets_d_K1000": bool(t / (1000 * t_step) < 0.10), "error": ""})
        log("cost_bench", problem=r["problem"], kind=b.kind, step_seconds=round(t_step, 4))
    keys = sorted({k for o in out for k in o})
    with open(RESULTS / "analysis" / "cost_bench.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o["problem"], o["kind"], o["group"], None if o.get("seconds") is None else round(o["seconds"], 3), "step", round(o["step_seconds"], 4), "K1000", o.get("overhead_K1000"))


if __name__ == "__main__":
    (RESULTS / "analysis").mkdir(exist_ok=True)
    main()
