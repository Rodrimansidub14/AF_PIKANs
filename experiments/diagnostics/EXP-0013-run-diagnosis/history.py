"""EXP-0013 — señales de entrenamiento (registradas, no recalculadas): pérdida, pendiente, plateau, ganancia de L-BFGS, costo.

Se leen de los archivos de historia que acompañan a cada checkpoint (history.json / history.csv / run.json / result.json / results-summary.json).
Cuando no existe la historia se devuelve `nulls` con la razón.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from common import REPO


def _slope_log(loss, k=10):
    loss = np.asarray([x for x in loss if x and x > 0 and math.isfinite(x)])
    if len(loss) < k + 1:
        return float("nan")
    y = np.log(loss[-k:])
    return float(np.polyfit(np.arange(k), y, 1)[0])


def _from_list(h, prefix):
    """h: lista de dicts con 'loss' (y opcional 'stage')."""
    out = {}
    adam = [e["loss"] for e in h if e.get("stage", "adam") == "adam" and "loss" in e]
    lb = [e["loss"] for e in h if e.get("stage") == "lbfgs" and "loss" in e]
    if not adam and not lb and h:
        adam = [e["loss"] for e in h if "loss" in e]
    if adam:
        out[f"{prefix}adam_loss_end"] = float(adam[-1])
        out[f"{prefix}adam_loss_start"] = float(adam[0])
        out[f"{prefix}adam_log_slope_last10"] = _slope_log(adam)
        n = len(adam)
        a, b = adam[max(0, n - n // 5 - 1)], adam[-1]
        out[f"{prefix}adam_rel_gain_last20pct"] = float((a - b) / a) if a > 0 else float("nan")
        out[f"{prefix}adam_n_logged"] = n
    if lb:
        out[f"{prefix}lbfgs_loss_end"] = float(lb[-1])
        out[f"{prefix}lbfgs_loss_start"] = float(lb[0])
        out[f"{prefix}lbfgs_log_slope_last10"] = _slope_log(lb)
        out[f"{prefix}lbfgs_n_logged"] = len(lb)
        if adam and adam[-1] > 0:
            out[f"{prefix}lbfgs_to_adam_loss_ratio"] = float(lb[-1] / adam[-1])
    return out


def history_signals(row: dict) -> tuple[dict, dict]:
    path = REPO / row["path"]
    d = path.parent
    if not (d / "history.json").exists() and not (d / "history.csv").exists() and not (d / "run.json").exists() and not (d / "result.json").exists():
        parts = list(d.parts)
        alt = d
        if "checkpoints" in parts:
            i = len(parts) - 1 - parts[::-1].index("checkpoints")
            parts[i] = "results"
            alt = Path(*parts)
        if alt.exists():
            d = alt
    sig, nulls = {}, {}
    found = False
    for name in ("history.json",):
        f = d / name
        if f.exists():
            try:
                h = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(h, list):
                    sig.update(_from_list(h, "train/"))
                    found = True
            except Exception as e:  # noqa: BLE001
                nulls["train/history"] = f"{type(e).__name__}: {e}"[:160]
    f = d / "history.csv"
    if f.exists() and not found:
        try:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
            h = [{"stage": r.get("stage", "adam"), "loss": float(r["loss"])} for r in rows if r.get("loss") not in (None, "", "None")]
            sig.update(_from_list(h, "train/"))
            found = True
        except Exception as e:  # noqa: BLE001
            nulls["train/history"] = f"{type(e).__name__}: {e}"[:160]
    for name in ("run.json", "result.json"):
        f = d / name
        if f.exists():
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for key, label in (("adam_seconds", "train/adam_seconds"), ("lbfgs_seconds", "train/lbfgs_seconds"), ("adam_steps", "train/adam_steps"),
                               ("lbfgs_iterations", "train/lbfgs_iterations"), ("lbfgs_iters", "train/lbfgs_iterations"), ("iters", "train/lbfgs_iterations"),
                               ("seconds", "train/seconds"), ("refresh_seconds", "train/refresh_seconds")):
                if key in r and isinstance(r[key], (int, float)):
                    sig[label] = float(r[key])
            for key in ("lbfgs_status", "lbfgs_stop_reason", "stop_reason", "adam_status", "status", "dtype"):
                if key in r and isinstance(r[key], str):
                    sig[f"train/meta_{key}"] = r[key]
            if "loss_history" in r and not found:
                sig.update(_from_list([{"stage": "adam", "loss": e["loss"]} for e in r["loss_history"]], "train/"))
                found = True
            if "adam_snapshots" in r:
                sn = r["adam_snapshots"]
                sig["train/n_snapshots"] = len(sn)
    if not found:
        nulls["train/history"] = "sin archivo de historia de pérdida junto al checkpoint"
    return sig, nulls
