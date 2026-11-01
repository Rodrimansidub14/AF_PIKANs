#!/usr/bin/env python3
"""EXP-0013 — etiquetas y auditoría de alineación (usa referencias; módulo separado de las señales). Solo CPU.

Uso: py -3.13 run_labels.py [--problem P] [--limit N] [--force]
Salidas: results/labels/<problema>/<id>.json, results/labels_wide.csv, results/run-log.jsonl y arreglos por punto en var/agent_runs/exp0013-fields/ (fuera de git).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, RESULTS, log, now, sha256, env_info  # noqa: E402
from grids import get_grid  # noqa: E402
from adapters import load_bundle, _to_t  # noqa: E402
from labels import label_row  # noqa: E402
from run_signals import row_id  # noqa: E402

LAB_DIR = RESULTS / "labels"
FIELD_DIR = REPO / "var/agent_runs/exp0013-fields"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(RESULTS / "inventory.csv", encoding="utf-8")) if r["load"] in ("ok", "not_tested")]
    if a.problem:
        rows = [r for r in rows if r["problem"] == a.problem]
    if a.limit:
        rows = rows[:a.limit]
    log("labels_start", n_rows=len(rows), code_sha256={f: sha256(Path(__file__).resolve().parent / f) for f in ("labels.py", "run_labels.py")}, env=env_info())
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    grids, done, fail = {}, 0, 0
    for i, r in enumerate(rows):
        rid = row_id(r)
        out = LAB_DIR / r["problem"] / f"{rid}.json"
        if out.exists() and not a.force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        rec = {"id": rid, "row": r, "computed_at": now()}
        try:
            g = grids.setdefault(r["problem"], get_grid(r["problem"]))
            b = load_bundle(r)
            pts = g["points"]
            with torch.no_grad():
                f = np.concatenate([b.fields(_to_t(pts[k:k + 8192], requires_grad=False)).numpy() for k in range(0, len(pts), 8192)])
            anchor = None
            if r["problem"] == "dfg_2d":
                with torch.no_grad():
                    anchor = float(b.fields(torch.tensor([[2.2, 0.2]]))[0, 2])
            res = b.residuals(pts)
            allr = np.sqrt(sum(v ** 2 for v in res.values()))
            lab, eall = label_row(r["problem"], pts, f, allr, anchor)
            rec["labels"], rec["status"] = lab, "ok"
            np.savez_compressed(FIELD_DIR / f"{rid}.npz", points=pts.astype(np.float32), err_all=eall.astype(np.float32), res_all=allr.astype(np.float32))
            done += 1
        except Exception as e:  # noqa: BLE001
            rec["status"], rec["error"], rec["traceback"] = "error", f"{type(e).__name__}: {e}", traceback.format_exc()[-1500:]
            fail += 1
        rec["seconds"] = time.perf_counter() - t0
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        log("label", id=rid, problem=r["problem"], campaign=r["campaign"], arm=r["arm"], seed=r["seed"], stage=r["stage"], status=rec["status"], seconds=round(rec["seconds"], 2), error=rec.get("error"))
        print(f"[{i+1}/{len(rows)}] {rec['status']:5s} {rec['seconds']:5.1f}s {rid}" + (f"  ERR {rec['error'][:100]}" if rec["status"] == "error" else ""), flush=True)
    log("labels_end", done=done, failed=fail)
    consolidate()


def consolidate():
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(LAB_DIR.rglob("*.json"))]
    ok = [r for r in recs if r.get("status") == "ok"]
    keys = sorted({k for r in ok for k in r["labels"]})
    with open(RESULTS / "labels_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "problem", "campaign", "arm", "width", "seed", "stage"] + keys)
        for r in ok:
            row = r["row"]
            w.writerow([r["id"], row["problem"], row["campaign"], row["arm"], row["width"], row["seed"], row["stage"]] + [r["labels"].get(k, "") for k in keys])
    print(f"etiquetas: {len(ok)} ok, {len(recs) - len(ok)} con error; {len(keys)} columnas")


if __name__ == "__main__":
    main()
