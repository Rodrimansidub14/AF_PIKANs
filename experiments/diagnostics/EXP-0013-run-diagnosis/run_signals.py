#!/usr/bin/env python3
"""EXP-0013 — orquestador de señales (solo CPU, sin referencias). Lee results/inventory.csv y escribe una ficha JSON por checkpoint.

Uso:
  py -3.13 run_signals.py --smoke          # un checkpoint por (problema, campaña, brazo, etapa)
  py -3.13 run_signals.py                  # todo el universo (reanuda: salta lo ya calculado)
  py -3.13 run_signals.py --problem burgers_1d --limit 5
Salidas: results/signals/<problema>/<id>.json, results/signals_wide.csv, results/signals_long.csv, results/nulls.csv, results/run-log.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, RESULTS, env_info, log, now, sha256  # noqa: E402
from grids import get_grid  # noqa: E402
from adapters import load_bundle  # noqa: E402
from signals import compute_signals  # noqa: E402
from history import history_signals  # noqa: E402

SIG_DIR = RESULTS / "signals"
CODE_FILES = ["common.py", "grids.py", "adapters.py", "signals.py", "history.py", "run_signals.py"]


def row_id(r: dict) -> str:
    key = f"{r['campaign']}|{r['arm']}|{r['width']}|{r['seed']}|{r['stage']}|{r['path']}"
    short = hashlib.sha1(key.encode()).hexdigest()[:8]
    w = f"w{r['width']}" if r["width"] else ""
    import re
    return re.sub(r"[^A-Za-z0-9_.\-]", "", f"{r['campaign']}__{r['arm'].replace('+', 'p').replace('-', '')}{w}__s{r['seed']}__{r['stage']}__{short}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--problem")
    ap.add_argument("--campaign")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    rows = list(csv.DictReader(open(RESULTS / "inventory.csv", encoding="utf-8")))
    rows = [r for r in rows if r["load"] in ("ok", "not_tested")]
    if a.problem:
        rows = [r for r in rows if r["problem"] == a.problem]
    if a.campaign:
        rows = [r for r in rows if r["campaign"] == a.campaign]
    if a.smoke:
        seen, pick = set(), []
        for r in rows:
            k = (r["problem"], r["campaign"], r["arm"], r["stage"])
            if k not in seen:
                seen.add(k)
                pick.append(r)
        rows = pick
    if a.limit:
        rows = rows[:a.limit]
    code_hashes = {f: sha256(Path(__file__).resolve().parent / f) for f in CODE_FILES}
    env = env_info()
    log("run_start", n_rows=len(rows), smoke=a.smoke, env=env, code_sha256=code_hashes, inventory_sha256=sha256(RESULTS / "inventory.csv"))
    grids = {}
    done = fail = skipped = 0
    t_all = time.perf_counter()
    for i, r in enumerate(rows):
        rid = row_id(r)
        out = SIG_DIR / r["problem"] / f"{rid}.json"
        if out.exists() and not a.force:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        rec = {"id": rid, "row": r, "computed_at": now(), "env": {"torch": env["torch"], "python": env["python"]}, "code_sha256": code_hashes}
        try:
            g = grids.setdefault(r["problem"], get_grid(r["problem"]))
            b = load_bundle(r)
            rec["bundle"] = {"family": b.family, "kind": b.kind, "params": b.params, "dtype_native": b.dtype_native, "notes": b.notes,
                             "layers": len(b.layers) if b.layers else 0, "grid_source": g["source"], "grid_points": int(len(g["points"]))}
            sig, nulls = compute_signals(b, g["points"], g["shape"], g["axes"])
            hs, hn = history_signals(r)
            sig.update(hs)
            nulls.update(hn)
            rec["signals"], rec["nulls"], rec["status"] = sig, nulls, "ok"
            done += 1
        except Exception as e:  # noqa: BLE001
            rec["status"], rec["error"], rec["traceback"] = "error", f"{type(e).__name__}: {e}", traceback.format_exc()[-1500:]
            fail += 1
        rec["seconds"] = time.perf_counter() - t0
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        log("checkpoint", id=rid, problem=r["problem"], campaign=r["campaign"], arm=r["arm"], seed=r["seed"], stage=r["stage"], status=rec["status"],
            seconds=round(rec["seconds"], 2), n_signals=len(rec.get("signals", {})), n_nulls=len(rec.get("nulls", {})), error=rec.get("error"))
        print(f"[{i+1}/{len(rows)}] {rec['status']:5s} {rec['seconds']:6.1f}s {rid}" + (f"  ERR {rec['error'][:100]}" if rec["status"] == "error" else ""), flush=True)
    log("run_end", done=done, failed=fail, skipped=skipped, seconds=round(time.perf_counter() - t_all, 1))
    consolidate()


def consolidate():
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SIG_DIR.rglob("*.json"))]
    ok = [r for r in recs if r.get("status") == "ok"]
    keys = sorted({k for r in ok for k in r["signals"]})
    meta = ["id", "problem", "campaign", "arm", "width", "seed", "stage", "path", "family", "kind", "params", "dtype_native"]
    with open(RESULTS / "signals_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(meta + keys)
        for r in ok:
            row = r["row"]
            w.writerow([r["id"], row["problem"], row["campaign"], row["arm"], row["width"], row["seed"], row["stage"], row["path"], r["bundle"]["family"],
                        r["bundle"]["kind"], r["bundle"]["params"], r["bundle"]["dtype_native"]] + [r["signals"].get(k, "") for k in keys])
    with open(RESULTS / "signals_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "problem", "campaign", "arm", "seed", "stage", "signal", "value"])
        for r in ok:
            row = r["row"]
            for k, v in r["signals"].items():
                w.writerow([r["id"], row["problem"], row["campaign"], row["arm"], row["seed"], row["stage"], k, v])
    with open(RESULTS / "nulls.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "problem", "campaign", "arm", "seed", "stage", "signal_group", "reason"])
        for r in recs:
            row = r["row"]
            if r.get("status") != "ok":
                w.writerow([r["id"], row["problem"], row["campaign"], row["arm"], row["seed"], row["stage"], "*", f"ERROR {r.get('error')}"])
            else:
                for k, v in r["nulls"].items():
                    w.writerow([r["id"], row["problem"], row["campaign"], row["arm"], row["seed"], row["stage"], k, v])
    print(f"consolidado: {len(ok)} ok, {len(recs) - len(ok)} con error; {len(keys)} señales distintas")
    log("consolidate", ok=len(ok), errors=len(recs) - len(ok), n_signal_keys=len(keys))


if __name__ == "__main__":
    main()
