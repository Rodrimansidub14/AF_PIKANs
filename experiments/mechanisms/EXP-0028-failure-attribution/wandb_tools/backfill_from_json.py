"""Reconstruye runs de W&B a partir de los JSON por corrida de EXP-0028.

IMPORTANTE: son runs RECONSTRUIDOS desde JSON, no corridas instrumentadas.
Solo contienen lo que se guardo (hist, lbfgs_traj, pre_diag, scores, tiempos,
config). No existen pesos, campos ni residuos por punto.
No lee ni usa WANDB_API_KEY (modo offline por defecto).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import read_env, public_cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
PHASES = ["phase1a", "phase1b", "phase1c-t1", "phase1c-dfg", "phase1d-ac",
          "phase1d-kpp", "phase1d-bu1", "phase1e-ns1"]
SCORED = {
    "phase1a": "phase1_scored.json", "phase1b": "phase1_scored.json",
    "phase1c-t1": "phase1c_scored.json", "phase1c-dfg": "phase1c_dfg_scored.json",
    "phase1d-ac": "phase1d_ac_kpp_scored.json", "phase1d-kpp": "phase1d_ac_kpp_scored.json",
    "phase1d-bu1": "phase1d_bu1_scored.json",
}
SERIES = {"hist", "lbfgs_traj", "pre_diag", "regrid_diag"}
SUMMARY_KEYS = ("primary_adam", "primary_final", "loss_adam_end", "loss_final",
                "adam_seconds", "lbfgs_seconds", "total_seconds", "pre_seconds",
                "lbfgs_iterations", "lbfgs_status", "adam_status")
NOTE = ("RECONSTRUIDO DESDE JSON (EXP-0028): no es una corrida instrumentada. "
        "Series y escalares provienen del JSON guardado por corrida; no hay "
        "pesos, campos ni residuos por punto.")


def discover(phases):
    ph = list(phases)
    for d in sorted(RES.glob("phase1e-*")):
        if d.name not in ph and d.is_dir():
            ph.append(d.name)
    for p in ph:
        for f in sorted((RES / p).rglob("*-seed*.json")):
            yield p, f


def scored_class(phase, j):
    name = SCORED.get(phase)
    if not name or not (RES / name).exists():
        return {}
    try:
        s = json.loads((RES / name).read_text(encoding="utf-8"))
        node = s.get(j["problem"]) or {}
        if "brazos" not in node:  # 1c / 1c-dfg: un nivel por metrica
            node = s.get(j.get("primary_metric") or "") or {}
        arm = node.get("brazos", {}).get(j["arm"], {})
        out = {}
        for k in ("adam", "final"):
            if isinstance(arm.get(k), dict) and "clase" in arm[k]:
                out[f"clase_{k}_agregada_brazo"] = arm[k]["clase"]
        return out
    except Exception:
        return {}


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def log_series(run, name, rows, step_key):
    if not rows:
        return
    run.define_metric(f"{name}/step", hidden=True)
    run.define_metric(f"{name}/*", step_metric=f"{name}/step")
    for i, r in enumerate(rows):
        d = {f"{name}/step": r.get(step_key, i)}
        for k, v in r.items():
            if k == step_key:
                continue
            if num(v):
                d[f"{name}/{k}"] = v
            elif isinstance(v, list) and v and all(num(x) for x in v):
                for n, x in enumerate(v):
                    d[f"{name}/{k}_{n}"] = x
        run.log(d)


def build(wandb, phase, f, j, args, ident):
    problem, arm, seed = j["problem"], j["arm"], j["seed"]
    name = f"{problem}-{arm}-seed{seed}"
    rid = hashlib.sha1(f"{phase}/{name}".encode()).hexdigest()[:10]
    out = Path(args.out_dir)
    if list(out.glob(f"wandb/offline-run-*-{rid}")) and not args.force:
        return "existe"
    cfg = {k: v for k, v in j.items()
           if k not in SERIES and not k.startswith("score_") and k not in SUMMARY_KEYS
           and k not in ("terms_adam_end", "terms_final")}
    cfg.update(fase=phase, reconstruido_desde_json=True, instrumentado=False,
               fuente_json=str(f.relative_to(RES.parent)).replace("\\", "/"),
               experimento="EXP-0028")
    out.mkdir(parents=True, exist_ok=True)
    run = wandb.init(mode=args.mode, dir=str(out), id=rid, name=name, group=phase,
                     job_type=problem, entity=ident["entity"], project=ident["project"],
                     tags=[phase, problem, arm, "reconstruido_desde_json"],
                     notes=NOTE, config=cfg, reinit=True)
    log_series(run, "adam", j.get("hist") or [], "step")
    log_series(run, "lbfgs", j.get("lbfgs_traj") or [], "iters")
    log_series(run, "pre", j.get("pre_diag") or [], "step")
    log_series(run, "regrid", j.get("regrid_diag") or [], "step")
    summ = {k: j[k] for k in SUMMARY_KEYS if k in j}
    for k in ("score_adam", "score_final", "terms_adam_end", "terms_final"):
        if isinstance(j.get(k), dict):
            for kk, v in j[k].items():
                if num(v):
                    summ[f"{k}/{kk}"] = v
    summ.update(scored_class(phase, j))
    summ["reconstruido_desde_json"] = True
    for k, v in summ.items():
        run.summary[k] = v
    run.finish()
    return "creado"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "online"], default="offline")
    ap.add_argument("--out-dir", default=str(HERE / "offline"))
    ap.add_argument("--phases", nargs="*", default=PHASES)
    ap.add_argument("--limit", type=int, help="maximo de runs (humo)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    import wandb
    ident = public_cfg(read_env())
    cnt, per = 0, {}
    for phase, f in discover(args.phases):
        if args.limit is not None and cnt >= args.limit:
            break
        j = json.loads(f.read_text(encoding="utf-8"))
        if "problem" not in j or "arm" not in j or "seed" not in j:
            continue
        st = build(wandb, phase, f, j, args, ident)
        per.setdefault(phase, {}).setdefault(st, 0)
        per[phase][st] += 1
        cnt += 1
    print(json.dumps(per, indent=1))
    print("total procesados:", cnt)


if __name__ == "__main__":
    main()
