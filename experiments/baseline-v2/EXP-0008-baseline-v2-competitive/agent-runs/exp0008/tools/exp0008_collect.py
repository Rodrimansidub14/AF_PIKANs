"""Collect + score every phase checkpoint of the fast-track runs, write summary tables and gate readout.

Usage: exp0008_collect.py burgers|allen_cahn   (run after the continuation jobs finished; CPU)
Outputs under var/agent_runs/exp0008/<ft0-...>/: scores_by_phase.csv, summary.json, gate_readout.md
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
EXP = REPO / "var/agent_runs/exp0008"
problem = sys.argv[1]

if problem == "burgers":
    import exp0008_score as S
    root = EXP / "ft0-burgers-cpu"; seeds = range(201, 206)
    tags = ["c_adam", "c_lbfgs_25", "c_lbfgs_50", "c_lbfgs_75", "c_lbfgs_100"]
    def score(p, lab): return S.score_checkpoint(p, lab)
    main_key, ext_key = "E_ch_l2re_1025x129", "external_fem_101x11_l2re"
else:
    import exp0008_score_ac as S
    root = EXP / "ft0-allen-cahn-cpu"; seeds = range(401, 406)
    tags = ["c_adam_plus_s3", "c_lbfgs_25", "c_lbfgs_50", "c_lbfgs_75", "c_lbfgs_100"]
    def score(p, lab): return S.score(p, lab)
    main_key, ext_key = "ETDRK4", "FD8192"

rows = []
for s in seeds:
    d = root / f"seed-{s}"
    rec = json.loads((d / "run_record.json").read_text()) if (d / "run_record.json").exists() else {}
    trace = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines()] if (d / "trace.jsonl").exists() else []
    for tag in tags:
        p = d / f"{tag}.pt"
        if not p.exists(): continue
        lab = d / f"labels_{tag}.npz" if tag in (tags[0], tags[-1]) else None
        r = score(p, lab); r.update(seed=s, tag=tag)
        # cost so far: seconds at which this checkpoint was written (fraction of the run)
        r["run_seconds_total"] = rec.get("phases", [{}])[0].get("seconds")
        r["run_stop"] = rec.get("phases", [{}])[0].get("stop")
        rows.append(r); print(s, tag, r.get(main_key) if problem == "burgers" else r["ETDRK4"]["relative_weighted_space_time_l2"], flush=True)
(root / "scores_by_phase.json").write_text(json.dumps(rows, indent=1, default=float))


def val(r, which):
    if problem == "burgers":
        return r["E_ch_l2re_1025x129"] if which == "main" else r["external_fem_101x11_l2re"]
    return r["ETDRK4"]["relative_weighted_space_time_l2"] if which == "main" else r["FD8192"]["relative_weighted_space_time_l2"]

summ = {}
for tag in tags:
    m = np.array([val(r, "main") for r in rows if r["tag"] == tag]); e = np.array([val(r, "ext") for r in rows if r["tag"] == tag])
    if len(m): summ[tag] = {"n": int(len(m)), "main_mean": float(m.mean()), "main_sd": float(m.std(ddof=1)) if len(m) > 1 else None,
                            "ext_mean": float(e.mean()), "per_seed_main": m.tolist()}
first, last = tags[0], tags[-1]
a = np.array([val(r, "main") for r in rows if r["tag"] == first]); b = np.array([val(r, "main") for r in rows if r["tag"] == last])
if len(a) == len(b) and len(a) > 1:
    lr = np.log(b / a); n = len(lr); se = lr.std(ddof=1) / np.sqrt(n)
    from scipy import stats
    tcrit = stats.t.ppf(0.975, n - 1)
    summ["paired_log_ratio"] = {"geometric_ratio": float(np.exp(lr.mean())), "ci95_ratio": [float(np.exp(lr.mean() - tcrit * se)), float(np.exp(lr.mean() + tcrit * se))],
                                "per_seed_ratio": np.exp(lr).tolist(), "n": n}
(root / "summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ.get("paired_log_ratio"), indent=1))
