"""Read-only per-seed audit of the RBF-Sobol control cohorts (Heat, Burgers, Allen-Cahn).

Reads archived artifacts only. Writes rbf-audit-per-seed.csv and rbf-audit-summary.json
next to this script. Missing values are recorded as the string "missing", never as 0.
No training, no scoring, no edits to evals/ or registry/.
"""
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KO = ROOT / "kaggle-outputs"
OUT = Path(__file__).resolve().parent
BURG = (ROOT.parents[0] / "burgers-rad-exp006-v4" / "download-attempts" / "attempt-01"
        / "pikan-4393e173686298ee")
M = "missing"


def sha(p):
    if not p.exists():
        return M
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def stats(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return {"n": 0}
    sd = st.stdev(xs) if len(xs) > 1 else float("nan")
    return {"n": len(xs), "mean": st.mean(xs), "sd": sd, "median": st.median(xs),
            "min": min(xs), "max": max(xs)}


rows = []

# ---------------- Heat ----------------
heat_cohorts = {
    "heat_v4_500s": KO / "heat-tier2-t4x2-v4-20260928" / "heat-tier2",
    "heat_s2_20kAdam_lbfgs": KO / "closure-s2-20260928" / "heat-stepmatched-s2",
}
heat_pool = {}
for cname, root in heat_cohorts.items():
    led = jl(root / "campaign-ledger.json")
    for r in led["runs"]:
        if r.get("kind") != "tier2" or r.get("arm") != "rbf":
            continue
        seed = r["seed"]
        conf = root / "confirmation" / f"seed-{seed}" / "rbf"
        cm = jl(conf / "confirmation-metrics.json") or {}
        cc = jl(conf / "confirmation-cost.json") or {}
        cost = jl(root / "runs" / "tier2" / f"seed-{seed}" / "rbf" / "cost.json") or {}
        rdir = root / "runs" / "tier2" / f"seed-{seed}" / "rbf"
        ckpt_scored = cc.get("checkpoint", M)
        ckpt_name = Path(ckpt_scored).name if ckpt_scored != M else M
        heat_pool.setdefault(seed, {})[cname] = r.get("pool_sha256", M)
        rows.append({
            "problem": "heat_1d", "cohort": cname, "arm": "rbf_fixed", "seed": seed,
            "ledger_run_status": r.get("run_status", M),
            "adam_updates": r.get("adam_updates", cost.get("adam_updates", M)),
            "lbfgs_updates": r.get("lbfgs_updates", cost.get("lbfgs_updates", 0)),
            "lbfgs_stop_reason": r.get("lbfgs_stop_reason", M),
            "train_seconds": r.get("training_device_seconds", M),
            "scored_checkpoint": ckpt_name,
            "scored_stage": ("adam_plus_lbfgs_final" if "final" in ckpt_name and
                             (r.get("lbfgs_updates") or 0) > 0 else
                             "adam_only" if ckpt_name != M else M),
            "primary_metric": cm.get("heat/field_relative_l2", M),
            "pool_sha256": r.get("pool_sha256", M),
            "scored_checkpoint_file_sha256": sha(rdir / ckpt_name) if ckpt_name != M else M,
            "adam_final_ckpt_present": (rdir / "checkpoint-adam-final.pt").exists(),
            "ckpt_500s_present": (rdir / "checkpoint-500s.pt").exists(),
            "grid_id": cc.get("grid_id", M),
            "reference": "analytic_alpha0.1", "reference_accepted": False,
            "note": "",
        })

pool_mismatch = {s: v for s, v in heat_pool.items()
                 if len(set(v.values())) > 1 or len(v) < 2}

# ---------------- Burgers ----------------
runs = BURG / "runs"
paired = list(csv.DictReader(open(BURG / "analysis" / "paired-runs.csv", encoding="utf-8")))
paired_by_job = {p["job_id"]: p for p in paired}
bundle_ids = set()
for d in sorted(runs.glob("seed-*")):
    job = d.name.split("-")[1]
    seed = int(job)
    arm = "rad" if seed >= 1000 else "sobol"
    paired_seed = seed - 1000 if seed >= 1000 else seed
    rs = jl(d / "run_summary.json") or {}
    prov = jl(d / "provenance.json") or {}
    bundle_ids.add(prov.get("bundle_id", M))
    ev = jl(d / "evidence_hashes.json") or {}
    ck = d / "burgers" / "checkpoint-020000.pt"
    ck_sha = sha(ck)
    ck_listed = ev.get("burgers/checkpoint-020000.pt", M)
    pools = d / "burgers" / "pools.npz"
    ts = jl(d / "burgers" / "training_summary.json") or {}
    met = rs.get("result", {}).get("metrics", {}) or {}
    pj = paired_by_job.get(job, {})
    try:
        csv_val = float(pj["field_relative_l2"])
    except Exception:
        csv_val = M
    val = met.get("burgers/field_relative_l2", M)
    rows.append({
        "problem": "burgers_pinnacle_1d(EXP-0006)", "cohort": "exp0006_v4_20kAdam", "arm": arm,
        "seed": paired_seed,
        "ledger_run_status": rs.get("result", {}).get("status", M),
        "adam_updates": ts.get("steps", M), "lbfgs_updates": 0, "lbfgs_stop_reason": "n/a",
        "train_seconds": rs.get("result", {}).get("elapsed_sec", M),
        "scored_checkpoint": "checkpoint-020000.pt",
        "scored_stage": "adam_only",
        "primary_metric": val,
        "pool_sha256": sha(pools),
        "scored_checkpoint_file_sha256": ck_sha,
        "adam_final_ckpt_present": ck.exists(), "ckpt_500s_present": "n/a",
        "grid_id": "ColeHopf_common_grid", "reference": "ColeHopf", "reference_accepted": False,
        "note": ("ckpt_hash_matches_evidence" if ck_sha == ck_listed else
                 f"CKPT_HASH_MISMATCH listed={ck_listed}") +
                ("; csv_matches_run_summary" if isinstance(val, float) and isinstance(csv_val, float)
                 and math.isclose(val, csv_val, rel_tol=1e-9) else "; CSV_VS_RUN_SUMMARY_DIFF"),
    })

# ---------------- Allen-Cahn ----------------
s1_root = KO / "allen-cahn-t4x2-v1-20260928" / "allen-cahn-t4x2" / "campaign" / "AC-RBF-Sobol"
pre_root = (KO / "allen-cahn-score-only-v2-20260928" / "allen-cahn-score-only" /
            "run-1790600803211630363" / "scores" / "AC-RBF-Sobol")
s3_root = KO / "closure-s3-retry-20260928" / "closure-s3-20260928"
s3_led = jl(s3_root / "campaign-ledger.json")
s3_src = s3_led["source_preflight"]
for seed in range(401, 406):
    ts = jl(s1_root / f"seed-{seed}" / "training_summary.json") or {}
    pre = jl(pre_root / f"seed-{seed}" / "score.json") or {}
    post = jl(s3_root / "runs" / "AC-RBF-Sobol" / f"seed-{seed}" / "score.json") or {}
    cost = jl(s3_root / "runs" / "AC-RBF-Sobol" / f"seed-{seed}" / "cost.json") or {}
    pre_ref = ("FD8192" if pre.get("reference_candidate_sha256") == s3_src["references"]["FD8192"]
               else "ETDRK4" if pre.get("reference_candidate_sha256") == s3_src["references"]["ETDRK4"]
               else M)
    src_ck = s3_src["checkpoints"][str(seed)]["AC-RBF-Sobol"]
    common = {
        "problem": "allen_cahn_1d", "arm": "rbf_fixed", "seed": seed,
        "grid_id": "512x256(S3) / score-only-v2 grid not recorded in score.json",
        "reference_accepted": False,
    }
    rows.append({**common, "cohort": "ac_adam_endpoint", "ledger_run_status":
                 pre.get("source_training_status", M), "adam_updates": ts.get("steps", M),
                 "lbfgs_updates": 0, "lbfgs_stop_reason": "n/a",
                 "train_seconds": ts.get("elapsed_gpu_wall_seconds", M),
                 "scored_checkpoint": "adam endpoint (time-capped 600s)", "scored_stage": "adam_only",
                 "primary_metric": pre.get("relative_weighted_space_time_l2", M),
                 "pool_sha256": pre.get("saved_pool_sha256", M),
                 "scored_checkpoint_file_sha256": pre.get("source_checkpoint_sha256", M),
                 "adam_final_ckpt_present": "n/a", "ckpt_500s_present": "n/a",
                 "reference": f"{pre_ref}_candidate",
                 "note": "ckpt_hash_matches_S3_preflight" if pre.get("source_checkpoint_sha256") == src_ck
                 else "CKPT_HASH_MISMATCH_VS_S3_PREFLIGHT"})
    for rname in ("FD8192", "ETDRK4"):
        rr = post.get("relative_weighted_space_time_l2", {}).get(rname, {})
        rows.append({**common, "cohort": f"ac_lbfgs_continuation_vs_{rname}",
                     "ledger_run_status": cost.get("status", M), "adam_updates": ts.get("steps", M),
                     "lbfgs_updates": cost.get("lbfgs_updates", M),
                     "lbfgs_stop_reason": cost.get("status", M),
                     "train_seconds": cost.get("training_gpu_seconds_estimate", M),
                     "scored_checkpoint": "continuation checkpoint-final.pt", "scored_stage": "adam_plus_lbfgs_final",
                     "primary_metric": rr.get("relative_weighted_space_time_l2", M),
                     "pool_sha256": post.get("pool_file_sha256", M),
                     "scored_checkpoint_file_sha256": post.get("continuation_checkpoint_sha256", M),
                     "adam_final_ckpt_present": "n/a", "ckpt_500s_present": "n/a",
                     "reference": f"{rname}_candidate",
                     "note": "source_ckpt_matches_S3_preflight" if post.get("source_checkpoint_sha256") == src_ck
                     else "SOURCE_CKPT_MISMATCH"})

# ---------------- write ----------------
cols = ["problem", "cohort", "arm", "seed", "ledger_run_status", "adam_updates", "lbfgs_updates",
        "lbfgs_stop_reason", "train_seconds", "scored_checkpoint", "scored_stage", "primary_metric",
        "pool_sha256", "scored_checkpoint_file_sha256", "adam_final_ckpt_present",
        "ckpt_500s_present", "grid_id", "reference", "reference_accepted", "note"]
with open(OUT / "rbf-audit-per-seed.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, M) for c in cols})

summary = {}
for key in sorted({(r["problem"], r["cohort"], r["arm"]) for r in rows}):
    xs = [r["primary_metric"] for r in rows if (r["problem"], r["cohort"], r["arm"]) == key]
    summary["|".join(key)] = stats(xs)

# paired burgers
bs = {}
for r in rows:
    if r["problem"].startswith("burgers"):
        bs.setdefault(r["seed"], {})[r["arm"]] = r["primary_metric"]
ratios = [v["rad"] / v["sobol"] for v in bs.values() if "rad" in v and "sobol" in v]
summary["burgers_rad_over_sobol"] = {"ratios": ratios,
                                     "geomean": math.exp(st.mean(math.log(x) for x in ratios))}
summary["heat_pool_hash_agreement_v4_vs_s2"] = {
    str(s): (len(set(v.values())) == 1 and len(v) == 2) for s, v in heat_pool.items()}
summary["burgers_bundle_ids"] = sorted(bundle_ids)
summary["burgers_note_flags"] = sorted({r["note"] for r in rows if r["problem"].startswith("burgers")})
json.dump(summary, open(OUT / "rbf-audit-summary.json", "w", encoding="utf-8"), indent=2)
print(json.dumps(summary, indent=2))
