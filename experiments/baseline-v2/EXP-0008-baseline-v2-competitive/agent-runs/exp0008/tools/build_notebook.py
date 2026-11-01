"""Build notebooks/experiments/exp0008-baseline-v2-fast-track-signals.ipynb (phases = notebook sections)."""
import json
from pathlib import Path
import nbformat as nbf

REPO = Path(__file__).resolve().parents[4]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# EXP-0008 · Baseline v2 (ruta acelerada) — resultados y batería de señales

**Estado:** exploratorio, `scientific_accepted=false`, `reference_accepted=false`. Este notebook **no produce evidencia**
(rol `exploration`); sirve para leer las corridas y orientar el mecanismo. Contrato: `experiments/EXP-0008-baseline-v2-competitive/`.

Fases del notebook: 0 procedencia · 1 baselines de contexto · 2 puntuaciones por fase · 3 error vs coste ·
4 dinámica de optimización · 5 batería de señales (escalares) · 6 mapas densos · 7 SS-1 · 8 evaluadores (FEM 101×11, DFG σn) · 9 lectura de gates.
""")
code("""
import json, sys, hashlib, platform
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path.cwd()
while not (REPO / "AGENTS.md").exists() and REPO != REPO.parent: REPO = REPO.parent
EXP = REPO / "var/agent_runs/exp0008"; TOOLS = EXP / "tools"; sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(REPO))
RUNS = {"burgers": EXP / "ft0-burgers-cpu", "allen_cahn": EXP / "ft0-allen-cahn-cpu"}
def jload(p, default=None):
    p = Path(p); return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
print("repo", REPO, "| python", platform.python_version())
""")
md("## Fase 0 · Procedencia y contrato")
code("""
manifest = jload(REPO / "experiments/EXP-0008-baseline-v2-competitive/manifest.json")
print("manifest", manifest["version"], manifest["status"], "sha", sha(REPO / "experiments/EXP-0008-baseline-v2-competitive/manifest.json"))
prov = []
for prob, root in RUNS.items():
    for d in sorted(root.glob("seed-*")) if root.exists() else []:
        r = jload(d / "run_record.json", {})
        prov.append({"problem": prob, "seed": r.get("seed"), "recipe": r.get("recipe_id"), "device": r.get("device"), "torch": r.get("torch"),
                     "stop": (r.get("phases") or [{}])[0].get("stop"), "seconds": (r.get("phases") or [{}])[0].get("seconds"),
                     "pool_sha": (r.get("pool_sha256") or "")[:12], "src_ckpt_sha": (r.get("source_ckpt_sha256") or "")[:12], "status": r.get("status")})
pd.DataFrame(prov)
""")
md("## Fase 1 · Baselines de contexto (congelables, no aceptados)")
code("""
pd.DataFrame([{"id": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}} for k, v in manifest["frozen_baselines_to_register"].items()])
""")
md("## Fase 2 · Puntuaciones por fase (E + evaluador externo) y gates")
code("""
def phase_table(prob):
    S = jload(RUNS[prob] / "summary.json")
    if not S: return None
    rows = [{"phase": k, **{a: b for a, b in v.items() if a != "per_seed_main"}} for k, v in S.items() if k != "paired_log_ratio"]
    return pd.DataFrame(rows), S.get("paired_log_ratio")
for prob in RUNS:
    out = phase_table(prob)
    print("=====", prob)
    if out is None: print("  (sin summary.json: ejecutar tools/exp0008_collect.py)"); continue
    df, pr = out; display(df)
    print("  razón geométrica pareada (final/inicial):", pr)
""")
md("""
Lectura de gates propuestos (no firmados): Burgers **G1** razón pareada ≤ 0.25 con IC95 superior < 0.5; **G2** CH ≤ 1.13e-3 y FEM ≤ 1.99e-2 (el FEM es casi un piso, ver fase 8).
Allen–Cahn **G1** ≤ 4.6e-3, **G2** ≤ 2.0e-3 (referencias candidatas, sin aceptar).
""")
code("""
g = {}
S = jload(RUNS["burgers"] / "summary.json")
if S and "paired_log_ratio" in S:
    pr = S["paired_log_ratio"]; last = S.get("c_lbfgs_100", {})
    g["burgers_G1_ratio<=0.25"] = pr["geometric_ratio"] <= 0.25
    g["burgers_G1_ci_upper<0.5"] = pr["ci95_ratio"][1] < 0.5
    g["burgers_G2_CH<=1.13e-3"] = last.get("main_mean", 9) <= 1.13e-3
    g["burgers_G2_FEM<=1.99e-2"] = last.get("ext_mean", 9) <= 1.99e-2
S = jload(RUNS["allen_cahn"] / "summary.json")
if S:
    last = S.get("c_lbfgs_100", {}); g["ac_G1<=4.6e-3"] = last.get("main_mean", 9) <= 4.6e-3; g["ac_G2<=2.0e-3"] = last.get("main_mean", 9) <= 2.0e-3
pd.Series(g, name="cumple (propuesto, no firmado)").to_frame()
""")
md("## Fase 3 · Error vs coste (log)")
code("""
fig = make_subplots(rows=1, cols=2, subplot_titles=("Burgers · L2 rel. Cole–Hopf", "Allen–Cahn · L2 ponderado (ETDRK4 candidata)"))
for col, prob in enumerate(RUNS, 1):
    rows = jload(RUNS[prob] / "scores_by_phase.json", []) or []
    order = {"c_adam": 0, "c_adam_plus_s3": 0, "c_lbfgs_25": .25, "c_lbfgs_50": .5, "c_lbfgs_75": .75, "c_lbfgs_100": 1.0}
    for s in sorted({r["seed"] for r in rows}):
        pts = sorted([(order[r["tag"]] * (r.get("run_seconds_total") or 0), (r["E_ch_l2re_1025x129"] if prob == "burgers" else r["ETDRK4"]["relative_weighted_space_time_l2"])) for r in rows if r["seed"] == s])
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=f"{prob} s{s}"), row=1, col=col)
    fig.update_yaxes(type="log", row=1, col=col); fig.update_xaxes(title="segundos de L-BFGS", row=1, col=col)
fig.update_layout(height=420); fig.show()
""")
md("## Fase 4 · Dinámica de optimización (señales de L-BFGS)")
code("""
fig = make_subplots(rows=2, cols=2, subplot_titles=("pérdida", "‖g‖₂", "‖Δθ‖/‖θ‖", "evaluaciones acumuladas"))
for d in sorted(RUNS["burgers"].glob("seed-*")) if RUNS["burgers"].exists() else []:
    tr = pd.DataFrame([json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines()]) if (d / "trace.jsonl").exists() else None
    if tr is None or tr.empty: continue
    for (r, c), key in zip([(1, 1), (1, 2), (2, 1), (2, 2)], ["loss", "gnorm2", "update_ratio", "func_evals"]):
        fig.add_trace(go.Scatter(x=tr["sec"], y=tr[key], mode="lines", name=d.name, legendgroup=d.name, showlegend=(key == "loss")), row=r, col=c)
        if key != "func_evals": fig.update_yaxes(type="log", row=r, col=c)
fig.update_layout(height=650); fig.show()
""")
md("## Fase 5 · Batería de señales (escalares) antes y después de L-BFGS")
code("""
def signal_frame(prob, tags):
    rows = []
    for d in sorted(RUNS[prob].glob("seed-*")) if RUNS[prob].exists() else []:
        for tag in tags:
            s = jload(d / f"signals_{tag}.json")
            if s: rows.append({"seed": int(d.name.split("-")[1]), "tag": tag, **s})
    return pd.DataFrame(rows)
KEYS = ["grad/conflict_cos_ic", "grad/conflict_cos_bc", "grad/norm_ratio_ic", "jac/effective_rank", "jac/cond", "basis/layer1/dead_fraction",
        "basis/layer0/effective_count", "burgers/antisym_err", "burgers/energy_balance_rel", "burgers/front_width_late", "burgers/ux_max_late",
        "residual/rms", "residual/p95", "grad_residual/p95", "pool/residual_top_decile_mass", "pool/residual_ess_fraction", "pool/region_residual_mass",
        "basis/influence_residual_layer0/max", "sensitivity/median"]
sb = signal_frame("burgers", ["c_adam", "c_lbfgs_100"])
if not sb.empty:
    keep = [k for k in KEYS if k in sb.columns]
    display(sb.groupby("tag")[keep].agg(["mean", "std"]).T)
""")
code("""
sa = signal_frame("allen_cahn", ["c_adam_plus_s3", "c_lbfgs_100"])
if not sa.empty:
    keys = [c for c in ["jac/effective_rank", "jac/cond", "basis/layer1/dead_fraction", "ac/evenness_err", "ac/energy_t0", "ac/energy_t1", "ac/energy_increments",
                        "ac/n_interfaces_t0.5", "ac/max_abs_ux_t0.5", "ac/interface_band_fraction", "residual/rms", "residual/p95", "grad_residual/p95",
                        "pool/residual_top_decile_mass", "pool/residual_ess_fraction"] if c in sa.columns]
    display(sa.groupby("tag")[keys].agg(["mean", "std"]).T)
""")
md("## Fase 6 · Mapas densos (rasgos vs. etiqueta) — semilla 201")
code("""
def maps(prob, seed, tag, nx=129, nt=65):
    d = RUNS[prob] / f"seed-{seed}"
    f, l = d / f"dense_signals_{tag}.npz", d / f"labels_{tag}.npz"
    if not (f.exists() and l.exists()): print("faltan", f.name, l.name); return
    F = np.load(f); E = np.load(l)["abs_error"]
    names = [("|error| (etiqueta)", E), ("|residual|", np.abs(F["residual"])), ("|∇residual|", F["grad_residual_norm"]), ("|u_x|", np.abs(F["u_x"])),
             ("dist. a interfaz (rasgo)", F["interface_distance"] if "interface_distance" in F.files else np.zeros_like(E)), ("entropía bases capa 0", F["basis0_entropy"])]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[n for n, _ in names])
    for i, (n, v) in enumerate(names):
        z = v.reshape(nx, nt).T; z = np.log10(np.abs(z) + 1e-12) if ("error" in n or "residual" in n or "u_x" in n) else z
        fig.add_trace(go.Heatmap(z=z, x=np.linspace(-1, 1, nx), y=np.linspace(0, 1, nt), colorscale="Viridis", showscale=False), row=i // 3 + 1, col=i % 3 + 1)
    fig.update_layout(height=600, title=f"{prob} seed {seed} · {tag} (log10 en error/residual/u_x)"); fig.show()
maps("burgers", 201, "c_adam"); maps("burgers", 201, "c_lbfgs_100")
""")
md("""
## Fase 7 · SS-1 — ¿aportan las señales información sobre el error más allá del residual?

Modelos M0 residual · M1 +geometría · M2 +derivadas · M3 +señales de red/bases; validación dejando fuera una semilla; unidad = semilla.
Con 5 semillas es **exploratorio**: los criterios P0–P4 de `rad-gate-and-v2.md` exigen ≥ 2 problemas.
""")
code("""
import exp0008_ss1 as ss1
ss = {}
for prob, tags in (("burgers", ["c_adam", "c_lbfgs_100"]), ("allen_cahn", ["c_adam_plus_s3", "c_lbfgs_100"])):
    if not RUNS[prob].exists(): continue
    for tag in tags:
        data = ss1.load(sorted(RUNS[prob].glob("seed-*")), tag)
        if len(data) >= 3:
            r = ss1.analyse(data); ss[(prob, tag)] = r
rows = []
for (prob, tag), r in ss.items():
    rows.append({"problem": prob, "checkpoint": tag, "n_seeds": r["n_seeds"], "Spearman(res,err)": r["spearman_mean"],
                 **{f"AUC {m}": v for m, v in r["auc_mean"].items() if m != "M2_on_M3_subset"},
                 "ΔAUC M1−M0": r["delta_auc"]["M1-M0"]["mean"], "ΔAUC M2−M1": r["delta_auc"]["M2-M1"]["mean"],
                 "ΔAUC M2−M1 IC95": r["delta_auc"]["M2-M1"]["ci95"], "ΔAUC M3−M2": r["delta_auc"]["M3-M2(subset)"]["mean"], "ΔAUC M3−M2 IC95": r["delta_auc"]["M3-M2(subset)"]["ci95"]})
pd.DataFrame(rows)
""")
code("""
p0 = [(k, r["spearman_mean"] >= 0.5 and r["auc_mean"]["M0"] >= 0.75) for k, r in ss.items() if k[1] in ("c_lbfgs_100",)]
print("P0 (residual proxy útil, tras L-BFGS):", p0 or "sin datos")
print("Regla: RAD residual-only solo si P0 se cumple en ≥2 problemas con referencia (rad-gate-and-v2.md).")
""")
md("## Fase 8 · Evaluadores: FEM 101×11 de PINNacle (WP-E1) y scorer σn de DFG (WP-E3)")
code("""
import glob
rows = []
for f in sorted(glob.glob(str(EXP / "wp-e1" / "*.json"))):
    r = json.load(open(f)); ch = np.array([x["E_ch_l2re_1025x129"] for x in r]); fe = np.array([x["external_fem_101x11_l2re"] for x in r])
    rows.append({"conjunto": Path(f).stem, "n": len(r), "CH 1025×129": f"{ch.mean():.4g} ± {ch.std(ddof=1):.3g}", "FEM 101×11": f"{fe.mean():.4g} ± {fe.std(ddof=1):.3g}", "piso FEM-vs-CH": r[0]["fem_floor_fem_vs_ch"]})
display(pd.DataFrame(rows))
""")
code("""
r = jload(EXP / "wp-e3/dfg_rescore_sigma_n.json", [])
rows = []
for coh, arm, st in [("301-305", "DFG-RBF-PIKAN", "pre_lbfgs"), ("301-305", "DFG-RBF-PIKAN", "post_lbfgs"), ("501-505", "DFG-RBF-PIKAN", "pre_lbfgs"),
                     ("501-505", "DFG-RBF-PIKAN", "post_lbfgs"), ("501-505", "DFG-Tanh-MLP", "pre_lbfgs"), ("501-505", "DFG-Tanh-MLP", "post_lbfgs")]:
    x = [e for e in r if e["cohort"] == coh and e["arm"] == arm and e["stage"] == st and "sigma_n" in e]
    if not x: continue
    cd = np.array([e["sigma_n"]["drag_coefficient"] for e in x]); dp = np.array([e["sigma_n"]["pressure_difference"] for e in x]); cl = np.array([e["sigma_n"]["lift_coefficient"] for e in x])
    rows.append({"cohorte": coh, "brazo": arm, "etapa": st, "Cd": f"{cd.mean():.3f} ± {cd.std(ddof=1):.3f}", "err Cd %": 100 * np.mean([e["sigma_n_err"]["cd_rel_err"] for e in x]),
                 "Cl": f"{cl.mean():+.3f} ± {cl.std(ddof=1):.3f}", "Δp": f"{dp.mean():.4f}", "err Δp %": 100 * np.mean([e["sigma_n_err"]["dp_rel_err"] for e in x])})
display(pd.DataFrame(rows))
""")
md("## Fase 9 · Lectura y siguiente decisión (humana)")
code("""
print("scientific_accepted = False · reference_accepted = False · thresholds: propuestos, no firmados")
print("Decisión humana pendiente: firma del preregistro, congelación de baselines (eval_integrity), aceptación de referencias AC/DFG, presupuesto GPU por tramo.")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "exp0008": {"role": "exploration", "scientific_accepted": False}}
out = REPO / "notebooks/experiments/exp0008-baseline-v2-fast-track-signals.ipynb"
nbf.write(nb, out); print("wrote", out)
