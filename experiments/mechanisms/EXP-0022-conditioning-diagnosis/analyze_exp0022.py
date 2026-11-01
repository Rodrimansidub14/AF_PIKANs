"""EXP-0022: aplica los criterios firmados a results/states/*.json (o a la carpeta dada). Uso: python analyze_exp0022.py [carpeta]"""
import glob, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/states"
rows = [json.load(open(f)) for f in sorted(glob.glob(str(D / "*.json")))]
out = {}
for prob in ("burgers", "ac"):
    R = [r for r in rows if r["problem"] == prob]
    if not R:
        continue
    k = lambda r, a, m="100": r["kappa"][a][m]
    e_raw = np.array([k(r, "edge") / k(r, "raw") for r in R]); e_jac = np.array([k(r, "edge") / k(r, "jacobi") for r in R])
    e_inp = np.array([k(r, "input") / k(r, "raw") for r in R]); rho = np.array([r["rho_ratio"] for r in R])
    e_raw10 = np.array([k(r, "edge", "10") / k(r, "raw", "10") for r in R])
    out[prob] = dict(
        n=len(R),
        H_COND_1=dict(median_rho_ratio=float(np.median(rho)), threshold=0.1, holds=bool(np.median(rho) <= 0.1), frac_states_below=float((rho <= 0.1).mean())),
        H_COND_2=dict(median_edge_over_raw=float(np.median(e_raw)), median_edge_over_jacobi=float(np.median(e_jac)),
                      holds_raw=bool(np.median(e_raw) <= 0.1), holds_jacobi=bool(np.median(e_jac) <= 0.33),
                      holds=bool(np.median(e_raw) <= 0.1 and np.median(e_jac) <= 0.33),
                      intermediate=bool(np.median(e_raw) <= 0.1 and 0.33 < np.median(e_jac) <= 0.6)),
        descriptive=dict(median_input_over_raw=float(np.median(e_inp)), median_edge_over_raw_m10=float(np.median(e_raw10)),
                         median_kappa_raw_100=float(np.median([k(r, "raw") for r in R])),
                         median_disp_norm_rel=float(np.median([r["disp_norm_rel"] for r in R])),
                         energy_top20_by_layer_median={l: float(np.median([r["energy_top20_by_layer"][l] for r in R if r["energy_top20_by_layer"][l] is not None])) for l in ("0", "1", "2")},
                         gram_basis_cond_median_by_layer={l: float(np.median([r["gram_basis"][l]["cond_median"] for r in R])) for l in ("0", "1", "2")},
                         gram_uniform_cond_median_by_layer={l: float(np.median([r["gram_basis"][l]["cond_uniform_median"] for r in R])) for l in ("0", "1", "2")}))
h1 = all(v["H_COND_1"]["holds"] for v in out.values()); h2 = all(v["H_COND_2"]["holds"] for v in out.values())
mid = all(v["H_COND_2"]["holds_raw"] for v in out.values()) and not h2 and all(v["H_COND_2"]["median_edge_over_jacobi"] <= 0.6 for v in out.values())
out["verdict"] = dict(H_COND_1=h1, H_COND_2=h2, intermediate_row=bool(mid and h1))
print(json.dumps(out, indent=1))
(D.parent / "analysis.json").write_text(json.dumps(out, indent=1)) if D.name == "states" else None
