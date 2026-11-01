"""Ronda 1-C (BORRADOR SIN FIRMA): analisis preregistrado. Se prueba con los datos de 1-B (T1 + V de Burgers); NO se ha corrido con datos de 1-C.

Por problema y por brazo (E, F) frente a B, misma semilla:
  - razon pareada r_s = error_final(brazo) / error_final(B)  (L2 relativo tras L-BFGS, campo 'primary_final'),
  - mediana, media geometrica, numero de semillas con mejora (r < 1), razon maxima, n y ausencias (no se imputan),
  - razones por region (interfaz = 5 % de mayor |u_x| de la referencia / resto) si las corridas las traen (solo 1-C),
  - Spearman entre cond(P) inicial de la capa 1 (propiedad de la semilla; ver arnes_round1c.p_cond_probe) y log(r_s).
CONTRASTE PREREGISTRADO UNICO (borrador): brazo E, problema 'burgers' (bloque P, semillas nuevas 11-20, arnes de 1-B), Spearman(cond_init capa 1, log r_s),
  direccion esperada NEGATIVA (a mayor condicionamiento inicial, menor razon = mejor), prueba de una cola. Todo lo demas es descriptivo (sin correccion).
La fuente de cond_init es el campo 'cond_init' (corridas de 1-C: sonda sobre cualquier brazo). Para los datos de 1-B (sin sonda) se usa el cond(P) de la capa 1 en el
paso 0 de la corrida E de la MISMA semilla (valido: el modelo inicial es identico en B/E/F de una semilla).
Uso: python analyze_round1c.py --dir <carpeta> [--dir <otra>] --problem burgers [--contrast-problem burgers] [--out analisis.json]
"""
import argparse, glob, json, math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

CONTRAST = dict(arm="E", problem="burgers", layer=1, direction="negative", one_sided=True)


def load_rows(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(str(Path(d) / "*-seed*-a*-l*.json")):
            if not f.endswith(".note.json"):
                rows.append(json.load(open(f)))
    return rows


def cond_init_by_seed(rows, problem, layer=1):
    out = {}
    for r in rows:
        if r["problem"] != problem:
            continue
        if r.get("cond_init") is not None:
            out.setdefault(r["seed"], r["cond_init"][layer])
        elif r["arm"] == "E" and r.get("pre_diag"):
            out.setdefault(r["seed"], r["pre_diag"][0]["p_cond"][layer])
    return out


def analyze(rows, problem, layer=1):
    get = lambda arm, seed: next((r for r in rows if r["problem"] == problem and r["arm"] == arm and r["seed"] == seed), None)
    seeds = sorted({r["seed"] for r in rows if r["problem"] == problem})
    cond = cond_init_by_seed(rows, problem, layer)
    out = {}
    for arm in ("E", "F"):
        pairs, missing = [], []
        for s in seeds:
            a, b = get(arm, s), get("B", s)
            if a is None or b is None or a.get("primary_final") is None or b.get("primary_final") is None:
                missing.append(s)
            else:
                pairs.append((s, a, b))
        if not pairs:
            out[arm] = dict(n=0, missing=missing)
            continue
        r = np.array([a["primary_final"] / b["primary_final"] for _, a, b in pairs])
        entry = dict(n=len(pairs), missing=missing, seeds=[s for s, _, _ in pairs], ratios=[float(x) for x in r], median=float(np.median(r)),
                     geomean=float(np.exp(np.mean(np.log(r)))), improved=int((r < 1).sum()), max_ratio=float(r.max()))
        reg = {}
        for name in ("top", "rest"):
            if all(a.get("region_final") and b.get("region_final") for _, a, b in pairs):
                rr = np.array([a["region_final"][f"l2re_{name}"] / b["region_final"][f"l2re_{name}"] for _, a, b in pairs])
                reg[name] = dict(median=float(np.median(rr)), improved=int((rr < 1).sum()), ratios=[float(x) for x in rr])
        if reg:
            entry["by_region"] = reg
        cs = [cond.get(s) for s, _, _ in pairs]
        if all(c is not None for c in cs) and len(pairs) >= 4 and len(set(cs)) > 1:
            rho, p = spearmanr(cs, np.log(r))
            entry["cond_init_layer"] = layer
            entry["spearman_cond_init_vs_log_ratio"] = dict(rho=float(rho), p_two_sided=float(p), p_one_sided_negative=float(p / 2 if rho < 0 else 1 - p / 2), n=len(pairs))
        out[arm] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--contrast-problem", default=CONTRAST["problem"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = analyze(load_rows(a.dir), a.problem)
    res["preregistered_contrast"] = dict(CONTRAST, applies_to_this_problem=bool(a.problem == a.contrast_problem))
    for arm in ("E", "F"):
        e = res[arm]
        if e.get("n"):
            sp = e.get("spearman_cond_init_vs_log_ratio")
            print(f"{a.problem} {arm}: n={e['n']} mediana {e['median']:.3f} geomean {e['geomean']:.3f} mejora {e['improved']}/{e['n']} max {e['max_ratio']:.2f}"
                  + (f"  Spearman cond_init~log r: rho {sp['rho']:+.2f} p2 {sp['p_two_sided']:.3f}" if sp else ""))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
