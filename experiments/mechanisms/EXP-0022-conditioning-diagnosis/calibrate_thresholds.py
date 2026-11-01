"""EXP-0022: calibracion de umbrales con la variabilidad del propio calculo (no con el efecto). Para un estado de cada problema:
(a) mitades disjuntas del pool de entrenamiento (variabilidad de muestreo de las razones que entran a los criterios),
(b) J en float64 frente a float32 (error numerico de los autovalores). Escribe calibration.json. Uso: python calibrate_thresholds.py"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r", str(HERE / "run_exp0022.py")); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
torch.set_num_threads(2)
CASES = [("burgers", 302, "0.6"), ("ac", 602, "0.6")]   # estados distintos del de la prueba de presupuesto (301, 0.25)


def ratios(o):
    k = o["kappa"]
    return dict(edge_over_raw=k["edge"]["100"] / k["raw"]["100"], edge_over_jacobi=k["edge"]["100"] / k["jacobi"]["100"], rho_ratio=o["rho_ratio"],
                kappa_raw_100=k["raw"]["100"], kappa_edge_100=k["edge"]["100"], kappa_jacobi_100=k["jacobi"]["100"])


out = {}
for problem, seed, f in CASES:
    cfg = r.PROBLEMS[problem]; prob = r.pc.make_problem(problem, str(cfg["ref"]))
    h0 = r.process(problem, cfg, prob, seed, f, 2048, subset=0)
    h1 = r.process(problem, cfg, prob, seed, f, 2048, subset=1)
    d64 = r.process(problem, cfg, prob, seed, f, 2048, subset=0, jdtype=torch.float64)
    e32, e64 = np.array(h0["spectrum_raw"]), np.array(d64["spectrum_raw"])
    prec = {str(m): float(abs(e32[m - 1] - e64[m - 1]) / e64[m - 1]) for m in (10, 100, 500)}
    a, b = ratios(h0), ratios(h1)
    spread = {k: float(abs(np.log(a[k] / b[k]))) for k in ("edge_over_raw", "edge_over_jacobi", "rho_ratio")}
    out[problem] = dict(state=[seed, f], half0=a, half1=b, log_ratio_between_halves=spread, float32_vs_float64_rel_error=prec,
                        seconds=[h0["seconds_total"], h1["seconds_total"], d64["seconds_total"]])
    print(problem, json.dumps(out[problem], indent=1), flush=True)
(HERE / "calibration.json").write_text(json.dumps(out, indent=1))
