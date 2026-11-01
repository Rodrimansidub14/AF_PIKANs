"""EXP-0024: ¿la reubicacion (regrid_layer / regrid_kan) da lo mismo en CPU y en GPU sobre el MISMO estado? Prueba de pocos segundos, sin entrenar.

Estado: Burgers RBF48, semilla 1, 50 pasos de Adam en float32 (donde D1 reubica), guardado en float64 (`d1_state_f64.pt`), con los mismos puntos de colocacion.
Variante principal (la pedida): modelo y reubicacion en float64. Variante de produccion: modelo en float32 (los pesos float64 redondeados una vez, iguales en ambos dispositivos);
la reubicacion calcula en float64 internamente.
En cada dispositivo se ejecuta `regrid_kan` (mix = 1.0, como D1) y se comparan: centros, anchos y coeficientes de las capas 1 y 2, la salida del modelo en 2048 puntos justo
despues de reubicar, y los diagnosticos (error del ajuste, numero de condicion, rango efectivo). Diferencia relativa = ‖CPU - GPU‖ / ‖CPU‖.
Criterio (fijado antes de correr, por el investigador): < ~1e-9 en float64 -> la reubicacion es inocente; mayor -> error localizado en la reubicacion.
Variables de entorno (solo para probar sin GPU): REGRIDCHECK_TINY=1 (compara CPU con CPU), BUNDLE_DIR, OUT_DIR.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

TINY = os.environ.get("REGRIDCHECK_TINY") == "1"
bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("d1_state_f64.pt")).parent)
out = Path(os.environ.get("OUT_DIR", "/kaggle/working/regridcheck"))
out.mkdir(parents=True, exist_ok=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw = load(bundle / "sweep.py", "rc_sweep")
rg = load(bundle / "regrid.py", "rc_regrid")
pc = sw.load_campaign(bundle / "pikan_campaign.py")
prob = pc.make_problem("burgers", str(bundle / "burgers_coleHopf_1025x129.npz"))
saved = torch.load(bundle / "d1_state_f64.pt", map_location="cpu", weights_only=False)
dev_b = "cpu" if TINY else "cuda:0"


def run(device, dtype):
    model = prob.build("rbf", 48, 11, 1).to(dtype=dtype)
    model.load_state_dict({k: v.to(dtype) for k, v in saved["state_dict"].items()})
    model.to(device)
    xy = saved["points"].to(device=device, dtype=dtype)
    diag = rg.regrid_kan(model, xy, mix=1.0)
    with torch.no_grad():
        y = model(xy[:2048]).double().cpu()
    return dict(centers=[l.centers.detach().double().cpu() for l in model.layers[1:]], widths=[l.width.detach().double().cpu() for l in model.layers[1:]],
                coeffs=[l.coefficients.detach().double().cpu() for l in model.layers[1:]], output=y, diag=diag)


def rel(a, b):
    den = torch.linalg.norm(b)
    return float(torch.linalg.norm(a - b) / den) if den > 0 else float(torch.linalg.norm(a - b))


results = {}
for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
    cpu, gpu = run("cpu", dtype), run(dev_b, dtype)
    r = {}
    for i, layer in enumerate(("capa1", "capa2")):
        r[f"centros_{layer}"] = rel(gpu["centers"][i], cpu["centers"][i])
        r[f"anchos_{layer}"] = rel(gpu["widths"][i], cpu["widths"][i])
        r[f"coeficientes_{layer}"] = rel(gpu["coeffs"][i], cpu["coeffs"][i])
    r["salida_del_modelo"] = rel(gpu["output"], cpu["output"])
    r["max_abs_coef_capa1"] = float((gpu["coeffs"][0] - cpu["coeffs"][0]).abs().max())
    r["diag_fit_error_cpu_gpu"] = [[cpu["diag"][k]["fit_rel_error"], gpu["diag"][k]["fit_rel_error"]] for k in ("l1_fit", "l2_fit")]
    r["diag_cond_capa1_antes_cpu_gpu"] = [cpu["diag"]["l1_before"]["cond"], gpu["diag"]["l1_before"]["cond"]]
    r["diag_cond_capa1_despues_cpu_gpu"] = [cpu["diag"]["l1_after"]["cond"], gpu["diag"]["l1_after"]["cond"]]
    r["max_rel_diff"] = max(v for k, v in r.items() if isinstance(v, float) and k != "max_abs_coef_capa1")
    results[name] = r
verdict = dict(threshold=1e-9, float64_max_rel_diff=results["float64"]["max_rel_diff"], innocent=bool(results["float64"]["max_rel_diff"] < 1e-9),
               float32_max_rel_diff=results["float32"]["max_rel_diff"], device_b=dev_b, tiny=TINY)
summary = dict(verdict=verdict, results=results)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1), flush=True)
