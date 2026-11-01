"""EXP-0024, diagnostico (CPU): ¿la reubicacion D1 amplifica diferencias de redondeo? Burgers float32, semilla 1, receta reducida (Adam 200).

Se entrena 50 pasos (25 % de 200), se copia el modelo y a la copia B se le perturban los coeficientes de la capa 0 con ruido relativo `eps` (el orden de la diferencia entre CPU y GPU
en float32 es ~1e-7). Luego A y B reciben EXACTAMENTE el mismo tratamiento: (i) reubicacion D1 y 150 pasos de Adam; (ii) control: sin reubicacion y 150 pasos de Adam.
Se reporta la diferencia relativa entre A y B de los coeficientes tras reubicar, y del error final (puntuado despues). Si D1 amplifica y el control no, la divergencia CPU-GPU de D1
es sensibilidad numerica de la reubicacion y no un error de codigo. Uso: python diag_sensitivity.py [eps]
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]] + sys.argv[1:2]
eps = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-7
spec = importlib.util.spec_from_file_location("t1_trainer_diag", str(HERE / "t1_trainer.py"))
t1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t1)
torch.set_num_threads(2)

cfg = t1.PROBLEMS["burgers"]
prob = t1.pc.make_problem("burgers", str(cfg["ref"]))
dev, dt = torch.device("cpu"), torch.float32
base = prob.build("rbf", cfg["width"], 11, 1).to(dtype=dt)
P = prob.pools(1, dev, dt)
t1.adam_with_hook(prob, base, P, 50)
print("preentrenado 50 pasos", flush=True)


def clone_perturbed(model, e):
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in m.layers[0].parameters():
            p.mul_(1 + e * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return m


def finish(model, regrid):
    d = None
    if regrid:
        d = t1.rg.regrid_kan(model, P["residual"].detach(), mix=1.0)
    coef = [l.coefficients.detach().clone() for l in model.layers]
    t1.adam_with_hook(prob, model, P, 150)
    st = t1.sw.clone_state(model)
    return coef, t1.sw.primary(t1.sw.score_run(prob, model, dev, dt, st))


out = {"eps": eps}
for name, regrid in (("D1", True), ("CTRL", False)):
    a, b = copy.deepcopy(base), clone_perturbed(base, eps)
    ca, ea = finish(a, regrid)
    cb, eb = finish(b, regrid)
    coef_diff = [float(torch.linalg.norm(x - y) / torch.linalg.norm(x)) for x, y in zip(ca, cb)]
    out[name] = dict(error_A=ea, error_B=eb, rel_diff_error=abs(ea - eb) / ea, coef_rel_diff_layers_after_regrid=coef_diff)
    print(name, json.dumps(out[name]), flush=True)
(HERE / "results/diag_sensitivity.json").write_text(json.dumps(out, indent=1))
