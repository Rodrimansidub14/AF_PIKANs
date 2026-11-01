"""EXP-0016, prueba previa en GPU (antes del bloque grande de la etapa A). Un solo T4, ~20-25 min.

Pregunta de la prueba (ii-GPU del spec): en GPU las reducciones atomicas pueden romper la igualdad bit a bit, asi que k = 1.00 del barrido
puede no reproducir el control exactamente. Esta prueba mide el RUIDO DE GPU (tres controles identicos entre si) y comprueba si k = 1.00 del
barrido cae dentro de ese ruido. Si no cae, la etapa A compara contra la replica del control DENTRO del barrido (ya es el comparador pareado).

Por problema (Burgers RBF48 y Allen-Cahn RBF16, f32, semilla 301/601, receta recortada: 4 000 pasos de Adam, L-BFGS limitado a 400 iteraciones):
  1. tres controles = `run_worker` original de la campana (pikan_campaign.py), mismas semilla y receta;
  2. un barrido corto con `sweep.py` (malla 0.25-1.00, registro de la capa 0 cada 500 pasos).
Se comparan los estados (distancia L2 relativa y maxima absoluta) y los errores finales. Tambien se miden: segundos por paso de Adam, sobrecosto del
registro y duracion de L-BFGS desde cada punto de la malla (la que el presupuesto del spec solo supone).

Criterio (fijado aqui, antes de correr): con ruido = maxima distancia relativa entre pares de controles y desvio = maxima distancia relativa
entre k = 1.00 del barrido y cada control, el barrido queda "dentro del ruido" si desvio <= 2 * ruido (si el ruido es 0, el desvio debe ser 0).
Esto no es un criterio cientifico: no puntua ninguna hipotesis.

Variables de entorno (solo para probar en CPU): PRECHECK_TINY=1 (red y receta minimas), BUNDLE_DIR, OUT_DIR.
"""
import json
import math
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

TINY = os.environ.get("PRECHECK_TINY") == "1"
if TINY:
    torch.set_num_threads(1)  # CPU determinista para la prueba de humo


# --------------------------------------------------------------------------- comparacion (pura, con pruebas en tests/test_precheck_gpu.py)
def flat(state):
    return np.concatenate([np.asarray(state[k].detach().cpu().double().numpy()).reshape(-1) for k in sorted(state)])


def rel_dist(a, b):
    """Distancia L2 relativa y maxima absoluta entre dos estados (dicts de tensores)."""
    fa, fb = flat(a), flat(b)
    den = np.linalg.norm(fb)
    return float(np.linalg.norm(fa - fb) / den) if den > 0 else float(np.linalg.norm(fa - fb)), float(np.abs(fa - fb).max())


def decide(noise, deviation):
    """Dentro del ruido si desvio <= 2 * ruido; con ruido 0 el desvio debe ser 0."""
    return bool(deviation <= 2.0 * noise)


def compare(controls, sweep_state):
    """controls: lista de estados; devuelve ruido (max entre pares), desvio (max del barrido a cada control) y la decision."""
    pair = [rel_dist(controls[i], controls[j])[0] for i in range(len(controls)) for j in range(i + 1, len(controls))]
    dev = [rel_dist(sweep_state, c)[0] for c in controls]
    noise, deviation = (max(pair) if pair else 0.0), max(dev)
    return dict(noise=noise, deviation=deviation, within_noise=decide(noise, deviation), pair=pair, dev=dev,
                max_abs_noise=max([rel_dist(controls[i], controls[j])[1] for i in range(len(controls)) for j in range(i + 1, len(controls))] or [0.0]))


def compare_scalar(values, sweep_value):
    """Lo mismo para un escalar (error final): ruido = maxima diferencia relativa entre controles."""
    pair = [abs(values[i] - values[j]) / abs(values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    dev = [abs(sweep_value - v) / abs(v) for v in values]
    noise, deviation = (max(pair) if pair else 0.0), max(dev)
    return dict(noise=noise, deviation=deviation, within_noise=decide(noise, deviation))


# --------------------------------------------------------------------------- ejecucion
def main():
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("sweep.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/precheck"))
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bundle))
    import sweep as sw

    pc = sw.load_campaign(bundle / "pikan_campaign.py")
    device = "cpu" if TINY else "cuda:0"
    info = dict(torch=torch.__version__, cuda=torch.version.cuda, device=device, tiny=TINY,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() and not TINY else None,
                cudnn_deterministic=torch.backends.cudnn.deterministic, deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    print(json.dumps(info), flush=True)
    problems = {
        "burgers": dict(ref=bundle / "burgers_coleHopf_1025x129.npz", width=8 if TINY else 48, seed=301),
        "ac": dict(ref=bundle / "Allen_Cahn.mat", width=8 if TINY else 16, seed=601),
    }
    steps = 40 if TINY else 4000
    max_iter = 60 if TINY else 400
    n_col = 256 if TINY else 0
    summary = dict(info=info, recipe=dict(adam_steps=steps, lbfgs_max_iter=max_iter, lbfgs_seconds_cap=300.0), problems={})
    for name, c in problems.items():
        t_problem = time.perf_counter()
        base = dict(problem=name, ref=str(c["ref"]), arm="rbf", width=c["width"], bases=11, dtype="float32", device=device, adam_steps=steps,
                    adam_seconds_cap=0.0, lbfgs_seconds=300.0, lbfgs_chunk=50 if not TINY else 10, lbfgs_max_iter=max_iter, n_col=n_col,
                    seed=c["seed"])
        if name == "ac" and TINY:
            base["n_col"] = 0
        controls, runjson = [], []
        for i in range(3):
            d = out / f"{name}-control{i + 1}"
            pc.run_worker(Namespace(**base, lbfgs_dtype="same", out=str(d)))
            controls.append(dict(adam=torch.load(d / "checkpoint-adam-final.pt", map_location="cpu", weights_only=False)["state_dict"],
                                 lbfgs=torch.load(d / "checkpoint-lbfgs-final.pt", map_location="cpu", weights_only=False)["state_dict"]))
            runjson.append(json.loads((d / "run.json").read_text()))
        sweep_args = Namespace(**{**base, "snapshot_every": 30.0, "grid": list(sw.GRID), "order": None, "record_every": 10 if TINY else 500})
        (out / f"{name}-sweep").mkdir(exist_ok=True)
        res, states, trace = sw.run_seed(pc, sweep_args, out_dir=out / f"{name}-sweep")
        key = lambda sc: next(iter(sc.values()))  # noqa: E731 - primera clave = L2 relativo contra la referencia
        r1 = res["runs"]["1.0"]
        cmp_ = dict(
            adam_state=compare([c["adam"] for c in controls], states["adam_final"]),
            lbfgs_state=compare([c["lbfgs"] for c in controls], states[1.0]),
            adam_error=compare_scalar([key(r["score_adam_final"]) for r in runjson], key(r1["adam_score_at_k"])),
            lbfgs_error=compare_scalar([key(r["score_lbfgs_final"]) for r in runjson], key(r1["final_score"])))
        timing = dict(
            sec_per_adam_step_controls=[r["adam_seconds"] / steps for r in runjson], sec_per_adam_step_sweep=res["adam_seconds"] / steps,
            record_seconds=res["record_seconds"], record_overhead_fraction=res["record_seconds"] / max(res["adam_seconds"], 1e-9),
            control_lbfgs_seconds=[r["lbfgs_seconds"] for r in runjson], control_lbfgs_iterations=[r["lbfgs_iterations"] for r in runjson],
            control_lbfgs_status=[r["lbfgs_status"] for r in runjson],
            sweep_lbfgs={f: dict(k=v["k"], seconds=v["lbfgs_seconds"], iterations=v["lbfgs_iterations"], status=v["lbfgs_status"],
                                 cap=v["cap"], final_error=key(v["final_score"]), error_at_handoff=key(v["adam_score_at_k"]))
                         for f, v in res["runs"].items()})
        summary["problems"][name] = dict(comparison=cmp_, timing=timing, trigger_loss=res.get("trigger"),
                                         trigger_settling=res.get("trigger_settling"), wall_seconds=time.perf_counter() - t_problem,
                                         trace_shapes={k: list(v.shape) for k, v in (trace or {}).items() if hasattr(v, "shape")})
        (out / f"{name}-sweep.json").write_text(json.dumps(res))
        print(name, json.dumps({k: (v["within_noise"], v["noise"], v["deviation"]) for k, v in cmp_.items()}), flush=True)
        print(name, "timing", json.dumps(timing["sweep_lbfgs"]), flush=True)
    summary["all_within_noise"] = all(v["within_noise"] for p in summary["problems"].values() for v in p["comparison"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("all_within_noise:", summary["all_within_noise"], flush=True)


if __name__ == "__main__":
    main()
