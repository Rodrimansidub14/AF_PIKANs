"""EXP-0016: comprobacion en GPU del calentamiento (solo Burgers, un T4, ~10 min).

Antecedente (prueba previa, 2026-10-19): en Burgers, de tres controles identicos el PRIMERO del proceso difirio (estado a 35 % de distancia, error final
1.7e-3 frente a 3.8e-3) y los controles 2 y 3 y k = 1.00 del barrido fueron identicos bit a bit. Hipotesis: efecto de arranque en frio.

Esta prueba: en un proceso nuevo, `sweep.warmup` (corrida desechable con las mismas formas) y despues
  A = control (run_worker original), B = control identico, S = barrido corto (malla 0.5-1.00, con registro de la capa 0).
Si el calentamiento sirve, A debe ser identico a B y a S bit a bit, y ademas igual a los controles 2 y 3 de la prueba anterior (hash del estado; esa
ultima comparacion cruza sesiones de Kaggle y es informativa, no decisiva). Los hashes de la prueba anterior estan fijados abajo, antes de correr.

Criterio (antes de correr): el calentamiento "sirve" si A == B == S (hash de Adam-final y de L-BFGS-final). No puntua ninguna hipotesis.
Variables de entorno (solo para probar en CPU): WARMUP_TINY=1, BUNDLE_DIR, OUT_DIR.
"""
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

TINY = os.environ.get("WARMUP_TINY") == "1"
if TINY:
    torch.set_num_threads(1)

# Hashes (sweep.state_sha256) de los estados finales de la prueba previa en Burgers, 2026-10-19.
PRIOR = {
    "control1_cold": {"adam": "73f202a903395ff7184f233060998e46a0c7c7f74f4a436e1c35ddb5c3bb8128",
                      "lbfgs": "989bdfa4db67a6f5c5ee2f153fc420c1eb88a5c1a3aee3d76518fd52799cc4ec"},
    "control2_and_3_warm": {"adam": "626497dc94c1e681bd681f6c1302e3ce0ce8d3ef00e54b953a75effe64ba817b",
                            "lbfgs": "cee39a9a6c48572b5ee361ccd5c9fea67706f2b0f7a0acefc60702e294a33c9c"},
}


def main():
    bundle = Path(os.environ.get("BUNDLE_DIR") or next(Path("/kaggle/input").rglob("sweep.py")).parent)
    out = Path(os.environ.get("OUT_DIR", "/kaggle/working/warmup_check"))
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bundle))
    import sweep as sw

    pc = sw.load_campaign(bundle / "pikan_campaign.py")
    device = "cpu" if TINY else "cuda:0"
    info = dict(torch=torch.__version__, cuda=torch.version.cuda, device=device, tiny=TINY,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() and not TINY else None)
    print(json.dumps(info), flush=True)
    steps, max_iter = (40, 60) if TINY else (4000, 400)
    base = dict(problem="burgers", ref=str(bundle / "burgers_coleHopf_1025x129.npz"), arm="rbf", width=8 if TINY else 48, bases=11,
                dtype="float32", device=device, adam_steps=steps, adam_seconds_cap=0.0, lbfgs_seconds=300.0,
                lbfgs_chunk=10 if TINY else 50, lbfgs_max_iter=max_iter, n_col=256 if TINY else 0, seed=301)
    t0 = time.perf_counter()
    wu = sw.warmup(pc, Namespace(**base), steps=10 if TINY else 50)
    print("warmup", json.dumps(wu), flush=True)

    def ckpt_hash(d, kind):
        return sw.state_sha256(torch.load(d / f"checkpoint-{kind}-final.pt", map_location="cpu", weights_only=False)["state_dict"])

    runs = {}
    for name in ("A", "B"):
        d = out / f"control{name}"
        pc.run_worker(Namespace(**base, lbfgs_dtype="same", out=str(d)))
        rj = json.loads((d / "run.json").read_text())
        runs[name] = dict(adam=ckpt_hash(d, "adam"), lbfgs=ckpt_hash(d, "lbfgs"), adam_error=next(iter(rj["score_adam_final"].values())),
                          lbfgs_error=next(iter(rj["score_lbfgs_final"].values())), lbfgs_iterations=rj["lbfgs_iterations"],
                          adam_seconds=rj["adam_seconds"], lbfgs_seconds=rj["lbfgs_seconds"])
    sweep_args = Namespace(**{**base, "snapshot_every": 30.0, "grid": [0.5, 1.0], "order": None, "record_every": 10 if TINY else 500})
    (out / "sweep").mkdir(exist_ok=True)
    res, states, _ = sw.run_seed(pc, sweep_args, out_dir=out / "sweep")
    runs["S"] = dict(adam=sw.state_sha256(states["adam_final"]), lbfgs=sw.state_sha256(states[1.0]),
                     adam_error=next(iter(res["runs"]["1.0"]["adam_score_at_k"].values())),
                     lbfgs_error=next(iter(res["runs"]["1.0"]["final_score"].values())),
                     lbfgs_iterations=res["runs"]["1.0"]["lbfgs_iterations"], adam_seconds=res["adam_seconds"],
                     lbfgs_seconds=res["runs"]["1.0"]["lbfgs_seconds"])
    same = lambda a, b: runs[a]["adam"] == runs[b]["adam"] and runs[a]["lbfgs"] == runs[b]["lbfgs"]  # noqa: E731
    verdict = dict(A_equals_B=same("A", "B"), A_equals_S=same("A", "S"), B_equals_S=same("B", "S"),
                   A_equals_prior_warm=(runs["A"]["adam"] == PRIOR["control2_and_3_warm"]["adam"]
                                        and runs["A"]["lbfgs"] == PRIOR["control2_and_3_warm"]["lbfgs"]),
                   A_equals_prior_cold=(runs["A"]["adam"] == PRIOR["control1_cold"]["adam"]))
    verdict["warmup_works"] = bool(verdict["A_equals_B"] and verdict["A_equals_S"])
    summary = dict(info=info, warmup=wu, runs=runs, verdict=verdict, wall_seconds=time.perf_counter() - t0)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(verdict), flush=True)
    print(json.dumps({k: (v["adam_error"], v["lbfgs_error"]) for k, v in runs.items()}), flush=True)


if __name__ == "__main__":
    main()
