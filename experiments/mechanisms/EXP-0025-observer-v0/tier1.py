"""EXP-0025, prueba 9 (Tier 1, CPU): receta real (Adam 4000, ancho 16, pool 8000+400+800 Sobol), K=500, n_p=25, H_d fresco (8192) por decision.
Allen-Cahn d=1e-3 (problema primario del spec). Mide frecuencia de acciones, inercia y reloj de la etapa Adam: B (Adam plano) frente a O (probe).
NO mide error contra palancas fijas (lo responde el bloque) y NO corre L-BFGS (identico en B y O; la razon de la etapa Adam acota por encima la del reloj total).
Semillas 2901/2902: fuera de T1=2001-2005 y V=2101-2105, para no tocar las semillas del bloque. Uso: python tier1.py --seed 2901 --threads 2 --out out.json"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import torch

import os
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None     # Kaggle: carpeta plana con todo
ROOT = Path(__file__).resolve().parents[3] if BUNDLE is None else BUNDLE
EXP23 = ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning" if BUNDLE is None else BUNDLE
sys.path.insert(0, str(EXP23))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ob = load((BUNDLE / "observer_v0.py") if BUNDLE else ROOT / "src/mechanisms/observer_v0.py", "observer_v0_t1")
import pre_trainer as pt  # noqa: E402
pc, gp = pt.pc, pt.gp
MAT = (BUNDLE / "Allen_Cahn.mat") if BUNDLE else EXP23 / "kaggle/dataset/Allen_Cahn.mat"


def fresh(seed):
    prob = pc.make_problem("ac", str(MAT))
    model = prob.build("rbf", 16, 11, seed).to(torch.float32)
    return prob, model, prob.pools(seed, torch.device("cpu"), torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2901)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--b-seconds", type=float, default=None, help="reloj de B ya medido (la etapa B de la corrida anterior termino y se perdio la O); mismas condiciones: 2 procesos x 3 hilos")
    ap.add_argument("--b-final-loss", type=float, default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = dict(problem="allen_cahn d=1e-3", seed=a.seed, steps=a.steps, K=500, n_p=25, threads=a.threads, dtype="float32")
    prob, model, P = fresh(a.seed)
    npool = {k: tuple(v.shape) for k, v in P.items()}
    res["pool_shapes"] = npool
    if a.b_seconds is None:
        A = pt.adam_with_pre(prob, model, P, a.steps, None)
        res["B"] = dict(seconds=A["seconds"], status=A["status"], final_loss=A["hist"][-1]["loss"] if A["hist"] else None)
    else:
        res["B"] = dict(seconds=a.b_seconds, status="adam_complete", final_loss=a.b_final_loss, reused_from_previous_session=True)
    print("B done", res["B"], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")      # guardado parcial: B
    prob, model, P = fresh(a.seed)
    lo, hi = prob.lo, prob.hi
    pre = gp.GramPreconditioner("E", n_points=2048)
    pre.attach(model)
    cfg = ob.ObserverConfig(K=500, n_p=25)
    obs = ob.Observer([ob.RadLever(lo, hi, 8192), ob.GramLever(pre, "E")], lo, hi, ob.FreshHeldout(lo, hi, 8192, a.seed), a.seed, cfg, "probe")
    o = obs.run_adam(prob, model, P, a.steps)
    res["O"] = dict(seconds=o["seconds"], status=o["status"], extra_steps=o["extra_steps"], probe_seconds=o["probe_seconds"], inert=o["inert"],
                    final_loss=o["hist"][-1]["loss"] if o["hist"] else None, chosen=[d["chosen"] for d in o["decisions"]],
                    improvement=[d["improvement"] for d in o["decisions"]], worse_cells=[d["worse_cells"] for d in o["decisions"]],
                    null_reason=[d["null_reason"] for d in o["decisions"]], h_block=[d["h_block"] for d in o["decisions"]])
    res["clock_ratio_O_over_B"] = res["O"]["seconds"] / res["B"]["seconds"]
    res["extra_steps_ratio"] = res["O"]["extra_steps"] / a.steps
    res["spec_limit"] = "reloj O <= 1.5x B"
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
