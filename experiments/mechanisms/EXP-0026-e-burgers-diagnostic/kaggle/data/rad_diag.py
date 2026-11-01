"""EXP-0026 (v1.1-D), solo lectura: que hace `RadLever` (aR) con el pool en Burgers. NO modifica aR ni rad.py. Entrena B (Adam 2000, convencion T1) y aplica la palanca una vez al modelo
resultante; registra puntos unicos del pool remuestreado, `effective_candidates`, la fraccion de puntos en la banda del choque (|x| < 0.1) frente a la del pool uniforme de entrenamiento
y la fraccion de masa de probabilidad en el 5 % de candidatos de mayor residuo. Separa «muestreo con reemplazo» de «agudeza del peso».
Uso: python rad_diag.py --seeds 3001,3002,3003 --out archivo.json"""
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
ROOT = HERE.parents[2]
sys.path.insert(0, str(BUNDLE or ROOT / "experiments/mechanisms/EXP-0023-gram-preconditioning"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ob = _load((BUNDLE / "observer_v0.py") if BUNDLE else ROOT / "src/mechanisms/observer_v0.py", "observer_v0_raddiag")
import pre_trainer as pt  # noqa: E402
pc, sw = pt.pc, pt.sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3001,3002,3003")
    ap.add_argument("--adam", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    rows = []
    for seed in map(int, a.seeds.split(",")):
        cfg = pt.PROBLEMS["burgers"]
        prob = pc.make_problem(cfg["name"], str(cfg["ref"]))
        model = prob.build("rbf", cfg["width"], 11, seed).to(device=dev, dtype=torch.float32)
        P = prob.pools(seed, dev, torch.float32)
        pt.adam_with_pre(prob, model, P, a.adam, None)
        pool0 = P["residual"].detach().cpu().numpy().copy()
        lever = ob.RadLever(prob.lo, prob.hi, 8192)
        rng = np.random.default_rng(seed)
        # misma construccion que RadLever.apply, pero guardando los candidatos y las probabilidades para medir la masa
        cand = np.asarray(lever.lo) + rng.random((lever.n_candidates, len(lever.lo))) * (np.asarray(lever.hi) - np.asarray(lever.lo))
        r = ob.residual_at(prob, model, torch.as_tensor(cand, dtype=P["residual"].dtype, device=dev))
        s = ob._rad.sample_rad(cand, r, int(pool0.shape[0]), rng, power=lever.power, uniform_fraction=lever.uniform_floor)
        p = s["probabilities"]
        top = np.argsort(np.abs(r))[::-1][: max(1, int(0.05 * len(r)))]
        band = lambda X: float((np.abs(X[:, 0]) < 0.1).mean())
        rows.append(dict(seed=seed, pool_size=int(pool0.shape[0]), n_candidates=int(len(cand)), unique_points=s["unique_count"], unique_fraction_of_pool=s["unique_count"] / pool0.shape[0],
                         effective_candidates=s["effective_candidates"], mass_top5pct_candidates=float(p[top].sum()), band_fraction_resampled=band(s["points"]), band_fraction_training_pool=band(pool0),
                         band_fraction_uniform_candidates=band(cand), residual_max_over_median=float(np.abs(r).max() / max(np.median(np.abs(r)), 1e-300))))
        print(rows[-1], flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
