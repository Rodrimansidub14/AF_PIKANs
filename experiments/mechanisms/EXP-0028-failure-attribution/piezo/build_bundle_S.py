"""Arma el bundle de las pruebas previas del brazo S (EXP-0029) para Kaggle: dataset `rodrimandub/af-pikans-exp0029-tests` (codigo + estados s004000 de B recortados + referencias FEM) y kernel
`rodrimandub/af-pikans-exp-0029-tests` (kaggle_tests_S.py). NO sube ni lanza nada."""
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
ROOT = HERE.parents[3]
KAG = EXP28 / "kaggle"
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0029-tests"
SOURCES = {n: HERE / n for n in ("run_piezo.py", "artifacts.py", "piezo_problem.py")}
SOURCES.update({n: ROOT / "src/mechanisms" / n for n in ("gram_preconditioning.py", "residual_terms.py")})
SOURCES.update({f"fem_{p}.npz": HERE / "reference" / f"fem_{p}.npz" for p in ("converse", "direct")})
SOURCES.update({f"fem_flux_{p}.npz": HERE / "reference" / f"fem_flux_{p}.npz" for p in ("converse", "direct")})
SEEDS = (20260728, 20260729, 20260730)


def find(problem, seed):
    hits = sorted((EXP28 / "results").rglob(f"piezo-{problem}-B-seed{seed}-artifacts.npz"))
    assert hits, (problem, seed)
    return hits[0]


def main():
    ds = KAG / "dataset_exp0029"
    (ds / "states").mkdir(parents=True, exist_ok=True)
    for n, s in SOURCES.items():
        shutil.copyfile(s, ds / n)
    for p in ("converse", "direct"):
        for sd in SEEDS:
            z = np.load(find(p, sd), allow_pickle=True)
            keep = {k: z[k] for k in z.keys() if k.startswith("s004000/state/")}
            assert keep, (p, sd)
            np.savez(ds / "states" / f"piezo-{p}-B-seed{sd}-artifacts.npz", **keep)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0029 tests", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    kd = KAG / "kernel_exp0029"
    kd.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "kaggle_tests_S.py", kd / "exp0029_tests_gpu.py")
    (kd / "kernel-metadata.json").write_text(json.dumps(
        {"id": f"{USER}/af-pikans-exp-0029-tests", "title": "AF PIKANs EXP-0029 tests", "code_file": "exp0029_tests_gpu.py", "language": "python", "kernel_type": "script", "is_private": "true",
         "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
    print(json.dumps({"files": len(list(ds.rglob('*'))), "bytes": sum(p.stat().st_size for p in ds.rglob('*') if p.is_file())}))


if __name__ == "__main__":
    main()
