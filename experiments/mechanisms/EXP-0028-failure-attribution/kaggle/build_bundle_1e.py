"""Arma el bundle de Kaggle de la Fase 1e del EXP-0028 (Enmienda v1.4). NO sube ni lanza nada, NO toca los archivos ya subidos (dataset*/, kernel*/, approval*.json, SHA256SUMS*.txt). Escribe:
  kaggle/dataset_1e/        dataset privado NUEVO `rodrimandub/af-pikans-exp0028-bundle-1e` (union de los bundles 1 / 1c / 1c-dfg / 1d + run_phase1e.py; todo plano, BUNDLE_DIR)
  kaggle/kernel_1e_{ns1,ns2,dfg1,dfg2,dfg3,acf,bu1,bu2,bu3}/   kernels `rodrimandub/af-pikans-exp-0028-phase1e-{ns1,...}`
  kaggle/approval_1e.json, SHA256SUMS_1e.txt
Verificaciones (abortan si fallan): (1) todo archivo que el bundle 1e comparte por nombre con `dataset_1d/` es IDENTICO byte a byte (asi el B de 1d, que usa esos archivos, es el mismo codigo que usa 1e);
(2) igual con `dataset_1c/` y `dataset_1c_dfg/` para sus archivos; (3) gram_preconditioning.py y dfg.py coinciden con src/; residual_terms.py es la copia exacta subida con 1c (src/ ya difiere solo en piezo).
Uso: python kaggle/build_bundle_1e.py"""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
ROOT = HERE.parents[3]
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0028-bundle-1e"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)             # solo para reutilizar SOURCES (no ejecutan main)
    return mod


_b1, _b1c, _b1cd = _load("bb_phase1", "build_bundle.py"), _load("bb_phase1c", "build_bundle_1c.py"), _load("bb_phase1c_dfg", "build_bundle_1c_dfg.py")
SOURCES = {}
for _mod in (_b1, _b1c, _b1cd):
    for _n, _s in _mod.SOURCES.items():
        assert _n not in SOURCES or Path(SOURCES[_n]).read_bytes() == Path(_s).read_bytes(), f"{_n}: fuentes distintas entre bundles"
        SOURCES[_n] = _s
# residual_terms.py: copia EXACTA de la que se subio con 1c / 1c-dfg (probada en GPU). El de src/ ya difiere de esa solo en piezo_residual_terms y en _ADAPTERS (trabajo de piezo en curso);
# NS y DFG no cambian, pero el bundle de confirmacion no debe depender de un archivo que sigue moviendose.
SOURCES["residual_terms.py"] = HERE / "dataset_1c" / "residual_terms.py"
SOURCES["run_phase1d.py"] = EXP28 / "run_phase1d.py"
SOURCES["run_phase1e.py"] = EXP28 / "run_phase1e.py"
# bloque -> (slug, titulo, minutos [esperado, peor caso]) segun budget_1e.md
KERNELS = {
    "kernel_1e_ns1": dict(slug="af-pikans-exp-0028-phase1e-ns1", block="ns1", minutes=[170, 200]),
    "kernel_1e_ns2": dict(slug="af-pikans-exp-0028-phase1e-ns2", block="ns2", minutes=[111, 130]),
    "kernel_1e_dfg1": dict(slug="af-pikans-exp-0028-phase1e-dfg1", block="dfg1", minutes=[194, 225]),
    "kernel_1e_dfg2": dict(slug="af-pikans-exp-0028-phase1e-dfg2", block="dfg2", minutes=[194, 225]),
    "kernel_1e_dfg3": dict(slug="af-pikans-exp-0028-phase1e-dfg3", block="dfg3", minutes=[116, 135]),
    "kernel_1e_acf": dict(slug="af-pikans-exp-0028-phase1e-acf", block="acf", minutes=[167, 195]),
    "kernel_1e_bu1": dict(slug="af-pikans-exp-0028-phase1e-bu1", block="bu1", minutes=[128, 140]),
    "kernel_1e_bu2": dict(slug="af-pikans-exp-0028-phase1e-bu2", block="bu2", minutes=[128, 140]),
    "kernel_1e_bu3": dict(slug="af-pikans-exp-0028-phase1e-bu3", block="bu3", minutes=[64, 70]),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ds = HERE / "dataset_1e"
    ds.mkdir(exist_ok=True)
    for n, s in SOURCES.items():
        shutil.copyfile(s, ds / n)
    assert (ds / "gram_preconditioning.py").read_bytes() == (ROOT / "src/mechanisms/gram_preconditioning.py").read_bytes()
    assert (ds / "residual_terms.py").read_bytes() == (HERE / "dataset_1c_dfg" / "residual_terms.py").read_bytes()
    assert (ds / "dfg.py").read_bytes() == (ROOT / "src/research_pipeline/sciml/dfg.py").read_bytes()
    shared = {}
    for prev in ("dataset_1d", "dataset_1c", "dataset_1c_dfg"):
        pd = HERE / prev
        if not pd.exists():
            continue
        for p in sorted(pd.iterdir()):
            if p.name == "dataset-metadata.json":
                continue
            assert (ds / p.name).exists(), f"{prev}/{p.name} no esta en dataset_1e"
            if p.name in ("run_phase1d.py", "run_phase1c.py", "run_phase1c_dfg.py"):
                assert p.read_bytes() == (ds / p.name).read_bytes(), f"{prev}/{p.name} difiere del ejecutor actual (los ejecutores ya subidos no deben cambiar)"
            else:
                assert p.read_bytes() == (ds / p.name).read_bytes(), f"{prev}/{p.name} difiere: el B de ese bundle no seria el mismo codigo"
            shared.setdefault(prev, []).append(p.name)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0028 bundle 1e", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    comps = {f"dataset_1e/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    template = (EXP28 / "kaggle_block_1e.py").read_text(encoding="utf-8")
    comps["kaggle_block_1e.py"] = sha(EXP28 / "kaggle_block_1e.py")
    appr = {"bundle": "exp0028-phase1e", "dataset": DATASET, "launch_status": "prepared_not_uploaded", "uploaded": False, "executed": False,
            "approved_by": "NINGUNA: preparado sin lanzar (Enmienda v1.4); la subida/ejecucion queda pendiente de la orden del investigador",
            "recipe": "NS2d-CG: run_phase1c.py --recipe full (Adam 20 000 f32 + L-BFGS 500 it.), B,F,J,E,I x 4101-4105; DFG: Adam 20 000 minibatch f32 + L-BFGS 4 000 it. (sweep.lbfgs_stage), B,F,J,E,I x 4101-4105; "
                      "ac/kpp/burgers: receta de run_phase1d.py, brazos J,I x 4101-4105 (B = el de 1d); ver budget_1e.md",
            "caps": "sin trabajo nuevo despues de 3.4 h; kill a 3.9 h", "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "scientific_accepted": False,
            "files_identical_to_previous_bundles": shared, "kernels": {}}
    for sub, k in KERNELS.items():
        kd = HERE / sub
        kd.mkdir(exist_ok=True)
        code = f"exp0028_phase1e_{k['block']}_gpu.py"
        for old in kd.glob("exp0028_phase1e_*_gpu.py"):
            if old.name != code:
                old.unlink()
        (kd / code).write_text(template.replace('BLOCK = "ns1"   # @@BLOCK@@', f'BLOCK = "{k["block"]}"   # fijado por build_bundle_1e.py'), encoding="utf-8")
        (kd / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{USER}/{k['slug']}", "title": f"AF PIKANs EXP-0028 phase1e {k['block']}", "code_file": code, "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
             "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
        comps[f"{sub}/{code}"], comps[f"{sub}/kernel-metadata.json"] = sha(kd / code), sha(kd / "kernel-metadata.json")
        appr["kernels"][sub] = dict(slug=f"{USER}/{k['slug']}", block=k["block"], session_estimate_minutes=k["minutes"])
    appr["component_sha256"] = comps
    (HERE / "approval_1e.json").write_text(json.dumps(appr, indent=2))
    (HERE / "SHA256SUMS_1e.txt").write_text("".join(f"{h}  {n}\n" for n, h in comps.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir()), "kernels": list(KERNELS),
                      "identical_to_previous": {k: len(v) for k, v in shared.items()}}))


if __name__ == "__main__":
    main()
