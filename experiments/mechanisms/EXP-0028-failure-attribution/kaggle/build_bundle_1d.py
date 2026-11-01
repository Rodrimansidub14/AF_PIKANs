"""Arma el bundle de Kaggle de la Fase 1d del EXP-0028 (patron: build_bundle.py de la fase 1). NO sube ni lanza nada, NO toca los archivos ya subidos (dataset/, kernel*/, approval.json, SHA256SUMS.txt). Escribe:
  kaggle/dataset_1d/        dataset privado `rodrimandub/af-pikans-exp0028-bundle-1d` (codigo de la fase 1 + run_phase1d.py + referencias; todo plano, BUNDLE_DIR)
  kaggle/kernel_1d_{ac,kpp,bu1,bu2,bu3}/   kernels `rodrimandub/af-pikans-exp-0028-phase1d-{ac,kpp,bu1,bu2,bu3}`
  kaggle/approval_1d.json, SHA256SUMS_1d.txt
Uso: python kaggle/build_bundle_1d.py"""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0028-bundle-1d"

_spec = importlib.util.spec_from_file_location("build_bundle_phase1", str(HERE / "build_bundle.py"))
_bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bb)                      # solo para reutilizar SOURCES (no ejecuta main)
SOURCES = dict(_bb.SOURCES)
SOURCES["run_phase1d.py"] = EXP28 / "run_phase1d.py"
KERNELS = {
    "kernel_1d_ac": dict(slug="af-pikans-exp-0028-phase1d-ac", title="AF PIKANs EXP-0028 phase1d ac", block="ac", minutes=[150, 190]),
    "kernel_1d_kpp": dict(slug="af-pikans-exp-0028-phase1d-kpp", title="AF PIKANs EXP-0028 phase1d kpp", block="kpp", minutes=[120, 150]),
    "kernel_1d_bu1": dict(slug="af-pikans-exp-0028-phase1d-bu1", title="AF PIKANs EXP-0028 phase1d bu1", block="bu1", minutes=[190, 205]),
    "kernel_1d_bu2": dict(slug="af-pikans-exp-0028-phase1d-bu2", title="AF PIKANs EXP-0028 phase1d bu2", block="bu2", minutes=[190, 205]),
    "kernel_1d_bu3": dict(slug="af-pikans-exp-0028-phase1d-bu3", title="AF PIKANs EXP-0028 phase1d bu3", block="bu3", minutes=[125, 135]),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ds = HERE / "dataset_1d"
    ds.mkdir(exist_ok=True)
    for n, s in SOURCES.items():
        shutil.copyfile(s, ds / n)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0028 bundle 1d", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    comps = {f"dataset_1d/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    template = (EXP28 / "kaggle_block_1d.py").read_text(encoding="utf-8")
    comps["kaggle_block_1d.py"] = sha(EXP28 / "kaggle_block_1d.py")
    appr = {"bundle": "exp0028-phase1d", "dataset": DATASET, "launch_status": "prepared_not_uploaded", "uploaded": False, "executed": False,
            "approved_by": "NINGUNA: preparado sin lanzar (Enmienda v1.2); la subida/ejecucion queda pendiente de la orden del investigador",
            "recipe": "receta COMPLETA de la linea base: ac Adam 40000 f32 + L-BFGS f32 tope 300 s; burgers 16384 pts, Adam 24000 f32 + L-BFGS f64 tope 1800 s; kpp = receta de ac (heredada, sin linea base); "
                      "brazos B,E,F,C (ac), B,F,E (burgers), B,E,F (kpp); semillas 4101-4105; ver budget_1d.md",
            "caps": "sin trabajo nuevo despues de 3.4 h; kill a 3.9 h", "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "scientific_accepted": False, "kernels": {}}
    for sub, k in KERNELS.items():
        kd = HERE / sub
        kd.mkdir(exist_ok=True)
        code = f"exp0028_phase1d_{k['block']}_gpu.py"
        for old in kd.glob("exp0028_phase1d_*_gpu.py"):
            if old.name != code:
                old.unlink()
        (kd / code).write_text(template.replace('BLOCK = "ac"   # @@BLOCK@@', f'BLOCK = "{k["block"]}"   # fijado por build_bundle_1d.py'), encoding="utf-8")
        (kd / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{USER}/{k['slug']}", "title": k["title"], "code_file": code, "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
             "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
        comps[f"{sub}/{code}"], comps[f"{sub}/kernel-metadata.json"] = sha(kd / code), sha(kd / "kernel-metadata.json")
        appr["kernels"][sub] = dict(slug=f"{USER}/{k['slug']}", block=k["block"], session_estimate_minutes=k["minutes"])
    appr["component_sha256"] = comps
    (HERE / "approval_1d.json").write_text(json.dumps(appr, indent=2))
    (HERE / "SHA256SUMS_1d.txt").write_text("".join(f"{h}  {n}\n" for n, h in comps.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir()), "kernels": list(KERNELS)}))


if __name__ == "__main__":
    main()
