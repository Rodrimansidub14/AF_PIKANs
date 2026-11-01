"""Arma el bundle de Kaggle de la REBANADA con artefactos del EXP-0028 (Enmienda v1.6). NO sube ni lanza nada, NO toca lo ya subido (dataset*/, kernel_1*/, kernel_a|b|piezo, approval*.json, SHA256SUMS*.txt). Escribe:
  kaggle/dataset_slice/        dataset privado NUEVO `rodrimandub/af-pikans-exp0028-bundle-slice` (copia plana de dataset_1e/ + run_slice.py + artifacts_core.py; BUNDLE_DIR)
  kaggle/kernel_slice_{ns,dfg,ackpp,bu}/   kernels `rodrimandub/af-pikans-exp-0028-slice-{ns,dfg,ackpp,bu}`
  kaggle/approval_slice.json, SHA256SUMS_slice.txt
Verificaciones (abortan si fallan): todo archivo heredado de dataset_1e/ es identico byte a byte al de dataset_1e/ (y estos a los de 1d / 1c / 1c-dfg: ya verificado por build_bundle_1e.py);
gram_preconditioning.py y dfg.py coinciden con src/; run_slice.py y artifacts_core.py no contienen la clave ni leen WANDB_API_KEY ni .env.
Uso: python kaggle/build_bundle_slice.py"""
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
ROOT = HERE.parents[3]
SLICE = EXP28 / "artifact_slice"
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0028-bundle-slice"
NEW_FILES = ("run_slice.py", "artifacts_core.py")
SKIP = ("dataset-metadata.json", "run_phase1e.py")            # 1e-only
KERNELS = {   # sesion esperada / peor caso en minutos (budget_slice.md)
    "kernel_slice_ns": dict(slug="af-pikans-exp-0028-slice-ns", block="ns", minutes=[64, 90]),
    "kernel_slice_dfg": dict(slug="af-pikans-exp-0028-slice-dfg", block="dfg", minutes=[63, 90]),
    "kernel_slice_ackpp": dict(slug="af-pikans-exp-0028-slice-ackpp", block="ackpp", minutes=[46, 70]),
    "kernel_slice_bu": dict(slug="af-pikans-exp-0028-slice-bu", block="bu", minutes=[81, 120]),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base = HERE / "dataset_1e"
    assert base.exists(), "falta kaggle/dataset_1e (se construye con build_bundle_1e.py)"
    ds = HERE / "dataset_slice"
    ds.mkdir(exist_ok=True)
    for p in sorted(base.iterdir()):
        if p.name in SKIP:
            continue
        shutil.copyfile(p, ds / p.name)
    for n in NEW_FILES:
        shutil.copyfile(SLICE / n, ds / n)
    assert (ds / "gram_preconditioning.py").read_bytes() == (ROOT / "src/mechanisms/gram_preconditioning.py").read_bytes(), "gram_preconditioning.py difiere de src/"
    assert (ds / "dfg.py").read_bytes() == (ROOT / "src/research_pipeline/sciml/dfg.py").read_bytes(), "dfg.py difiere de src/"
    for p in sorted(base.iterdir()):
        if p.name not in SKIP:
            assert (ds / p.name).read_bytes() == p.read_bytes(), f"{p.name} difiere de dataset_1e"
    for n in NEW_FILES:                                         # nada de credenciales: el kernel no tiene internet ni clave
        txt = (ds / n).read_text(encoding="utf-8")
        assert not re.search(r"os\.environ(\.get)?\(?\[?[\"']WANDB_API_KEY|load_dotenv|\.env[\"']|wandb\.login", txt), f"{n}: toca credenciales"
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0028 bundle slice", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    comps = {f"dataset_slice/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    tpl_path = SLICE / "kaggle_block_slice.py"
    template = tpl_path.read_text(encoding="utf-8")
    comps["artifact_slice/kaggle_block_slice.py"] = sha(tpl_path)
    appr = {"bundle": "exp0028-slice", "dataset": DATASET, "launch_status": "prepared_not_uploaded", "uploaded": False, "executed": False,
            "approved_by": "NINGUNA: preparado sin lanzar (Enmienda v1.6); la subida/ejecucion queda pendiente de la orden del investigador",
            "recipe": "receta MEDIA (budget_slice.md): ns Adam 10000, dfg 6000, ac 10000, kpp 8000, burgers 10000 (16 384 pts) + L-BFGS 500 it.; brazos B,F,J,E,I; semillas 4201,4202; "
                      "instantaneas completas en {0,2000,M/2,M,cada 5000} y L-BFGS final; livianas cada 1000 pasos",
            "caps": "sin trabajo nuevo despues de 3.4 h; kill a 3.9 h", "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "wandb": "OFFLINE dentro de la salida; sin clave en el bundle ni en el kernel",
            "scientific_accepted": False, "kernels": {}}
    for sub, k in KERNELS.items():
        kd = HERE / sub
        kd.mkdir(exist_ok=True)
        code = f"exp0028_slice_{k['block']}_gpu.py"
        for old in kd.glob("exp0028_slice_*_gpu.py"):
            if old.name != code:
                old.unlink()
        (kd / code).write_text(template.replace('BLOCK = "ns"   # @@BLOCK@@', f'BLOCK = "{k["block"]}"   # fijado por build_bundle_slice.py'), encoding="utf-8")
        (kd / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{USER}/{k['slug']}", "title": f"AF PIKANs EXP-0028 slice {k['block']}", "code_file": code, "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
             "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
        comps[f"{sub}/{code}"], comps[f"{sub}/kernel-metadata.json"] = sha(kd / code), sha(kd / "kernel-metadata.json")
        appr["kernels"][sub] = dict(slug=f"{USER}/{k['slug']}", block=k["block"], session_estimate_minutes=k["minutes"])
    appr["component_sha256"] = comps
    (HERE / "approval_slice.json").write_text(json.dumps(appr, indent=2))
    (HERE / "SHA256SUMS_slice.txt").write_text("".join(f"{h}  {n}\n" for n, h in comps.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir()), "kernels": list(KERNELS)}))


if __name__ == "__main__":
    main()
