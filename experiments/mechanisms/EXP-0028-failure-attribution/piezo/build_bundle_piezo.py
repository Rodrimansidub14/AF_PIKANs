"""Arma el bundle de Kaggle del bloque final de piezo (bimorfo PVDF, converso y directo) del EXP-0028. NO sube ni lanza nada. Escribe:
  kaggle/dataset_piezo/   dataset privado `rodrimandub/af-pikans-exp0028-bundle-piezo` (todo plano, BUNDLE_DIR; NUEVO: no toca los otros bundles)
  kaggle/kernel_piezo/    kernel `rodrimandub/af-pikans-exp-0028-piezo` (BLOCK = --block: 't' por defecto = solo B == ID con calentamiento + medicion; 'kNN' = el par NN; 'all')
  kaggle/approval_piezo.json, SHA256SUMS_piezo.txt
gram_preconditioning.py y residual_terms.py se copian SIEMPRE de src/ y se verifica byte a byte. Las referencias FEM (fem_converse.npz, fem_direct.npz) son scikit-fem P2 200x8 (no aceptadas
como referencia de terceros). Uso: python piezo/build_bundle_piezo.py [--block t|all|k01..k26]"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
ROOT = HERE.parents[3]
KAG = EXP28 / "kaggle"
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0028-bundle-piezo"
SOURCES = {
    "run_piezo.py": HERE / "run_piezo.py",
    "stage0.py": HERE / "stage0.py",
    "artifacts.py": HERE / "artifacts.py",
    "piezo_problem.py": HERE / "piezo_problem.py",
    "gram_preconditioning.py": ROOT / "src/mechanisms/gram_preconditioning.py",
    "residual_terms.py": ROOT / "src/mechanisms/residual_terms.py",
    "fem_converse.npz": HERE / "reference/fem_converse.npz",
    "fem_direct.npz": HERE / "reference/fem_direct.npz",
    "fem_flux_converse.npz": HERE / "reference/fem_flux_converse.npz",
    "fem_flux_direct.npz": HERE / "reference/fem_flux_direct.npz",
}
KERNEL = dict(sub="kernel_piezo", slug="af-pikans-exp-0028-piezo", title="AF PIKANs EXP-0028 piezo")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="t")
    a = ap.parse_args()
    ds = KAG / "dataset_piezo"
    ds.mkdir(parents=True, exist_ok=True)
    for n, s in SOURCES.items():
        if s.exists():
            shutil.copyfile(s, ds / n)
        else:
            print('AVISO: falta', s)
    for n in ("gram_preconditioning.py", "residual_terms.py"):
        assert (ds / n).read_bytes() == (ROOT / "src/mechanisms" / n).read_bytes(), n
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0028 bundle piezo", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    comps = {f"dataset_piezo/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    template = (HERE / "kaggle_block_piezo.py").read_text(encoding="utf-8")
    k = KERNEL
    kd = KAG / k["sub"]
    kd.mkdir(exist_ok=True)
    code = f"exp0028_piezo_{a.block}_gpu.py"
    for old in kd.glob("exp0028_piezo_*_gpu.py"):
        old.unlink()
    (kd / code).write_text(template.replace('BLOCK = "t"   # @@BLOCK@@', f'BLOCK = "{a.block}"   # fijado por build_bundle_piezo.py'), encoding="utf-8")
    (kd / "kernel-metadata.json").write_text(json.dumps(
        {"id": f"{USER}/{k['slug']}", "title": k["title"], "code_file": code, "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
         "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
    comps[f"{k['sub']}/{code}"], comps[f"{k['sub']}/kernel-metadata.json"] = sha(kd / code), sha(kd / "kernel-metadata.json")
    appr = {"bundle": "exp0028-piezo", "dataset": DATASET, "launch_status": "prepared_not_uploaded", "uploaded": False, "executed": False,
            "approved_by": "NINGUNA: piezo NO se corre antes de que el investigador firme predictions.md; la subida/ejecucion queda pendiente de su orden",
            "recipe": "Bimorfo PVDF del paper (converso 100 V y directo 0.1 N), RBF-PIKAN 4-16-16-8 (11 bases) con las entradas/salidas del paper; B,F,J,E,I x semillas 20260728-20260732 (cohorte del paper) x 2 problemas = 50 corridas + 2 de referencia MLP-paper (MLP 3x50 SiLU, semilla 20260728, baja prioridad, bloque k26); "
                      "Adam 20 000 pasos lr 1e-3 float64 sin L-BFGS, remuestreo cada 200 (2x2048 interiores + 256 por frontera), pesos fijos del paper; recalculo del bloque antes del paso 1 y cada 1000 pasos "
                      "(20 por corrida); GN con 2048 puntos por clave del pool actual; error de u,v,phi cada 500 pasos contra el FEM de scikit-fem (P2 200x8, rejilla 201x21)",
            "block": a.block, "caps": "sin trabajo nuevo despues de CUTOFF_S (3.4 h); kill a HARD_CAP_S (3.9 h)", "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "scientific_accepted": False,
            "kernels": {k["sub"]: dict(slug=f"{USER}/{k['slug']}", block=a.block)}, "component_sha256": comps}
    (KAG / "approval_piezo.json").write_text(json.dumps(appr, indent=2))
    (KAG / "SHA256SUMS_piezo.txt").write_text("".join(f"{h}  {n}\n" for n, h in comps.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir()), "block": a.block}))


if __name__ == "__main__":
    main()
