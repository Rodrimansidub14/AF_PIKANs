"""Arma el bundle de Kaggle de la Fase 1c-DFG (DFG 2D-1 Re20) del EXP-0028. NO sube ni lanza nada. Escribe:
  kaggle/dataset_1c_dfg/     dataset privado `rodrimandub/af-pikans-exp0028-bundle-dfg` (todo plano, BUNDLE_DIR; NUEVO: no toca `af-pikans-exp0028-bundle-1c`)
  kaggle/kernel_1c_dfg/      kernel `rodrimandub/af-pikans-exp-0028-phase1c-dfg` (BLOCK all: semillas 4001-4010, B,F,J,E,I; antes de despachar corre --check-id --warmup-steps 50 y --timing)
  kaggle/approval_1c_dfg.json, SHA256SUMS_1c_dfg.txt
gram_preconditioning.py, residual_terms.py y dfg.py se copian SIEMPRE de src/ y se verifica byte a byte. La referencia FEM (dfg_L3_fields.npz) es la de trabajo (no aceptada).
Uso: python kaggle/build_bundle_1c_dfg.py"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
ROOT = HERE.parents[3]
M = ROOT / "experiments/mechanisms"
C1 = M / "EXP-0023-gram-preconditioning/kaggle/c1dataset"
REF = ROOT / "var/agent_runs/baseline-competitive-2026-09-27/ns2d-cg-20260930/reference/dfg_L3_fields.npz"
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0028-bundle-dfg"
SOURCES = {
    "run_phase1c_dfg.py": EXP28 / "run_phase1c_dfg.py",
    "gram_preconditioning.py": ROOT / "src/mechanisms/gram_preconditioning.py",
    "residual_terms.py": ROOT / "src/mechanisms/residual_terms.py",
    "stall_handoff.py": ROOT / "src/mechanisms/stall_handoff.py",       # sweep.py los importa (no se usan)
    "settling_focus.py": ROOT / "src/mechanisms/settling_focus.py",
    "pre_trainer.py": M / "EXP-0023-gram-preconditioning/pre_trainer.py",
    "t1_trainer.py": M / "EXP-0024-wind-tunnel-protocol/t1_trainer.py",
    "regrid.py": M / "EXP-0024-wind-tunnel-protocol/regrid.py",
    "sweep.py": M / "EXP-0016-stall-handoff/sweep.py",
    "pikan_campaign.py": C1 / "pikan_campaign.py",                      # t1_trainer lo importa
    "dfg.py": ROOT / "src/research_pipeline/sciml/dfg.py",             # DFGRBFPIKAN, generate_dfg_pools, evaluate_dfg_qois, train_dfg (scipy.stats.qmc)
    "dfg_L3_fields.npz": REF,
}
KERNEL = dict(sub="kernel_1c_dfg", slug="af-pikans-exp-0028-phase1c-dfg", title="AF PIKANs EXP-0028 phase1c dfg", block="all", minutes=[40, 100])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ds = HERE / "dataset_1c_dfg"
    ds.mkdir(exist_ok=True)
    for n, s in SOURCES.items():
        shutil.copyfile(s, ds / n)
    for n in ("gram_preconditioning.py", "residual_terms.py"):
        assert (ds / n).read_bytes() == (ROOT / "src/mechanisms" / n).read_bytes(), n
    assert (ds / "dfg.py").read_bytes() == (ROOT / "src/research_pipeline/sciml/dfg.py").read_bytes()
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0028 bundle dfg", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    comps = {f"dataset_1c_dfg/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    template = (EXP28 / "kaggle_block_1c_dfg.py").read_text(encoding="utf-8")
    k = KERNEL
    appr = {"bundle": "exp0028-phase1c-dfg", "dataset": DATASET, "launch_status": "prepared_not_uploaded", "uploaded": False, "executed": False,
            "approved_by": "NINGUNA para este lanzamiento: la subida/ejecucion queda pendiente de la orden del investigador (receta DFG decidida por el investigador 2026-10-24)",
            "recipe": "DFG 2D-1 Re20, DFGRBFPIKAN (2-16-16-3, 11 bases), B,F,J,E,I x 4001-4010; Adam 2000 pasos fijos lr 1e-3 minibatch 1024+256/frontera con reemplazo float32 + L-BFGS 500 it. (sweep.lbfgs_stage, pool completo); "
                      "recalculo antes del paso 1, cada 1000 pasos y al arrancar L-BFGS; GN con 2048 puntos por clave (4608); puntuacion l2re_flat/QoI contra dfg_L3_fields.npz (referencia de trabajo)",
            "caps": "sin trabajo nuevo despues de CUTOFF_S; kill a HARD_CAP_S (ver kaggle_block_1c_dfg.py)", "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "scientific_accepted": False, "kernels": {}}
    kd = HERE / k["sub"]
    kd.mkdir(exist_ok=True)
    code = f"exp0028_phase1c_dfg_{k['block']}_gpu.py"
    (kd / code).write_text(template.replace('BLOCK = "all"   # @@BLOCK@@', f'BLOCK = "{k["block"]}"   # fijado por build_bundle_1c_dfg.py'), encoding="utf-8")
    (kd / "kernel-metadata.json").write_text(json.dumps(
        {"id": f"{USER}/{k['slug']}", "title": k["title"], "code_file": code, "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
         "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
    comps[f"{k['sub']}/{code}"], comps[f"{k['sub']}/kernel-metadata.json"] = sha(kd / code), sha(kd / "kernel-metadata.json")
    appr["kernels"][k["sub"]] = dict(slug=f"{USER}/{k['slug']}", block=k["block"], session_estimate_minutes=k["minutes"])
    appr["component_sha256"] = comps
    (HERE / "approval_1c_dfg.json").write_text(json.dumps(appr, indent=2))
    (HERE / "SHA256SUMS_1c_dfg.txt").write_text("".join(f"{h}  {n}\n" for n, h in comps.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir())}))


if __name__ == "__main__":
    main()
