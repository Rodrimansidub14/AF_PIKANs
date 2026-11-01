"""Arma el bundle de Kaggle de la Fase 1c (NS2d-CG) del EXP-0028. NO sube ni lanza nada. Escribe:
  kaggle/dataset_1c/        dataset privado `rodrimandub/af-pikans-exp0028-bundle-1c` (todo plano, BUNDLE_DIR)
  kaggle/kernel_1c_<b>/     un kernel por bloque de semillas: c1 (4001-4003) `af-pikans-exp-0028-phase1c`, c2 (4004-4006) `...-phase1c-b2`, c3 (4007-4009) `...-phase1c-b3`,
                            c4 (4010) `...-phase1c-b4`, t (solo prueba B == ID y medicion de costo) `...-phase1c-timing`
  kaggle/approval_1c.json, SHA256SUMS_1c.txt
gram_preconditioning.py y residual_terms.py se copian SIEMPRE de src/mechanisms (la copia de EXP-0023/kaggle/dataset es anterior a residual_terms y a pde_residual: no se usa) y se verifica byte a byte.
Uso: python kaggle/build_bundle_1c.py"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
ROOT = HERE.parents[3]
M = ROOT / "experiments/mechanisms"
C1 = M / "EXP-0023-gram-preconditioning/kaggle/c1dataset"      # pikan_campaign.py (NSProblem identico al del kernel NS2d-CG del 2026-09-30; verificado por diff)
NS_REF = ROOT / "var/agent_runs/baseline-competitive-2026-09-27/ns2d-cg-20260930/reference/ns_0_obstacle.dat"
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0028-bundle-1c"
SOURCES = {
    "run_phase1c.py": EXP28 / "run_phase1c.py",
    "gram_preconditioning.py": ROOT / "src/mechanisms/gram_preconditioning.py",
    "residual_terms.py": ROOT / "src/mechanisms/residual_terms.py",
    "stall_handoff.py": ROOT / "src/mechanisms/stall_handoff.py",       # sweep.py los importa (no se usan en 1c)
    "settling_focus.py": ROOT / "src/mechanisms/settling_focus.py",
    "pre_trainer.py": M / "EXP-0023-gram-preconditioning/pre_trainer.py",
    "t1_trainer.py": M / "EXP-0024-wind-tunnel-protocol/t1_trainer.py",
    "regrid.py": M / "EXP-0024-wind-tunnel-protocol/regrid.py",
    "sweep.py": M / "EXP-0016-stall-handoff/sweep.py",
    "pikan_campaign.py": C1 / "pikan_campaign.py",
    "ns_0_obstacle.dat": NS_REF,
}
KERNELS = {
    "kernel_1c_c1": dict(slug="af-pikans-exp-0028-phase1c", title="AF PIKANs EXP-0028 phase1c", block="c1", minutes=[190, 215]),
    "kernel_1c_c2": dict(slug="af-pikans-exp-0028-phase1c-b2", title="AF PIKANs EXP-0028 phase1c b2", block="c2", minutes=[190, 215]),
    "kernel_1c_c3": dict(slug="af-pikans-exp-0028-phase1c-b3", title="AF PIKANs EXP-0028 phase1c b3", block="c3", minutes=[190, 215]),
    "kernel_1c_t1": dict(slug="af-pikans-exp-0028-phase1c-t1", title="AF PIKANs EXP-0028 phase1c t1", block="all_t1", minutes=[70, 90]),
    "kernel_1c_c4": dict(slug="af-pikans-exp-0028-phase1c-b4", title="AF PIKANs EXP-0028 phase1c b4", block="c4", minutes=[65, 80]),
    "kernel_1c_t": dict(slug="af-pikans-exp-0028-phase1c-timing", title="AF PIKANs EXP-0028 phase1c timing", block="t", minutes=[5, 15]),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ds = HERE / "dataset_1c"
    ds.mkdir(exist_ok=True)
    for n, s in SOURCES.items():
        shutil.copyfile(s, ds / n)
    for n in ("gram_preconditioning.py", "residual_terms.py"):
        assert (ds / n).read_bytes() == (ROOT / "src/mechanisms" / n).read_bytes(), n
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0028 bundle 1c", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    comps = {f"dataset_1c/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    template = (EXP28 / "kaggle_block_1c.py").read_text(encoding="utf-8")
    appr = {"bundle": "exp0028-phase1c", "dataset": DATASET, "launch_status": "prepared_not_uploaded", "uploaded": False, "executed": False,
            "approved_by": "NINGUNA para este lanzamiento: spec v1.1 (Fase 1c requiere que el mecanismo en residuo vectorial pase sus pruebas) y autorizacion GPU de la Enmienda (presupuesto medido <= 4 h por lanzamiento); la subida/ejecucion queda pendiente de la orden del investigador",
            "recipe": "NS2d-CG, RBF16 (2-16-16-3, 11 bases), B,F,J,E,I x 4001-4010; Adam 20000 lr 1e-3 lote completo float32 + L-BFGS 500 it. (sweep.lbfgs_stage); recalculo cada 1000 pasos y al arrancar L-BFGS; GN con 2048 puntos por clave; ver budget_1c.md",
            "caps": "sin trabajo nuevo despues de 3.4 h; kill a 3.9 h", "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "scientific_accepted": False, "kernels": {}}
    for sub, k in KERNELS.items():
        kd = HERE / sub
        kd.mkdir(exist_ok=True)
        code = f"exp0028_phase1c_{k['block']}_gpu.py"
        (kd / code).write_text(template.replace('BLOCK = "all"   # @@BLOCK@@', f'BLOCK = "{k["block"]}"   # fijado por build_bundle_1c.py'), encoding="utf-8")
        (kd / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{USER}/{k['slug']}", "title": k["title"], "code_file": code, "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
             "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
        comps[f"{sub}/{code}"], comps[f"{sub}/kernel-metadata.json"] = sha(kd / code), sha(kd / "kernel-metadata.json")
        appr["kernels"][sub] = dict(slug=f"{USER}/{k['slug']}", block=k["block"], session_estimate_minutes=k["minutes"])
    appr["component_sha256"] = comps
    (HERE / "approval_1c.json").write_text(json.dumps(appr, indent=2))
    (HERE / "SHA256SUMS_1c.txt").write_text("".join(f"{h}  {n}\n" for n, h in comps.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir())}))


if __name__ == "__main__":
    main()
