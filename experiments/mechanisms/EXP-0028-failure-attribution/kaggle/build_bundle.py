"""Arma el bundle de Kaggle de la Fase 1 del EXP-0028 (patron: EXP-0023/kaggle/build_bundle.py y EXP-0026). NO sube ni lanza nada. Escribe:
  kaggle/dataset/            dataset privado `rodrimandub/af-pikans-exp0028-bundle` (codigo + referencias; todo plano, BUNDLE_DIR)
  kaggle/kernel/             kernel `rodrimandub/af-pikans-exp-0028-phase1`  (BLOCK=all: 4 problemas)
  kaggle/kernel_a/, kernel_b/ alternativa partida en dos lanzamientos (a: ac+ac2e4, b: burgers+kpp), slugs ...-phase1a / ...-phase1b
  kaggle/approval.json, SHA256SUMS.txt
Uso: python kaggle/build_bundle.py"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
ROOT = HERE.parents[3]
M = ROOT / "experiments/mechanisms"
C1 = M / "EXP-0023-gram-preconditioning/kaggle/c1dataset"      # campana, referencias y arnes 1-C ya empaquetados y con hash registrado
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0028-bundle"
SOURCES = {
    "run_phase1.py": EXP28 / "run_phase1.py",
    "gram_preconditioning.py": ROOT / "src/mechanisms/gram_preconditioning.py",
    "observer_v0.py": ROOT / "src/mechanisms/observer_v0.py",
    "rad.py": ROOT / "src/mechanisms/rad.py",
    "self_gated_sampler.py": ROOT / "src/mechanisms/self_gated_sampler.py",
    "stall_handoff.py": ROOT / "src/mechanisms/stall_handoff.py",
    "settling_focus.py": ROOT / "src/mechanisms/settling_focus.py",
    "responsibility_focus.py": ROOT / "src/mechanisms/responsibility_focus.py",
    "pre_trainer.py": M / "EXP-0023-gram-preconditioning/pre_trainer.py",
    "arnes_round1c.py": M / "EXP-0023-gram-preconditioning/round1C/arnes_round1c.py",
    "problems_round1c.py": M / "EXP-0023-gram-preconditioning/round1C/problems_round1c.py",
    "provenance_round1c.json": M / "EXP-0023-gram-preconditioning/round1C/provenance_round1c.json",
    "fisher_kpp_ref.npz": M / "EXP-0023-gram-preconditioning/round1C/fisher_kpp_ref.npz",
    "allen_cahn_d2e-4_ref.npz": M / "EXP-0023-gram-preconditioning/round1C/allen_cahn_d2e-4_ref.npz",
    "t1_trainer.py": M / "EXP-0024-wind-tunnel-protocol/t1_trainer.py",
    "regrid.py": M / "EXP-0024-wind-tunnel-protocol/regrid.py",
    "sweep.py": M / "EXP-0016-stall-handoff/sweep.py",
    "run_ceiling.py": M / "EXP-0018-responsibility-ceiling/run_ceiling.py",
    "signals.py": M / "EXP-0017-representation-battery/signals.py",
    "pikan_campaign.py": C1 / "pikan_campaign.py",
    "burgers_coleHopf_1025x129.npz": C1 / "burgers_coleHopf_1025x129.npz",
    "Allen_Cahn.mat": C1 / "Allen_Cahn.mat",
}
KERNELS = {
    "kernel": dict(slug="af-pikans-exp-0028-phase1", title="AF PIKANs EXP-0028 phase1", block="all", minutes=[240, 250]),
    "kernel_a": dict(slug="af-pikans-exp-0028-phase1a", title="AF PIKANs EXP-0028 phase1a", block="a", minutes=[115, 135]),
    "kernel_b": dict(slug="af-pikans-exp-0028-phase1b", title="AF PIKANs EXP-0028 phase1b", block="b", minutes=[120, 140]),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ds = HERE / "dataset"
    ds.mkdir(exist_ok=True)
    for n, s in SOURCES.items():
        shutil.copyfile(s, ds / n)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0028 bundle", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    hashes = {f"dataset/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    comps = dict(hashes)
    template = (EXP28 / "kaggle_block.py").read_text(encoding="utf-8")
    appr = {"bundle": "exp0028-phase1", "dataset": DATASET, "launch_status": "prepared_not_uploaded", "uploaded": False, "executed": False,
            "approved_by": "NINGUNA para este lanzamiento: spec v1.1 Enmienda (chat 2026-10-23) autoriza GPU si pruebas previas pasan, presupuesto medido <= 4 h y umbrales sin cambio; "
                           "la subida/ejecucion queda pendiente de la orden del investigador",
            "recipe": "B,F,J,E,I x 4001-4010 + S,R,C,W x 4001-4005 x {ac, ac2e4, burgers, kpp}; Adam 2000 (burgers) / 4000, L-BFGS 500 it., float32; ver budget.md",
            "recomendacion": "PARTIR EN DOS: lanzar phase1a y phase1b (cada uno ~2 h); el kernel unico estima ~4.1 h y excede el tope de 4 h por lanzamiento", "caps": "sin trabajo nuevo despues de 3.4 h; kill a 3.9 h", "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "scientific_accepted": False, "kernels": {}}
    for sub, k in KERNELS.items():
        kd = HERE / sub
        kd.mkdir(exist_ok=True)
        code = f"exp0028_phase1_{k['block']}_gpu.py"
        for old in kd.glob("exp0028_phase1_*_gpu.py"):
            if old.name != code:
                old.unlink()
        (kd / code).write_text(template.replace('BLOCK = "all"   # @@BLOCK@@', f'BLOCK = "{k["block"]}"   # fijado por build_bundle.py'), encoding="utf-8")
        (kd / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{USER}/{k['slug']}", "title": k["title"], "code_file": code, "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
             "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
        comps[f"{sub}/{code}"], comps[f"{sub}/kernel-metadata.json"] = sha(kd / code), sha(kd / "kernel-metadata.json")
        appr["kernels"][sub] = dict(slug=f"{USER}/{k['slug']}", block=k["block"], session_estimate_minutes=k["minutes"])
    appr["component_sha256"] = comps
    (HERE / "approval.json").write_text(json.dumps(appr, indent=2))
    (HERE / "SHA256SUMS.txt").write_text("".join(f"{h}  {n}\n" for n, h in comps.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir())}))


if __name__ == "__main__":
    main()
