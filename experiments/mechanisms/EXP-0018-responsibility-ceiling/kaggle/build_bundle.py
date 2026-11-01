"""Arma el bundle de Kaggle del EXP-0018 (techo oraculo del foco por responsabilidad) a partir de las fuentes del repo; escribe approval.json y SHA256SUMS. No sube nada.

  kaggle/dataset/    dataset privado `af-pikans-exp0018-bundle`: codigo (ejecutor, mecanismo, senales, barrido, campana), referencias, los 8 estados finales de Adam
                     de la etapa A del EXP-0016 y los hashes de referencia de su L-BFGS (k = 1.00)
  kaggle/precheck/   kernel de la prueba 7 (B reproduce la etapa A bit a bit), dos trabajos en paralelo, ~6 min
  kaggle/ceiling/    kernel del bloque completo (Burgers 301-305 y Allen-Cahn 601-605; ramas B, P, H, N, O)
Los campos de lanzamiento de un approval.json ya existente se conservan.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[3]))
EXP18 = HERE.parent
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
EXP17 = ROOT / "experiments/mechanisms/EXP-0017-representation-battery"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
STAGE_A = EXP16 / "results/stage-a/stageA"
USER = "rodrimandub"
DATASET_SLUG = f"{USER}/af-pikans-exp0018-bundle"
KEEP = ("launch_status", "launch_authorized_by", "results", "launched_at")

SOURCES = {
    "run_ceiling.py": EXP18 / "run_ceiling.py",
    "responsibility_focus.py": ROOT / "src/mechanisms/responsibility_focus.py",
    "signals.py": EXP17 / "signals.py",
    "sweep.py": EXP16 / "sweep.py",
    "stall_handoff.py": ROOT / "src/mechanisms/stall_handoff.py",
    "settling_focus.py": ROOT / "src/mechanisms/settling_focus.py",
    "pikan_campaign.py": BL / "burgers-close-20261003/pikan_campaign.py",
    "burgers_coleHopf_1025x129.npz": BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz",
    "Allen_Cahn.mat": BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat",
}
STATES = [("burgers", s, 20000) for s in (301, 302, 303, 304, 305)] + [("ac", s, 40000) for s in (601, 602, 603)]

KERNELS = {
    "precheck": dict(
        slug="af-pikans-exp-0018-precheck", title="AF PIKANs EXP-0018 precheck", code="precheck_ceiling_gpu.py", minutes=[5, 10],
        purpose="prueba 7 del spec: en T4, la rama B del ejecutor reproduce bit a bit el L-BFGS de la etapa A del EXP-0016 (k = 1.00) desde el mismo estado final de Adam",
        recipe="Burgers 301 y Allen-Cahn 601, solo la rama B, un proceso por semilla con calentamiento, una GPU cada uno",
        approved_by="researcher, chat 2026-10-20: firmo EXP-0018 v2 y aprobo el uso de la referencia como oraculo; 'prepara el bundle de Kaggle, haz la prueba 7 en T4 "
                    "(B reproduce la etapa A bit a bit) y, si pasa, lanza [el bloque completo]'",
        scope="test7_only", big_block=False),
    "ceiling": dict(
        slug="af-pikans-exp-0018-ceiling", title="AF PIKANs EXP-0018 ceiling", code="ceiling_gpu.py", minutes=[70, 180],
        purpose="bloque completo del EXP-0018 (techo oraculo del foco por responsabilidad, oracle_ceiling): ramas B, P, H, N, O desde el estado final de Adam; la referencia da A* y puntua",
        recipe="Burgers 301-305 (RBF48, estado de Adam de la etapa A, L-BFGS tope 600 s) y Allen-Cahn 601-605 (RBF16, tope 300 s; Adam de 604 y 605 entrenado aqui); beta = 1, q = 0.5; "
               "N-H NO (solo si H pasa H-CEIL-1)",
        approved_by="researcher, chat 2026-10-20: firmo EXP-0018 v2 (criterios, umbrales 0.85/0.90 con 4 de 5, beta=1, q=0.5, guardias g1-g3) y apruebo el uso de la referencia "
                    "como oraculo (oracle_ceiling); 'haz la prueba 7 en T4 y, si pasa, lanza [el bloque completo]'. Lanzamiento condicionado a que la prueba 7 pase",
        scope="exp0018_block", big_block=True,
        caps="no se despacha trabajo nuevo despues de 2.5 h; se matan los trabajos en curso a las 3.0 h (tope duro del spec)"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ds = HERE / "dataset"
    ds.mkdir(exist_ok=True)
    for name, src in SOURCES.items():
        shutil.copyfile(src, ds / name)
    ref_hashes = {}
    for problem, seed, k in STATES:
        shutil.copyfile(STAGE_A / f"{problem}-seed{seed}-adam-k{k}.pt", ds / f"{problem}-seed{seed}-adam-k{k}.pt")
        if problem == "ac" or True:
            r = json.loads((STAGE_A / f"{problem}-seed{seed}.json").read_text())["runs"]["1.0"]
            ref_hashes[f"{problem}-{seed}"] = dict(state_sha256=r["final_state_sha256"], global_l2=next(iter(r["final_score"].values())),
                                                   lbfgs_iterations=r["lbfgs_iterations"], lbfgs_seconds=r["lbfgs_seconds"])
    (ds / "stageA_reference_hashes.json").write_text(json.dumps(ref_hashes, indent=1))
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "AF PIKANs EXP-0018 bundle", "id": DATASET_SLUG, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    dataset_hashes = {f"dataset/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    all_components = dict(dataset_hashes)
    for sub, k in KERNELS.items():
        kd = HERE / sub
        kd.mkdir(exist_ok=True)
        (kd / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{USER}/{k['slug']}", "title": k["title"], "code_file": k["code"], "language": "python", "kernel_type": "script",
             "is_private": "true", "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
             "dataset_sources": [DATASET_SLUG], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
        components = {f"{sub}/{k['code']}": sha(kd / k["code"]), f"{sub}/kernel-metadata.json": sha(kd / "kernel-metadata.json")}
        all_components.update(components)
        old = {}
        if (kd / "approval.json").exists():
            prev = json.loads((kd / "approval.json").read_text())
            old = {key: prev[key] for key in KEEP if key in prev}
        approval = {
            "bundle": f"exp0018-{sub}", "slug": f"{USER}/{k['slug']}", "dataset": DATASET_SLUG, "purpose": k["purpose"], "recipe": k["recipe"],
            "session_estimate_minutes": k["minutes"], "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "approved_by": k["approved_by"],
            "scope": k["scope"], "big_block_authorized": bool(k["big_block"]), "launch_status": "prepared_not_uploaded",
            **({"caps": k["caps"]} if "caps" in k else {}),
            "hypothesis": "H-CEIL-1 y H-CEIL-2 (spec firmado); los brazos con A* leen la referencia y estan etiquetados oracle_ceiling" if k["big_block"] else "ninguna: verificacion de reproducibilidad",
            "code_surface": "experiments/mechanisms/EXP-0018-responsibility-ceiling, src/mechanisms/responsibility_focus.py", "remote_execution": True,
            "scientific_accepted": False, "reference_accepted": False, "label": "oracle_ceiling",
            "component_sha256": {**components, **{n: h for n, h in dataset_hashes.items()}}}
        approval.update(old)
        (kd / "approval.json").write_text(json.dumps(approval, indent=2))
    (HERE / "SHA256SUMS.txt").write_text("".join(f"{h}  {n}\n" for n, h in all_components.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir())}))


if __name__ == "__main__":
    main()
