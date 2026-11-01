"""Arma el bundle de Kaggle del EXP-0016 a partir de las fuentes del repo (una sola fuente de verdad) y escribe approval.json + SHA256SUMS.

  kaggle/dataset/       dataset privado `af-pikans-exp0016-bundle`: codigo (sweep, mecanismos, campana) y referencias de Burgers y Allen-Cahn
  kaggle/precheck/      kernel privado de la prueba previa en GPU (lanzado y completo el 2026-10-19)
  kaggle/warmup_check/  kernel privado que comprueba el calentamiento en GPU (solo Burgers, un T4, ~10 min)

Uso: python build_bundle.py   (desde cualquier carpeta). No sube nada. Los campos de lanzamiento de un approval.json ya existente se conservan.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[3]))
EXP = HERE.parent
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
USER = "rodrimandub"
DATASET_SLUG = f"{USER}/af-pikans-exp0016-bundle"
KEEP = ("launch_status", "launch_authorized_by", "results", "launched_at")

SOURCES = {
    "sweep.py": EXP / "sweep.py",
    "stall_handoff.py": ROOT / "src/mechanisms/stall_handoff.py",
    "settling_focus.py": ROOT / "src/mechanisms/settling_focus.py",
    "pikan_campaign.py": BL / "burgers-close-20261003/pikan_campaign.py",
    "burgers_coleHopf_1025x129.npz": BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz",
    "Allen_Cahn.mat": BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat",
}

# slug = el que Kaggle deriva del titulo (minusculas, espacios -> guiones)
KERNELS = {
    "precheck": dict(
        slug="af-pikans-exp-0016-precheck-gpu", title="AF PIKANs EXP-0016 precheck GPU", code="precheck_gpu.py",
        purpose="prueba previa (ii-GPU): ruido de GPU con tres controles y k = 1.00 del barrido corto; tiempos de L-BFGS desde cada punto de la malla",
        recipe="Burgers RBF48 (semilla 301) y Allen-Cahn RBF16 (semilla 601), f32, 4000 pasos de Adam, L-BFGS limitado a 400 iteraciones; 3 controles + 1 barrido por problema",
        minutes=[20, 25],
        approved_by="researcher, chat 2026-10-19: firmo EXP-0016 (etapas A y B); aprobo el uso de la referencia (oraculo y etiquetas de B1); "
                    "pidio preparar el bundle con la prueba previa de GPU y NO lanzar el bloque grande",
        scope="precheck_only"),
    "warmup_check": dict(
        slug="af-pikans-exp-0016-warmup-check", title="AF PIKANs EXP-0016 warmup check", code="verify_warmup_gpu.py",
        purpose="comprobar en GPU que el calentamiento (sweep.warmup) hace que la primera corrida de un proceso reproduzca a las siguientes",
        recipe="Burgers RBF48 (semilla 301), f32, 4000 pasos de Adam, L-BFGS limitado a 400 iteraciones; calentamiento + 2 controles + barrido corto (malla 0.5-1.00)",
        minutes=[9, 12],
        approved_by="researcher, chat 2026-10-19: 'agrega el calentamiento a sweep.py (corrida desechable con las mismas formas, con prueba en CPU), "
                    "prepara el kernel de verificacion y lanzalo en T4 (solo Burgers, unos 8 min). No lances el bloque grande.'",
        scope="warmup_check_only"),
    "stage_a": dict(
        slug="af-pikans-exp-0016-stage-a", title="AF PIKANs EXP-0016 stage A", code="stage_a_gpu.py",
        purpose="etapa A del EXP-0016 (techo del traspaso Adam -> L-BFGS, oraculo oracle_ceiling): barrido de traspaso por semilla; la referencia solo puntua despues de entrenar",
        recipe="Burgers RBF48 semillas 301-305 (20000 Adam f32, L-BFGS tope 600 s, parada por meseta) y despues Allen-Cahn RBF16 semillas 601-603 (40000 Adam, tope 300 s); "
               "malla 0.25/0.40/0.60/0.80/1.00; calentamiento por proceso; un proceso por semilla, dos ranuras (una por GPU). HEAT NO.",
        minutes=[150, 300],
        approved_by="researcher, chat 2026-10-19: firmo EXP-0016 v2 (etapas A y B), aprobo el uso de la referencia (oraculo y etiquetas de B1) y ordeno: "
                    "'lanza la etapa A en Burgers (301-305) y despues Allen-Cahn (601-603) en el mismo kernel, con calentamiento. Heat no.'",
        scope="stage_A_burgers_allen_cahn", big_block=True,
        caps="no se despacha trabajo nuevo despues de 3.5 h; se matan los trabajos en curso a las 5.0 h (tope duro del spec)"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ds = HERE / "dataset"
    ds.mkdir(exist_ok=True)
    for name, src in SOURCES.items():
        shutil.copyfile(src, ds / name)
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "AF PIKANs EXP-0016 bundle", "id": DATASET_SLUG, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    dataset_hashes = {f"dataset/{n}": sha(ds / n) for n in SOURCES}
    all_components = dict(dataset_hashes)
    for sub, k in KERNELS.items():
        kd = HERE / sub
        kd.mkdir(exist_ok=True)
        (kd / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{USER}/{k['slug']}", "title": k["title"], "code_file": k["code"], "language": "python", "kernel_type": "script",
             "is_private": "true", "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
             "dataset_sources": [DATASET_SLUG], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
        components = dict(dataset_hashes)
        components[f"{sub}/{k['code']}"] = sha(kd / k["code"])
        components[f"{sub}/kernel-metadata.json"] = sha(kd / "kernel-metadata.json")
        all_components.update({n: h for n, h in components.items() if not n.startswith("dataset/")})
        old = {}
        if (kd / "approval.json").exists():
            prev = json.loads((kd / "approval.json").read_text())
            old = {key: prev[key] for key in KEEP if key in prev}
        approval = {
            "bundle": f"exp0016-{sub}", "slug": f"{USER}/{k['slug']}", "dataset": DATASET_SLUG, "purpose": k["purpose"], "recipe": k["recipe"],
            "session_estimate_minutes": k["minutes"], "accelerator": "NvidiaTeslaT4 (un T4 usado)", "internet": False,
            "approved_by": k["approved_by"], "scope": k["scope"], "big_block_authorized": bool(k.get("big_block", False)),
            "launch_status": "prepared_not_uploaded", **({"caps": k["caps"]} if "caps" in k else {}),
            "hypothesis": "ninguna: no puntua hipotesis; fija el presupuesto y la reproducibilidad de la etapa A",
            "code_surface": "experiments/mechanisms/EXP-0016-stall-handoff y src/mechanisms", "remote_execution": True,
            "campaign_script_note": "pikan_campaign.py (sha 1e1b94...) = el de la campana burgers-close (sha 4dab23..., recuperado del cuaderno) mas las opciones "
                                    "--n-col y --lbfgs-dtype; con sus valores por defecto (0 y same) la receta es identica. Diff verificado el 2026-10-19",
            "scientific_accepted": False, "reference_accepted": False, "component_sha256": components}
        approval.update(old)
        (kd / "approval.json").write_text(json.dumps(approval, indent=2))
    (HERE / "SHA256SUMS.txt").write_text("".join(f"{h}  {n}\n" for n, h in all_components.items()))
    print(json.dumps({"dataset_files": sorted(p.name for p in ds.iterdir()), "bytes": sum(p.stat().st_size for p in ds.iterdir())}))


if __name__ == "__main__":
    main()
