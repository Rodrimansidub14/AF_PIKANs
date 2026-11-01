"""Arma el bundle de Kaggle de la prueba corta del EXP-0023 (plantilla: EXP-0024/kaggle/build_bundle.py). No sube nada. Escribe approval.json y SHA256SUMS.

  kaggle/dataset/    dataset privado `af-pikans-exp0023-bundle`: mecanismo, entrenador, medidor de presupuesto, barrido (EXP-0016), campana y referencias
  kaggle/shorttest/  kernel de la prueba corta en T4 (pruebas 6 y 7 del spec); NO es el bloque
"""
import hashlib, json, os, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[3]))
EXP23, EXP24 = HERE.parent, ROOT / "experiments/mechanisms/EXP-0024-wind-tunnel-protocol"
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
USER = "rodrimandub"
DATASET = f"{USER}/af-pikans-exp0023-bundle"
KEEP = ("launch_status", "launch_authorized_by", "results", "launched_at")
SOURCES = {
    "gram_preconditioning.py": ROOT / "src/mechanisms/gram_preconditioning.py",
    "pre_trainer.py": EXP23 / "pre_trainer.py",
    "measure_budget.py": EXP23 / "measure_budget.py",
    "t1_trainer.py": EXP24 / "t1_trainer.py",
    "regrid.py": EXP24 / "regrid.py",
    "sweep.py": EXP16 / "sweep.py",
    "stall_handoff.py": ROOT / "src/mechanisms/stall_handoff.py",
    "settling_focus.py": ROOT / "src/mechanisms/settling_focus.py",
    "pikan_campaign.py": BL / "burgers-close-20261003/pikan_campaign.py",
    "burgers_coleHopf_1025x129.npz": BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz",
    "Allen_Cahn.mat": BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat",
}
AUTH_SHORT = "researcher, chat 2026-10-21: «Acepto el minimo de P (0.05-0.6)... Repeti la 7b en la T4 con el criterio de la enmienda (5 brazos, 2 problemas, CPU y GPU, 100 pasos), sin lanzar el bloque»"
AUTH_BLOCK = ("researcher, chat 2026-10-21: «Autorizo armar y lanzar el bloque T1 del EXP-0023/ronda 1-B en Kaggle T4: brazos B,J,E,I,F x semillas 1-5 x Burgers y Allen-Cahn "
              "(50 corridas), convencion T1, criterio mediana <=0.90 en ambos problemas, topes 4 h / 5 h como en el 0024, sin variantes nuevas»")
AUTH_V = ("researcher, chat 2026-10-21: «Pasan E y F a V (semillas 100-199, mismo presupuesto, con B de comparador). Preparalo y pedime autorizacion antes de lanzar.» "
          "(preparado; lanzamiento SIN autorizar hasta nuevo mensaje del investigador)")
KERNELS = {
    "shorttest": dict(slug="af-pikans-exp-0023-shorttest", title="AF PIKANs EXP-0023 shorttest", code="exp0023_shorttest_gpu.py", minutes=[15, 40], auth=AUTH_SHORT,
                      purpose="7b repetida tras la Enmienda 1 (P normalizado) y 7a de nuevo, EXP-0023; no lanza el bloque", scope="exp0023_shorttest_only", big=False,
                      recipe="semilla 1; 7a estado compartido float64 (N=512); 7b: B/J/E/I/F con Adam 100, recalculo cada 50, N=512, criterio de la Enmienda 1 (perdida finita y <= 2x B en el paso 100)"),
    "vblock": dict(slug="af-pikans-exp-0023-v-block", title="AF PIKANs EXP-0023 V block", code="exp0023_v_block_gpu.py", minutes=[20, 90], auth=AUTH_V,
                   purpose="etapa V de la ronda 1-B: validar E y F frente a B en semillas disjuntas de T1 (mediana <= 0.90 y mejora en >= 4 de 5 en Burgers y Allen-Cahn)",
                   scope="exp0023_v_block", big=True,
                   recipe="B,E,F x semillas 101-105 (bloque V = 100-199) x Burgers y Allen-Cahn (30 corridas); mismo presupuesto y receta que el bloque T1 (Adam 2 000 / 4 000, L-BFGS 500 it., P normalizado)",
                   caps="no se despacha trabajo nuevo despues de 4 h; se matan los trabajos en curso a las 5 h; lo que no termine es ausencia de medicion"),
    "t1block": dict(slug="af-pikans-exp-0023-t1-block", title="AF PIKANs EXP-0023 T1 block", code="exp0023_t1_block_gpu.py", minutes=[40, 240], auth=AUTH_BLOCK,
                    purpose="bloque T1 de la ronda 1-B: ¿algun precondicionador (J, E, I, F) baja el error respecto de B (mediana de la razon <= 0.90 en Burgers y en Allen-Cahn)?",
                    scope="exp0023_t1_block", big=True,
                    recipe="B,J,E,I,F x semillas 1-5 x Burgers y Allen-Cahn (50 corridas); Adam 10 % del control (2 000 / 4 000), L-BFGS tope de 500 iteraciones; recalculo cada 1000 pasos "
                           "(0, 1000) y al arrancar L-BFGS, N = 2048, P normalizado (Enmienda 1); un proceso por (problema, semilla) con calentamiento",
                    caps="no se despacha trabajo nuevo despues de 4 h; se matan los trabajos en curso a las 5 h; lo que no termine es ausencia de medicion"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ds = HERE / "dataset"
    ds.mkdir(exist_ok=True)
    for n, s in SOURCES.items():
        shutil.copyfile(s, ds / n)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "AF PIKANs EXP-0023 bundle", "id": DATASET, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
    hashes = {f"dataset/{p.name}": sha(p) for p in sorted(ds.iterdir()) if p.name != "dataset-metadata.json"}
    all_comp = dict(hashes)
    for sub, k in KERNELS.items():
        kd = HERE / sub
        kd.mkdir(exist_ok=True)
        (kd / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{USER}/{k['slug']}", "title": k["title"], "code_file": k["code"], "language": "python", "kernel_type": "script", "is_private": "true",
             "enable_gpu": "true", "enable_internet": "false", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [],
             "kernel_sources": [], "model_sources": []}, indent=2))
        comp = {f"{sub}/{k['code']}": sha(kd / k["code"]), f"{sub}/kernel-metadata.json": sha(kd / "kernel-metadata.json")}
        all_comp.update(comp)
        old = {}
        if (kd / "approval.json").exists():
            prev = json.loads((kd / "approval.json").read_text())
            old = {key: prev[key] for key in KEEP if key in prev}
        appr = {"bundle": f"exp0023-{sub}", "slug": f"{USER}/{k['slug']}", "dataset": DATASET, "purpose": k["purpose"], "recipe": k["recipe"],
                "session_estimate_minutes": k["minutes"], "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "approved_by": k["auth"], "scope": k["scope"],
                "big_block_authorized": k["big"], "launch_status": "prepared_not_uploaded", **({"caps": k["caps"]} if "caps" in k else {}),
                "hypothesis": "ninguna firmada por corrida: criterio T1 de la regla de salida (mediana <= 0.90 en ambos problemas), comparador B de la misma sesion" if k["big"] else "ninguna: verificacion previa",
                "code_surface": "src/mechanisms/gram_preconditioning.py; experiments/mechanisms/EXP-0023-gram-preconditioning", "remote_execution": True,
                "scientific_accepted": False, "reference_accepted": False, "component_sha256": {**comp, **hashes}}
        appr.update(old)
        (kd / "approval.json").write_text(json.dumps(appr, indent=2))
    (HERE / "SHA256SUMS.txt").write_text("".join(f"{h}  {n}\n" for n, h in all_comp.items()))
    print(json.dumps({"files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir())}))


if __name__ == "__main__":
    main()
