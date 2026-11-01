"""Arma el bundle de Kaggle del EXP-0024 nivel T1 (plantilla: EXP-0018/kaggle/build_bundle.py); escribe approval.json y SHA256SUMS. No sube nada.

  kaggle/dataset/    dataset privado `af-pikans-exp0024-bundle`: entrenador T1, reubicacion, barrido (EXP-0016), campana, referencias y la referencia de CPU de la prueba corta
  kaggle/shorttest/  kernel de la prueba corta en T4 (comparabilidad con la CPU + tiempos + estimacion del bloque), ~15 min
  kaggle/t1block/    kernel del bloque (CTRL, D1, D2, D3 x semillas 1-5 x Burgers y Allen-Cahn = 40 corridas)
Los campos de lanzamiento de un approval.json ya existente se conservan.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", HERE.parents[3]))
EXP24 = HERE.parent
EXP16 = ROOT / "experiments/mechanisms/EXP-0016-stall-handoff"
BL = ROOT / "var/agent_runs/baseline-competitive-2026-09-27"
USER = "rodrimandub"
DATASET_SLUG = f"{USER}/af-pikans-exp0024-bundle"
KEEP = ("launch_status", "launch_authorized_by", "results", "launched_at")
CPU_REF_DIR = EXP24 / "results/shorttest-cpu"

SOURCES = {
    "t1_trainer.py": EXP24 / "t1_trainer.py",
    "regrid.py": EXP24 / "regrid.py",
    "seeds.py": EXP24 / "seeds.py",
    "sweep.py": EXP16 / "sweep.py",
    "stall_handoff.py": ROOT / "src/mechanisms/stall_handoff.py",
    "settling_focus.py": ROOT / "src/mechanisms/settling_focus.py",
    "pikan_campaign.py": BL / "burgers-close-20261003/pikan_campaign.py",
    "burgers_coleHopf_1025x129.npz": BL / "burgers-close-20261003/reference/burgers_coleHopf_1025x129.npz",
    "Allen_Cahn.mat": BL / "ns2d-cg-20260930/outputs/ac/ac/Allen_Cahn.mat",
}

AUTH = ("researcher, chat 2026-10-20: «si firmo» autorizando el bloque T1 de 40 corridas en Kaggle T4 condicionado a que la prueba corta pase "
        "(T1-plan.md, «Autorización de lanzamiento»); orden: armar el bundle con el builder del 0018, hacer la prueba corta en la T4 y, si pasa y la estimación es de 4 h o menos, lanzar el bloque")
KERNELS = {
    "shorttest": dict(
        slug="af-pikans-exp-0024-t1-shorttest", title="AF PIKANs EXP-0024 T1 shorttest", code="t1_shorttest_gpu.py", minutes=[10, 20],
        purpose="prueba corta en T4 del bloque T1: el control reproduce la corrida de CPU dentro de tolerancia, se mide el tiempo por paso y se estima el bloque",
        recipe="semilla 1; parte A: CTRL y D1 con Adam 200 / L-BFGS 50 (referencia de CPU); parte B: CTRL y D3 con el presupuesto T1 completo; Burgers en la GPU 0, Allen-Cahn en la 1",
        scope="t1_shorttest_only", big_block=False),
    "diag": dict(
        slug="af-pikans-exp-0024-t1-d1diag", title="AF PIKANs EXP-0024 T1 d1diag", code="t1_d1_diag_gpu.py", minutes=[5, 12],
        purpose="diagnostico de la falla de la prueba corta (Burgers D1 20.11 % frente a la CPU): variacion de la GPU consigo misma y D1 en float64",
        recipe="Burgers D1, semilla 1, Adam 200 / L-BFGS 50; float32 dos veces (un proceso cada una, con calentamiento) y float64 en la GPU",
        scope="t1_d1_diagnostic_only", big_block=False),
    "regridcheck": dict(
        slug="af-pikans-exp-0024-regrid-check", title="AF PIKANs EXP-0024 regrid check", code="regrid_cpu_vs_gpu.py", minutes=[1, 5],
        purpose="¿la reubicacion de centros ocultos da lo mismo en CPU y GPU sobre el mismo estado? (criterio del investigador: < ~1e-9 en float64)",
        recipe="Burgers RBF48 semilla 1 tras 50 pasos de Adam, estado guardado en float64; regrid_kan una vez en CPU y una en GPU; sin entrenar",
        scope="regrid_cpu_vs_gpu_only", big_block=False),
    "t1block": dict(
        slug="af-pikans-exp-0024-t1-block", title="AF PIKANs EXP-0024 T1 block", code="t1_block_gpu.py", minutes=[40, 240],
        purpose="bloque T1 de la ronda 1 del EXP-0024: ¿alguno de D1, D2, D3 baja el error respecto del control (mediana <= 0.90 en Burgers y Allen-Cahn)?",
        recipe="CTRL, D1, D2, D3 x semillas 1-5 x Burgers y Allen-Cahn (40 corridas); Adam 10 % del control (2 000 / 4 000), L-BFGS tope de 500 iteraciones; un proceso por (problema, semilla) con calentamiento",
        scope="exp0024_t1_block", big_block=True,
        caps="no se despacha trabajo nuevo despues de 4 h; se matan los trabajos en curso a las 5 h; lo que no termine es ausencia de medicion"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_reference():
    ref = {}
    for f in sorted(CPU_REF_DIR.glob("*-seed1-a200-l50.json")):
        r = json.loads(f.read_text())
        ref[f"{r['problem']}-{r['design']}"] = dict(primary_adam=r["primary_adam"], primary_final=r["primary_final"], total_seconds=r["total_seconds"])
    return ref


def main():
    ds = HERE / "dataset"
    ds.mkdir(exist_ok=True)
    for name, src in SOURCES.items():
        shutil.copyfile(src, ds / name)
    ref = cpu_reference()
    (ds / "cpu_reference.json").write_text(json.dumps(ref, indent=1))
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "AF PIKANs EXP-0024 bundle", "id": DATASET_SLUG, "licenses": [{"name": "other"}], "isPrivate": True}, indent=2))
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
            "bundle": f"exp0024-{sub}", "slug": f"{USER}/{k['slug']}", "dataset": DATASET_SLUG, "purpose": k["purpose"], "recipe": k["recipe"],
            "session_estimate_minutes": k["minutes"], "accelerator": "NvidiaTeslaT4 (T4x2)", "internet": False, "approved_by": AUTH,
            "scope": k["scope"], "big_block_authorized": bool(k["big_block"]), "launch_status": "prepared_not_uploaded", **({"caps": k["caps"]} if "caps" in k else {}),
            "hypothesis": "ninguna firmada por corrida: el protocolo EXP-0024 autoriza explorar en T0/T1; se mide contra el control de la misma sesion" if k["big_block"] else "ninguna: verificacion previa",
            "code_surface": "experiments/mechanisms/EXP-0024-wind-tunnel-protocol", "remote_execution": True, "scientific_accepted": False, "reference_accepted": False,
            "component_sha256": {**components, **dataset_hashes}}
        approval.update(old)
        (kd / "approval.json").write_text(json.dumps(approval, indent=2))
    (HERE / "SHA256SUMS.txt").write_text("".join(f"{h}  {n}\n" for n, h in all_components.items()))
    print(json.dumps({"dataset_files": len(list(ds.iterdir())), "bytes": sum(p.stat().st_size for p in ds.iterdir()), "cpu_reference_entries": sorted(ref)}))


if __name__ == "__main__":
    main()
