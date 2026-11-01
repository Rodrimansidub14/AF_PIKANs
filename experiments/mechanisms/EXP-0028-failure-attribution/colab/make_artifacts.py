"""Genera (local, sin tocar nada de Kaggle) los manifiestos y los cuadernos de Colab:
  manifest_piezo.json         50 trabajos de piezo (converso y directo; B,F,J,E,I; semillas pareadas 4001-4005). Prioridad 1 = semillas 4001-4003 de AMBOS problemas; prioridad 2 = 4004-4005.
  manifest_blocks.json        bloques diferidos: DFG 1e (4101-4105), NS 1e (4104-4105; 4101-4103 ya corrieron), rebanada con artefactos (4201-4202), Burgers 1d (bu2/bu3) y Burgers 1e.
  manifest_smoke.json         humo en CPU del despachador (piezo diminuto); manifest_smoke_blocks.json: humo en CPU de las recetas de bloques.
  piezo_colab.ipynb, blocks_colab.ipynb
Las estimaciones `est_s_t4` salen de budget_1e_calc.py / budget_slice_calc.py (tiempos MEDIDOS en T4 del repositorio); el cuaderno las reescala con la prueba de tiempo en la A100.
Uso: python make_artifacts.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP28 = HERE.parent
sys.path.insert(0, str(EXP28))
sys.path.insert(0, str(EXP28 / "artifact_slice"))
import budget_1e_calc as b1e  # noqa: E402
import budget_slice_calc as bsl  # noqa: E402

ARMS = ("B", "F", "J", "E", "I")


def piezo_manifest():
    jobs = []
    for prio, seeds in ((1, (4001, 4002, 4003)), (2, (4004, 4005))):
        for s in seeds:
            for problem in ("converse", "direct"):
                for arm in ARMS:
                    pa = {"B": ["B"], "F": ["B"], "J": ["E"], "E": ["E"], "I": ["I", "E"]}[arm]
                    jobs.append(dict(id=f"piezo-{problem}-{arm}-seed{s}", recipe="piezo", group="piezo", problem=problem, arm=arm, seed=s, block=f"semillas-{'4001-4003' if prio == 1 else '4004-4005'}",
                                     priority=prio, steps=20000, n_recomputes=0 if arm in ("B", "F") else 20, fixed_s=60.0, probe_as=pa, step_factor=1.05 if arm == "F" else 1.0))
    return dict(manifest="piezo", version=1, device_note="una A100; 1 proceso por GPU (--jobs-per-gpu 2 solo si la prueba de concurrencia lo justifica)",
                notes=["Orden: semilla completa (ambos problemas, B,F,J,E,I) antes de la siguiente; la prioridad 1 son 30 trabajos (3 semillas), la 2 son 20.",
                       "steps/n_recomputes/fixed_s/probe_as: la proyeccion usa s/paso y s/recalculo medidos en la A100 (B para B y F x1.05; E para E y J; I para I, con E de respaldo).",
                       "Receta: Adam 20 000 pasos lr 1e-3 float64 sin L-BFGS; recalculo antes del paso 1 y cada 1000 (20 por corrida de J/E/I); ver piezo/budget_piezo.md."],
                probe=dict(targets=[dict(recipe="piezo", problem=p, arm=a, seed=4001) for p in ("converse", "direct") for a in ("B", "E", "I")],
                           t4_ref={"converse:B": dict(step_s=0.47), "direct:B": dict(step_s=0.31), "converse:E": dict(step_s=0.47), "direct:E": dict(step_s=0.31)}),
                jobs=jobs)


def step_ref():
    """s/paso de Adam en T4 SIN recalculo (de las mismas cifras de los calc) por (problema:brazo)."""
    ref = {}
    for a in ARMS:
        ref[f"ns:{a}"] = dict(step_s={"B": 0.0535, "F": 0.0555}.get(a, 0.0561))
        adam = {"B": (154.3, 0.0), "F": (160.1, 0.0), "J": (177.3, 19.03 / 3), "E": (172.8, 18.70 / 3), "I": (179.2, 20.84 / 3)}[a]
        ref[f"dfg:{a}"] = dict(step_s=(adam[0] - 2 * adam[1]) / 2000)
        ref[f"burgers:{a}"] = dict(step_s=0.0782)
        ref[f"ac:{a}"] = dict(step_s={"B": 0.0196, "F": 0.0203}.get(a, 0.0212))
    return ref


def burgers_1d_est(arm):                      # B,F,E de 1d (budget_1d: B 0.0782 s/paso; recalculo E 3.5 s x 25; L-BFGS en el tope de 1800 s)
    return 24000 * 0.0782 * (1.03 if arm == "F" else 1.0) + (3.5 * 25 if arm == "E" else 0) + 1800 + 25


def blocks_manifest():
    jobs = []

    def add(recipe, group, problem, arms, seeds, prio, block, est):
        for s in seeds:
            for arm in arms:
                jobs.append(dict(id=f"{group}-{arm}-seed{s}", recipe=recipe, group=group, problem=problem, arm=arm, seed=s,
                                 block=block, priority=prio, est_s_t4=round(est(problem, arm), 1)))

    e1e = lambda p, a: b1e.EST[p](a)  # noqa: E731
    esl = lambda p, a: bsl.est(p, a)  # noqa: E731
    e1d = lambda p, a: burgers_1d_est(a)  # noqa: E731
    r = range
    add("phase1e", "p1e-ns", "ns", ARMS, r(4104, 4106), 1, "1e-ns2 (ns1 = 4101-4103 ya corrio en Kaggle)", e1e)
    add("phase1e", "p1e-dfg", "dfg", ARMS, r(4101, 4104), 1, "1e-dfg1/2 (4101-4103)", e1e)
    add("phase1e", "p1e-dfg", "dfg", ARMS, r(4104, 4106), 2, "1e-dfg2/3 (4104-4105)", e1e)
    for p in ("ns", "dfg", "ac", "kpp", "burgers"):
        add("slice", f"slice-{p}", p, ARMS, (4201, 4202), 3, "rebanada con artefactos (Enmienda v1.6)", esl)
    add("phase1d", "p1d-burgers", "burgers", ("B", "F", "E"), r(4103, 4106), 4, "1d bu2/bu3 (bu1 = 4101-4102 ya corrio)", e1d)
    add("phase1e", "p1e-burgers", "burgers", ("J", "I"), r(4101, 4106), 5, "1e bu1-bu3 (pareo con el B de 1d)", e1e)
    ref = step_ref()
    return dict(manifest="blocks", version=1,
                notes=["Prioridades: 1 = NS 1e (4104-4105) + DFG 1e (4101-4103); 2 = DFG 1e (4104-4105); 3 = rebanada con artefactos (ns, dfg, ac, kpp, burgers; B,F,J,E,I; 4201-4202); "
                       "4 = Burgers 1d (B,F,E; 4103-4105, de los que depende el pareo de Burgers 1e); 5 = Burgers 1e (J,I; 4101-4105).",
                       "AC/KPP 1e (kernel acf) NO esta aqui: va con la cuota de Kaggle. Lo ya corrido se detecta por archivos en --out, no por este manifiesto: si trae resultados previos a la carpeta de salida, se saltan.",
                       "est_s_t4 = segundos medidos/estimados en T4 (budget_1e_calc.py, budget_slice_calc.py); la proyeccion los multiplica por (s/paso medido en A100) / (s/paso T4) por problema."],
                probe=dict(targets=[dict(recipe="phase1e", problem="dfg", arm="B", seed=4101), dict(recipe="phase1e", problem="dfg", arm="J", seed=4101),
                                    dict(recipe="phase1e", problem="ns", arm="J", seed=4101), dict(recipe="phase1e", problem="burgers", arm="J", seed=4101),
                                    dict(recipe="phase1e", problem="ac", arm="J", seed=4101)],
                           t4_ref={k: v for k, v in ref.items() if k in ("dfg:B", "dfg:J", "ns:J", "burgers:J", "ac:J")}),
                jobs=jobs)


def smoke_manifest():
    tiny = ["--adam", "10", "--eval-every", "5", "--pre-every", "5", "--pre-n", "64", "--smoke-pool", "16"]
    jobs = []
    for i, (p, a, s) in enumerate([("converse", "B", 4001), ("converse", "E", 4001), ("direct", "B", 4001), ("direct", "J", 4001)]):
        jobs.append(dict(id=f"smoke-piezo-{p}-{a}-seed{s}", recipe="piezo", group="piezo", problem=p, arm=a, seed=s, block="humo", priority=1 if s == 4001 and p == "converse" else 2, extra_args=tiny,
                         steps=10, n_recomputes=0, fixed_s=0.0, probe_as=[a]))
    return dict(manifest="smoke", version=1, notes=["Humo en CPU: pasos y puntos diminutos; NO es ciencia."],
                probe=dict(targets=[dict(recipe="piezo", problem="converse", arm="B", seed=4001, probe_extra=["--smoke-pool", "16", "--pre-n", "64"]),
                                    dict(recipe="piezo", problem="converse", arm="E", seed=4001, probe_extra=["--smoke-pool", "16", "--pre-n", "64"])], t4_ref={}), jobs=jobs)


def smoke_blocks_manifest():
    return dict(manifest="smoke_blocks", version=1, notes=["Humo en CPU de las recetas phase1d / phase1e / slice con el bundle de bloques. NO es ciencia."], probe=dict(targets=[], t4_ref={}), jobs=[
        dict(id="smoke-p1d-ac-B", recipe="phase1d", group="smoke-p1d", problem="ac", arm="B", seed=4101, priority=1,
             extra_args=["--adam", "10", "--lbfgs-iters", "5", "--lbfgs-seconds", "30", "--pre-every", "5", "--pre-n", "64", "--n-col", "512"]),
        dict(id="smoke-p1e-ns-J", recipe="phase1e", group="smoke-p1e", problem="ns", arm="J", seed=4101, priority=1,
             extra_args=["--adam", "10", "--lbfgs-iters", "5", "--pre-every", "5", "--pre-n", "64", "--smoke-pool", "128"]),
        dict(id="smoke-slice-ac-J", recipe="slice", group="smoke-slice", problem="ac", arm="J", seed=4201, priority=2,
             extra_args=["--adam", "10", "--lbfgs-iters", "5", "--pre-every", "5", "--pre-n", "64", "--light-every", "5"])])


# --------------------------------------------------------------------------------------------------------------------------------------------------- cuadernos
def nb(cells):
    out = []
    for kind, src in cells:
        c = dict(cell_type=kind, metadata={}, source=src.strip("\n").splitlines(keepends=True))
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return dict(cells=out, metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="A100"), kernelspec=dict(display_name="Python 3", name="python3"), language_info=dict(name="python")),
                nbformat=4, nbformat_minor=5)


def notebook(kind):
    piezo = kind == "piezo"
    manifest = "manifest_piezo.json" if piezo else "manifest_blocks.json"
    bundle = "bundle_piezo" if piezo else "bundle_blocks"
    title = "EXP-0028 piezo (bimorfo PVDF, converso y directo) en Colab A100" if piezo else "EXP-0028 bloques diferidos (DFG 1e, NS 1e, rebanada con artefactos, Burgers 1d/1e) en Colab A100"
    prios = ("1 = semillas 4001-4003 de ambos problemas (30 trabajos); 2 = semillas 4004-4005 (20 trabajos)" if piezo else
             "1 = NS 1e 4104-4105 + DFG 1e 4101-4103; 2 = DFG 1e 4104-4105; 3 = rebanada con artefactos; 4 = Burgers 1d; 5 = Burgers 1e")
    c = []
    c.append(("markdown", f"""# {title}

**Antes de nada**: Entorno de ejecucion > Cambiar tipo > GPU **A100**; active el pago por uso en su cuenta de Colab. Este cuaderno **no pide ni usa ninguna clave** (W&B queda OFFLINE; se sincroniza despues, en local).
Orden: 1) montar Drive, 2) verificar GPU, 3) descomprimir el zip, 4) **prueba de tiempo**, 5) decidir cuantas semillas, 6) lanzar, 7) reanudar si se corta, 8) empaquetar y descargar.
Prioridades: {prios}."""))
    c.append(("code", f'''# 1. PARAMETROS (edite solo esta celda)
ZIP_EN_DRIVE = "/content/drive/MyDrive/x28_colab/exp0028_colab_bundle.zip"   # el zip que subio a Drive (build_colab_bundle.py)
SHA256_ESPERADO = ""                      # opcional: pegue el SHA256 del zip que imprimio build_colab_bundle.py
OUT = "/content/drive/MyDrive/x28_colab/salida_{kind}"          # carpeta PERSISTENTE (Drive): resultados, ledger.jsonl, wandb offline
SCRATCH = "/content/x28_scratch"          # carpeta LOCAL de trabajo (rapida); lo terminado se copia a OUT
WORK = "/content/x28"                     # donde se descomprime el bundle (local)
UNIDADES_POR_HORA = None                  # VERIFICAR en su cuenta de Colab (consumo de la A100 en unidades/hora). No hay valor por defecto: sin el no se proyectan unidades.
PROCESOS_POR_GPU = 1                      # 2 solo si la prueba de concurrencia (celda 5) muestra ganancia de rendimiento
MAX_PRIORIDAD = 1                         # {"1 = 3 semillas (4001-4003); suba a 2 despues de ver el costo" if piezo else "empiece con 1; suba 2, 3, ... segun costo"}
TOPE_DESPACHO_H = 11.0                    # horas de esta sesion tras las cuales NO se lanza trabajo nuevo
TOPE_KILL_H = 11.5                        # horas tras las cuales se MATAN los trabajos en curso
MAX_UNIDADES = None                       # opcional: tope de gasto en unidades de computo (exige UNIDADES_POR_HORA)
SOLO_PROBLEMA = ""                       # p. ej. "dfg" para correr solo ese problema del manifiesto (vacio = todos los de la prioridad)
PASOS_PRUEBA = 300                        # pasos de Adam de la prueba de tiempo (200-500)'''))
    c.append(("code", '''# 2. Montar Drive y verificar la GPU
from google.colab import drive
drive.mount("/content/drive")
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
assert ("A100" in name) or ("T4" in name), f"GPU no soportada ({name.strip()}): use T4 o A100 y reinicie."'''))
    c.append(("code", f'''# 3. Descomprimir el bundle (sin internet salvo pip) y verificar SHA256
import hashlib, os, shutil, zipfile, subprocess, sys, json
from pathlib import Path
z = Path(ZIP_EN_DRIVE); assert z.exists(), f"no existe {{z}}: suba el zip a Drive"
h = hashlib.sha256(z.read_bytes()).hexdigest(); print("SHA256 del zip:", h)
assert (not SHA256_ESPERADO) or h == SHA256_ESPERADO.strip().lower(), "el SHA256 del zip NO coincide"
shutil.rmtree(WORK, ignore_errors=True); zipfile.ZipFile(z).extractall(WORK)
bad = []
for line in Path(WORK, "SHA256SUMS.txt").read_text().splitlines():
    digest, name = line.split("  ", 1)
    if hashlib.sha256(Path(WORK, name).read_bytes()).hexdigest() != digest: bad.append(name)
assert not bad, f"archivos alterados: {{bad}}"
print("bundle verificado:", len(Path(WORK, "SHA256SUMS.txt").read_text().splitlines()), "archivos")
# dependencias: torch y numpy ya vienen en Colab; wandb solo si falta (pip es lo unico que usa internet). NO se hace wandb.login.
try:
    import wandb
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "wandb"], check=False)
RUNNER = f"{{WORK}}/colab_runner.py"; MANIFEST = f"{{WORK}}/{manifest}"; BUNDLE = f"{{WORK}}/{bundle}"
os.environ.pop("WANDB_API_KEY", None); os.environ["WANDB_MODE"] = "offline"
Path(OUT).mkdir(parents=True, exist_ok=True)'''))
    c.append(("markdown", "## 4. Prueba de tiempo en la A100 (primero, antes de gastar)\nCorre unos cientos de pasos de " + ("B, E e I en converso y directo, float64" if piezo else "los brazos B/J de DFG, NS, Burgers y AC") +
              " en un proceso aparte por prueba, y escribe `timing_probe.json` (s/paso y s/recalculo reales), `projection.json` y la tabla de horas por prioridad. "
              "La tasa de unidades/hora la pone usted (celda de parametros); la ganancia de concurrencia se mide con 2 procesos simultaneos."))
    c.append(("code", '''# 5. PRUEBA DE TIEMPO + proyeccion de costo
rate = f"--units-per-hour {UNIDADES_POR_HORA}" if UNIDADES_POR_HORA else ""
!python {RUNNER} probe --manifest {MANIFEST} --bundle {BUNDLE} --out {OUT} --steps {PASOS_PRUEBA} --concurrency 2 --scratch /content/x28_probe {rate}
# Lea la tabla: horas de pared por prioridad y (si declaro la tasa) unidades de computo acumuladas. Decida MAX_PRIORIDAD (y PROCESOS_POR_GPU) con eso.'''))
    c.append(("code", '''# 6. LANZAR (deje la pestana abierta; si se corta, vaya a la celda 7). Crea OUT/STOP para detener el despacho de forma ordenada desde otra pestana de Drive.
extra = ""
if UNIDADES_POR_HORA and MAX_UNIDADES: extra += f" --max-compute-units {MAX_UNIDADES} --units-per-hour {UNIDADES_POR_HORA}"
!python {RUNNER} run --manifest {MANIFEST} --bundle {BUNDLE} --out {OUT} --scratch {SCRATCH} --gpus 0 --jobs-per-gpu {PROCESOS_POR_GPU} --max-priority {MAX_PRIORIDAD} {('--only-problem ' + SOLO_PROBLEMA) if SOLO_PROBLEMA else ''} --cutoff-hours {TOPE_DESPACHO_H} --hard-cap-hours {TOPE_KILL_H} --fit-check {extra}'''))
    c.append(("code", '''# 7. REANUDAR tras un corte (nueva sesion: repita las celdas 1-3, no hace falta repetir la 5). Salta todo lo que ya esta en OUT; los trabajos que se cortaron se repiten desde cero.
!python {RUNNER} status --manifest {MANIFEST} --out {OUT}
!python {RUNNER} run --manifest {MANIFEST} --bundle {BUNDLE} --out {OUT} --scratch {SCRATCH} --gpus 0 --jobs-per-gpu {PROCESOS_POR_GPU} --max-priority {MAX_PRIORIDAD} {('--only-problem ' + SOLO_PROBLEMA) if SOLO_PROBLEMA else ''} --cutoff-hours {TOPE_DESPACHO_H} --hard-cap-hours {TOPE_KILL_H} --fit-check'''))
    c.append(("code", f'''# 8. EMPAQUETAR resultados a Drive (zip con JSON, npz, ledger, logs y carpetas wandb offline) y mostrar su SHA256
import time
stamp = time.strftime("%Y%m%d-%H%M")
dest = Path(OUT).parent / f"resultados_{kind}_{{stamp}}"
shutil.make_archive(str(dest), "zip", OUT)
zf = Path(str(dest) + ".zip"); print(zf, round(zf.stat().st_size / 2**20, 1), "MB")
print("SHA256:", hashlib.sha256(zf.read_bytes()).hexdigest())
!python {{RUNNER}} status --manifest {{MANIFEST}} --out {{OUT}}'''))
    c.append(("markdown", """## 9. Despues (en su PC, no aqui)
1. Descargue `resultados_*.zip` de Drive y descomprimalo en una carpeta bajo `experiments/mechanisms/EXP-0028-failure-attribution/results/`.
2. Sincronice W&B con la clave del `.env` solo en memoria: `python wandb_tools/sync_wandb.py <carpeta>` (dry-run) y luego `--apply`.
3. **Desconecte y elimine el entorno de ejecucion** (Entorno de ejecucion > Cancelar conexion y eliminar) para dejar de consumir unidades."""))
    return nb(c)


def main():
    for name, m in (("manifest_piezo.json", piezo_manifest()), ("manifest_blocks.json", blocks_manifest()), ("manifest_smoke.json", smoke_manifest()), ("manifest_smoke_blocks.json", smoke_blocks_manifest())):
        (HERE / name).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(name, len(m["jobs"]), "trabajos")
    for kind in ("piezo", "blocks"):
        (HERE / f"{kind}_colab.ipynb").write_text(json.dumps(notebook(kind), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    b = json.loads((HERE / "manifest_blocks.json").read_text(encoding="utf-8"))
    from collections import Counter
    print("bloques por prioridad:", dict(Counter(j["priority"] for j in b["jobs"])))
    print("horas T4 por prioridad:", {p: round(sum(j["est_s_t4"] for j in b["jobs"] if j["priority"] == p) / 3600, 1) for p in sorted({j["priority"] for j in b["jobs"]})})


if __name__ == "__main__":
    main()
