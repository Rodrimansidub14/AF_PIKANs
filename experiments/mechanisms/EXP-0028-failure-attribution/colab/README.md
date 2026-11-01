# EXP-0028 en Google Colab (A100, pago por uso)

Lanzador portátil para terminar lo que no cabe en la cuota de Kaggle: **piezo** (50 corridas) y los **bloques diferidos** (DFG 1e, NS 1e, rebanada con artefactos, Burgers 1d/1e). Nada de esto se ha ejecutado en la nube; solo humo en CPU.

**Reglas fijas:** no se escribe ninguna clave en los cuadernos ni en el zip (el constructor aborta si encuentra una); W&B va siempre OFFLINE; internet solo para `pip install wandb` si falta; no se toca `evals/`, `registry/claims.jsonl` ni lo subido a Kaggle.

## Archivos
| archivo | para qué |
|---|---|
| `colab_runner.py` | despachador: `run`, `probe` (prueba de tiempo), `project`, `status`. Reanudable, ledger, topes |
| `manifest_piezo.json` | 50 trabajos: {converso, directo} × {B,F,J,E,I} × semillas 4001-4005. Prioridad 1 = semillas 4001-4003 (30 trabajos); prioridad 2 = 4004-4005 (20) |
| `manifest_blocks.json` | 104 trabajos de bloques; prioridades 1 NS 1e (4104-4105) y DFG 1e (4101-4103), 2 DFG 1e (4104-4105), 3 rebanada con artefactos (4201-4202), 4 Burgers 1d (4103-4105), 5 Burgers 1e |
| `piezo_colab.ipynb`, `blocks_colab.ipynb` | cuadernos para Colab |
| `build_colab_bundle.py` | arma el zip (código + datos + SHA256) en local |
| `make_artifacts.py` | regenera manifiestos y cuadernos |
| `manifest_smoke*.json` | humo en CPU (no es ciencia) |

## Pasos exactos
**En su PC (una vez)**
1. `python colab/make_artifacts.py` (solo si cambió algo) y luego `python colab/build_colab_bundle.py`. Imprime el **SHA256** del zip y lo guarda en `colab/dist/exp0028_colab_bundle.zip.sha256`. El zip lleva `bundle_piezo/` y `bundle_blocks/` por separado (usan versiones distintas de `residual_terms.py`).
2. Suba a su Google Drive, en `Mi unidad/x28_colab/`: `exp0028_colab_bundle.zip`, `piezo_colab.ipynb` y `blocks_colab.ipynb`.

**En Colab (cada sesión)**
3. Abra el cuaderno desde Drive. Entorno de ejecución > Cambiar tipo > **GPU A100**. Active el **pago por uso** en su cuenta de Colab. Un cuaderno a la vez (una sola A100).
4. Celda 1 (parámetros): pegue el SHA256 esperado si quiere. `UNIDADES_POR_HORA`: **déjela en `None` hasta verificarla en su cuenta** (el consumo de la A100 en unidades/hora lo muestra Colab; aquí no se asume). `MAX_PRIORIDAD = 1`.
5. Celdas 2-3: montar Drive, comprobar `nvidia-smi` (aborta si no es A100), descomprimir y verificar SHA256.
6. Celda 5, **prueba de tiempo (siempre primero)**: corre ~300 pasos de Adam por brazo (piezo: B, E, I en converso y directo, float64; bloques: DFG B/J, NS J, Burgers J, AC J) en procesos separados, mide además 2 procesos simultáneos en la misma GPU, y escribe `timing_probe.json`, `projection.json` y la tabla de **horas de pared y unidades por prioridad** (con la tasa que usted declare). Ahí decide.
7. **Decida cuántas semillas**: `MAX_PRIORIDAD = 1` (piezo: 3 semillas, 4001-4003, ambos problemas) y súbala a 2 si el costo lo permite. Puede parar tras 3 semillas sin perder nada: las semillas se completan enteras (ambos problemas, B,F,J,E,I) antes de pasar a la siguiente.
8. Celda 6: lanzar. Topes de la celda 1: `TOPE_DESPACHO_H` (sin trabajo nuevo) y `TOPE_KILL_H` (se matan los trabajos en curso; ponga un margen para que lo que corre termine), `MAX_UNIDADES` (tope de gasto, opcional). `--fit-check` evita lanzar un trabajo que, según la proyección, no cabe antes del kill. Para detener de forma ordenada: cree un archivo vacío llamado `STOP` dentro de la carpeta de salida en Drive (los trabajos en curso terminan, no se lanzan más).
9. Celda 8: empaquetar: zip de resultados en Drive (`resultados_*.zip`) + su SHA256. Luego **Entorno de ejecución > Cancelar conexión y eliminar** para dejar de consumir unidades.
10. Descargue el zip de resultados.

## Cómo reanudar si se corta
Nueva sesión: repita las celdas 1-3 (no la 5) y corra la celda 7 (`status` + `run`). Todo trabajo cuyos archivos ya estén en la carpeta de salida (Drive) se salta; el que se cortó a medias se repite **desde cero** (un trabajo = una corrida de 1-3 h; no hay puntos de control a mitad de corrida). Un trabajo fallido se reintenta hasta `--max-attempts` (2) sesiones. Cada trabajo corre en una carpeta **local** y solo se copia a Drive si terminó bien; el JSON se copia al final (es la marca de «hecho»), así que una copia interrumpida no cuenta como hecha.

## Qué queda en la carpeta de salida (`salida_piezo/` o `salida_blocks/`)
- `<grupo>/…json` (un JSON por corrida; piezo: `piezo/piezo-<problema>-<brazo>-seed<s>.json`), `…-artifacts.npz` (piezo y rebanada), `fem_*_reference.npz`, `wandb/offline-run-*` (W&B OFFLINE).
- `ledger.jsonl` (un evento por línea: `session_start`, `precheck`, `start`, `end`/`killed`, `skipped`, `session_end`; con GPU, comando, segundos, código de salida, estado), `progress.json`, `logs/` (un log por trabajo; los fallidos como `<id>.failed-N.log`), `checks/` (pruebas B == ID), `timing_probe.json`, `projection.json`, `projection_summary.json`.
- **Guarde todo esto**: es lo que se analiza (los JSON y npz), lo que prueba qué se corrió (ledger, checks, logs, SHA256 del zip) y lo que se sube a W&B.

## Pruebas previas B == ID
Como en los kernels de Kaggle, antes del primer trabajo de cada problema el despachador corre la prueba B == ID en la GPU (`run_piezo.py --check-id`, `run_phase1e.py --check-id` para NS/DFG, `run_slice.py --check` para la rebanada). Si falla, **no se despacha** ese problema (queda `not_started_check_failed` en el ledger). Las de piezo y de DFG/NS duran minutos. Se saltan con `--skip-prechecks` solo para humo.

## Sincronizar W&B después (en su PC, no en Colab)
1. Descomprima `resultados_*.zip` en `experiments/mechanisms/EXP-0028-failure-attribution/results/colab-<fecha>/`.
2. `python wandb_tools/sync_wandb.py results/colab-<fecha>` (por defecto es dry-run) y, si la lista es correcta, `--apply`. La clave se lee del `.env` dentro del proceso; nunca va al cuaderno ni a Colab.

## Notas y límites
- **Costo**: la proyección sale de lo medido en la A100 durante la prueba; las unidades solo se calculan con la tasa que usted declare. Las cifras T4 del repositorio (`piezo/budget_piezo.md`: ~121 GPU-h T4 para las 50 corridas de piezo) son de referencia, no una promesa para la A100.
- **Piezo, brazo I**: el chunk 16 de `GramPreconditioner('I')` en float64 solo está probado en CPU con chunk 2 (riesgo 1 de `budget_piezo.md`). La prueba de tiempo lo mide primero; si falla en la A100, la proyección usa E como respaldo y lo avisa. Si las corridas de I fallan, relance con `--pre-chunk 2` agregando `"extra_args": ["--warmup-steps","50","--pre-chunk","2"]` a esos trabajos del manifiesto.
- **Varios procesos por GPU** (`PROCESOS_POR_GPU = 2`): solo si la prueba de concurrencia muestra ganancia (> 1.3×); no estaba medido antes.
- **Semillas pareadas**: la misma semilla da el mismo modelo inicial y los mismos puntos para los cinco brazos; no cambie el orden ni mezcle sesiones con `extra_args` distintos para una misma semilla.
- **Burgers 1e** necesita el B de 1d de las mismas semillas (4101-4105): 1d `bu1` (4101-4102) ya corrió en Kaggle; la prioridad 4 corre `bu2/bu3`. Si trae los resultados de Kaggle a `salida_blocks/p1d-burgers/` esos trabajos se saltan solos.
- **Qué no está en el manifiesto de bloques**: AC/KPP 1e (kernel `acf`, entra en la cuota de Kaggle) y NS 1e 4101-4103 (ya corrió, `results/phase1e-ns1`). Para agregarlos basta añadir trabajos al JSON (`recipe: phase1e`, `group`, `problem`, `arm`, `seed`, `priority`, `est_s_t4`).
- El reporte de humo en CPU (`manifest_smoke*.json`) valida el despachador y las órdenes de comando; no valida ciencia.
