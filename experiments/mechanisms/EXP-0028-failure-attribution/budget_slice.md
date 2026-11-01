# EXP-0028 rebanada con artefactos (Enmienda v1.6) — receta media, presupuesto y W&B (2026-10-24). NO LANZADO, NO SUBIDO

Código nuevo, todo en `artifact_slice/` y `kaggle/*_slice*` (nada de lo ya subido ni de los ejecutores 1/1c/1c-dfg/1d/1e se modificó; `run_slice.py` los **importa y envuelve en tiempo de ejecución**): `artifacts_core.py` (módulo genérico), `run_slice.py`, `kaggle_block_slice.py`, `kaggle/build_bundle_slice.py`, `analyze_slice.py`, `attach_artifacts_local.py`, `budget_slice_calc.py`, `recipe_analysis.py` (deriva la receta con datos). Bundle: dataset NUEVO `rodrimandub/af-pikans-exp0028-bundle-slice` (copia plana de `dataset_1e/` + 2 archivos), kernels `rodrimandub/af-pikans-exp-0028-slice-{ns,dfg,ackpp,bu}`.

Diseño: problemas AC d=1e-3, Burgers RBF48, KPP, NS2d-CG, DFG; brazos B,F,J,E,I; **semillas 4201, 4202** (verificado por búsqueda en `experiments/`, `registry/`, `src/`, `docs/`, `tests/`, `evals/`: cero apariciones; las usadas son 4001–4010 y 4101–4105; `run_slice.py` rechaza otras salvo humo). 5 problemas × 5 brazos × 2 semillas = **50 corridas**. Piezo lo hace otro agente (`piezo/artifacts.py`); el esquema de claves del `.npz` es el mismo.

## 1. Receta media, definida con datos (`artifact_slice/recipe_analysis.py`, reproduce todo)

Método. (i) De los `hist` (pérdida de entrenamiento cada 100 pasos) de los B se toma la mediana entre semillas, suavizada con mediana móvil de 5 puntos; **M_2x** = primer paso en que la pérdida de B queda ≤ 2× la de B al final de la receta completa (AC/KPP 1d, NS 1e, Burgers 1d; DFG: ver abajo). (ii) Se compara el efecto e_X/e_B al final de Adam entre la receta corta y la completa. (iii) L-BFGS = 500 iteraciones (orden del investigador).

| problema | B corta: pasos, pérdida | B completa: pasos, pérdida (suav.) | M_2x (1.er paso ≤ 2×) | M elegido |
|---|---|---|---|---|
| AC d=1e-3 | 4 000, 2.5e-4 (6.9× la completa) | 40 000, 3.6e-5 (n=5) | 9 600 | **10 000** |
| KPP | 4 000, 2.5e-5 (4.2×) | 40 000, 6.0e-6 (n=5) | 6 500 (1.er múltiplo de 1000 que no salta: 8 000) | **8 000** |
| Burgers RBF48 | 2 000, 5.1e-3 (11×) | 24 000, 4.7e-4 (**n=2**, ruidoso) | 4 300 (1.er múltiplo estable: 9 000) | **10 000** (pool 16 384 como la completa) |
| NS2d-CG | 2 000, 3.5e-4 (5.5×) | 20 000, 6.4e-5 (n=3) | 7 100 (1.er múltiplo estable: 10 000) | **10 000** |
| DFG | 2 000, 1.4e-2 (10 semillas) | **no existe** (1e-DFG no corrió) | criterio vacío: la forma de la pérdida de `dfg_long` (RBF16, 20 000 Adam; otra formulación) ya está a 1.4× en el paso 2 000 | **6 000** (juicio, ver nota) |

Efecto E/B y F/B (mediana pareada del error primario) a fin de Adam, corta → completa, y tras L-BFGS:

| problema | fin de Adam E | fin de Adam F | tras L-BFGS E | tras L-BFGS F | lectura para la receta media |
|---|---|---|---|---|---|
| AC | 0.283 → 0.212 | 0.292 → 0.281 | 0.162 → 0.389 | 0.352 → 0.812 | A fin de Adam el efecto ya está en 4 000 pasos y casi no cambia hasta 40 000: M = 10 000 lo reproduce. **Tras L-BFGS no**: 500 it. dejan a B lejos de su meseta (la pérdida de B a 500 it. es 6× la de la meseta; la de E, 1.2×; `lbfgs_traj` de 1d), así que el efecto post-L-BFGS de la rebanada sobrestima al de la completa (≈ 0.16–0.2 en vez de 0.39). Límite declarado. |
| KPP | 0.762 → 0.077 | 1.094 → 0.429 | 0.767 → 0.119 | 1.835 → 0.640 | El efecto cambia por un orden de magnitud entre 4 000 y 40 000 pasos, pero B empeora con la receta larga (error 0.049 → 0.132): sin curva intermedia no hay forma de saber si 8 000 se parece a 40 000. **El caso menos fiel; las instantáneas de error cada 1 000 pasos son justo para esto.** |
| Burgers | 1.144 → 0.435 (n=2) | 0.625 → 0.258 (n=2) | 0.240 → 1.369 (n=2) | 0.285 → 2.045 (n=2) | E es "dependiente de semilla" en la completa (n=2 no alcanza); el signo cambia entre recetas. M = 10 000 queda en la zona intermedia; la rebanada con 2 semillas solo ilustra. |
| NS | 0.641 → 0.960 | 0.676 → 0.970 | 0.792 → 0.981 | 0.830 → 0.984 | El efecto **se disipa con el presupuesto**: la razón de pérdidas E/B se queda en ≈ 0.4 en todos los pasos, pero el error pasa de 0.64 a 0.96 porque B alcanza la meseta común (error ≈ 0.066). Interpolando en log(pérdida de B) entre los dos puntos medidos, a M = 10 000 se espera E/B ≈ 0.84 (F 0.85, I 0.81) frente a 0.96 de la completa: **M = 10 000 queda en la transición, no en el valor final**. Las instantáneas en 2 000, 5 000 y 10 000 miden la curva; reproducirla hasta 0.96 exige ≈ 20 000 pasos (+1.7 GPU-h; alternativa `--adam 20000`). |
| DFG | 0.834 → sin dato | 0.800 → sin dato | 0.815 → sin dato | 0.567 → sin dato | Sin receta completa de DFG con los brazos. El error de B a fin de Adam en la corta es u 0.645 (2 000 pasos) y en `dfg_long` (20 000 pasos, otra formulación) 0.461; tras L-BFGS 500 it. de la corta ya llega a u 0.167, el nivel de `dfg_long` (0.191 con ≈ 4 000 it.). La pérdida satura temprano pero el error no; se triplica la corta (6 000) por costo (0.077–0.082 s/paso) y se deja la curva a las instantáneas (2 000, 3 000, 6 000). |

**Límite de la inferencia (declarado).** Adam con lr constante: el estado en el paso M de una corrida larga es el de una corrida de M pasos (salvo los recálculos de J/E/I, que caen en múltiplos de 1 000 y son los mismos). Por eso el runner toma instantáneas y se lee la curva efecto-vs-presupuesto en **una** corrida. Pero (a) lo único medido a pasos intermedios es la pérdida, no el error: el error solo existe en los extremos corto y completo, y el mapeo pérdida → efecto no es monótono (NS: pérdida E/B constante, error E/B de 0.64 a 0.96); (b) KPP y Burgers (n=2) tienen la completa ruidosa; (c) L-BFGS por 500 iteraciones no es la meseta de la completa (AC, Burgers); (d) DFG no tiene receta completa con brazos. La receta media es un compromiso de costo con esas cuatro lagunas, no una réplica de la completa; las instantáneas de L-BFGS (100, 250, 500 it.) acotan (c).

## 2. Qué se guarda por corrida (`artifacts_core.py`; float32, `np.savez_compressed`, tope declarado 25 MB por corrida)

Un `<problema>-<brazo>-seed<semilla>-artifacts.npz` por corrida (+ el JSON habitual del ejecutor) y un `ref_<problema>.npz` por problema (malla de evaluación + referencia, **una vez**; Burgers submuestreado en x a ≤ 40 000 puntos).
- **Estado**: `state_dict` completo (coeficientes c′, base, centros, anchos, `pre_P`, `lo/hi`, escalas de salida) + coeficientes efectivos `c_eff` en cada instantánea.
- **Campos** en la malla de evaluación (AC/KPP/Burgers: u en la rejilla del problema; NS: u,v,p en los 12 910 puntos de la referencia; DFG: u,v,p en los 9 203 de la malla FEM, presión con ancla) vs referencia, **error vs paso** (`score_hist`: la puntuación completa del problema en cada instantánea).
- **Residuos** por punto y por término (AC/KPP/Burgers: `pde` sobre el pool; NS: momentum_x/y, continuidad y los 5 de borde; DFG: los 11 términos de `DFGTerms`) sobre el pool (≤ 16 384 puntos por clave), términos de la pérdida y **normas de gradiente por término × tensor** en un subconjunto fijo de evaluación.
- **Historia** cada 100 pasos: pérdida total, por término (NS/DFG: 8; escalar: 1) y norma de gradiente por tensor (`layers.k.coefficients|base`).
- **Mecanismo**: en cada recálculo de J/E/I, paso, **traza de B por arista** (o entrada), cond de cada bloque normalizado, valores singulares de P instalada; en cada instantánea completa, espectro (SVD) y cond de P por bloque, **ocupación** de cada base RBF por capa (fracción de evaluaciones con activación > 0.1) y **mapa de asentamiento** (cambio relativo de c_eff por arista entre instantáneas consecutivas).
- **Meta**: semilla, versiones (python, torch, numpy, wandb), commit de git, SHA-256 del código (ejecutores, `gram_preconditioning`, `residual_terms`, `sweep`, `dfg`, `run_slice`, `artifacts_core`) y de los datos de referencia, configuración completa, calendario.
- Calendario: instantáneas **completas** en {0, 2 000, M/2, M} ∪ {cada 5 000} y al terminar L-BFGS; **livianas** (error, c_eff, términos, asentamiento) cada 1 000 pasos y a 100/250 it. de L-BFGS.
- **Solo lectura**: no cambia parámetros, no consume RNG, no altera la trayectoria. Puntuar durante el entrenamiento es monitoreo: nada del entrenamiento lee la referencia. El tiempo de las instantáneas entra en `adam_seconds`/`hist.seconds` (se informa `snapshot_seconds`); los tiempos de la rebanada no son comparables con los de 1d/1e.

Tamaño (medido en las pruebas de humo y extrapolado por forma de arreglos): ≈ 1.7 MB (AC, KPP), 2.2 (Burgers), 2.5 (NS), 2.6 (DFG) por corrida J/E/I; B menos (sin `pre_P` ni recálculos). **Total ≈ 110 MB** (+ ≈ 3 MB de carpetas W&B offline); tope declarado 25 MB/corrida (si se acerca, se omiten campos/residuos de instantáneas intermedias, nunca la final). `analyze_slice.py` imprime el tamaño real.

## 3. Weights & Biases

**Auditoría (hoy).**
- EXP-0023 ronda 1-C: `arnes_round1c.note_run` (wandb 0.28.1 ya instalado en la imagen de Kaggle: Python 3.13.15, torch 2.11.0+cu128; ninguna instalación de ruedas) escribió **una nota por corrida** en modo OFFLINE (`wandb/offline-run-*` dentro de la salida + `*.note.json`): 60 notas con `"wandb_offline_logged"`, una sola llamada `wandb.log` con **métricas escalares finales**; sin series temporales, sin artefactos. Sincronizadas después (`.wandb.synced`).
- EXP-0028 (fases 1, 1b, 1c, 1c-dfg, 1d, 1e y sus `kaggle_block_*`/`build_bundle_*`): **no registran nada en W&B**; importan `arnes_round1c` (que define `note_run`) pero ningún ejecutor lo llama (verificado por búsqueda: cero llamadas a `note_run(` y cero apariciones de `wandb` en `run_phase*.py`, `kaggle_block*.py`, `analyze_*.py`). Solo JSON.
- Entorno: Kaggle (imagen de EXP-0023 1-C) trae `wandb 0.28.1`; Python312 local trae `wandb 0.31.0` (ya instalado; no hizo falta `pip --user`). Cliente 0.31 sincroniza runs de 0.28 sin problema conocido (no probado por falta de sesión).

**Propuesto e implementado (rebanada).** Proyecto **`AF_PIKANS`**, entity `man22611-uvg` (los del `.env`; el código no lee el `.env`, solo los fija como constantes públicas); se distingue por `group = "EXP-0028-artifact-slice"`, `job_type = <problema>`, `tags = [<problema>, <brazo>, seed<semilla>, medium, artifact-slice]`, `name/id = <problema>-<brazo>-seed<semilla>`. Modo **OFFLINE** forzado (`WANDB_MODE=offline`, `mode="offline"`), `dir` = salida del kernel, `WANDB_DATA_DIR/CACHE_DIR/CONFIG_DIR` también dentro de la salida (nada en `~`). **No lee ni escribe `WANDB_API_KEY`, no hace `wandb.login`, no toca el `.env`**; `kaggle_block_slice.py` además quita `WANDB_API_KEY` del entorno de los procesos hijo. Ninguna clave en el bundle (`build_bundle_slice.py` lo comprueba por búsqueda de texto).
- Config completa (receta, brazo, semilla, calendario, hashes, versiones).
- Series temporales: `loss`, `term/<t>`, `gradnorm/<tensor>` cada 100 pasos; `score/*` (error en la malla) y `eval_term/*` en cada instantánea; `P_cond_median` y `refresh_seconds` en cada recálculo.
- Artefacto: `wandb.Artifact(<tag>-artifacts, type=run-state)` con metadatos (archivo, MB, SHA-256) y el `.npz` **por referencia** (URI + digest): en modo offline los bytes de `add_file` quedan en un staging fuera de lo que Kaggle devuelve, así que adjuntarlos allí se perdería. Los bytes viajan en la salida del kernel y se suben con `attach_artifacts_local.py` (opcional, después del `wandb sync`).
- **Sincronizar (acto del investigador, en local, nunca en Kaggle):** descargar la salida (`kaggle kernels output rodrimandub/af-pikans-exp-0028-slice-ns -p OUT_ns`), luego `"C:\Program Files\Python312\python.exe" experiments\mechanisms\EXP-0028-failure-attribution\wandb_tools\sync_wandb.py OUT_ns` (dry-run) y con `--apply` (herramienta del otro agente: lee la clave del `.env` en memoria). Equivalente manual: `wandb sync --project AF_PIKANS --entity man22611-uvg OUT_ns\wandb\offline-run-*`. Para subir los `.npz`: `python artifact_slice\attach_artifacts_local.py OUT_ns --apply` (usa la sesión de W&B del entorno; no lee el `.env`).

## 4. Presupuesto (T4, tiempos medidos; `artifact_slice/budget_slice_calc.py` reproduce las tablas)

Fuentes: NS Adam 0.0535 s/paso (B), 0.0555 (F), 0.0561 (J/E/I), recálculo 5.4 s, L-BFGS 0.0776 s/it. (1c-t1, `timing_ns.json`); DFG B 0.0771, F 0.0801, J 0.0823, E 0.0801, I 0.0826 s/paso, recálculo 6.2–7.0 s, L-BFGS 0.123 s/it. (1c-dfg); AC 0.0196–0.0213 s/paso, KPP 0.0204–0.0225 (fase 1, 1d); Burgers 0.0782 s/paso a 16 384 puntos y recálculo 3.3–4.9 s (`budget_1d.md`). **Único dato NO medido: el sobrecosto de las instantáneas** (estimado 40 s NS, 80 s DFG, 25 s AC/KPP, 50 s Burgers por corrida: ~4 instantáneas completas + ~10 livianas + historia + W&B); el primer kernel (`ns`) lo mide.

| corrida (s) | B | F | J | E | I |
|---|---|---|---|---|---|
| NS (M=10 000) | 654 | 674 | 739 | 739 | 739 |
| DFG (M=6 000) | 644 | 662 | 720 | 706 | 726 |
| AC (M=10 000) | 261 | 268 | 277 | 278 | 278 |
| KPP (M=8 000) | 228 | 235 | 242 | 243 | 245 |
| Burgers (M=10 000, 16 384 pts) | 907 | 930 | 943 | 946 | 961 |

Kernels (T4×2, 2 ranuras, despacho ≤ 3.4 h, kill 3.9 h; reanudables), **en orden de prioridad**:

| kernel (`rodrimandub/af-pikans-exp-0028-slice-…`) | prio. | contenido | corridas | sesión (pared + prueba) | margen al kill | acumulado |
|---|---|---|---|---|---|---|
| `ns` | 1 | NS B,F,J,E,I × 4201–4202 | 10 | 1.07 h | 2.83 h | 1.07 h |
| `dfg` | 1 | DFG × 4201–4202 | 10 | 1.05 h | 2.85 h | 2.13 h |
| `ackpp` | 2 | AC y luego KPP × 4201–4202 | 20 | 0.77 h | 3.13 h | 2.90 h |
| `bu` | 3 | Burgers × 4201–4202 | 10 | 1.35 h | 2.55 h | 4.25 h |

**Total: 7.9 GPU-h (dos T4 ocupadas) = 4.25 h de sesión (cuota), ±25 % por el sobrecosto de instantáneas no medido** (≈ 3.2–5.3 h). La cuota restante (~14 h menos lo que corra en Kaggle) alcanza con holgura; NS+DFG (prioridad 1) = 2.1 h; con AC+KPP, 2.9 h; Burgers (prioridad 3) agrega 1.35 h y es lo primero que se recorta. Alternativa NS con 20 000 pasos: +1.7 GPU-h (+0.85 h de sesión).

## 5. Pruebas hechas (CPU; nada se lanzó ni se subió)

PRUEBAS_PENDIENTES

## 6. Comandos (NINGUNO de subida/lanzamiento ejecutado)

```
PY="C:\Program Files\Python312\python.exe"; cd C:\Users\rodri\Documents\AF_PIKANs\experiments\mechanisms\EXP-0028-failure-attribution
# derivar la receta (lectura) y el presupuesto
$PY artifact_slice/recipe_analysis.py ; $PY artifact_slice/budget_slice_calc.py
# construir (ya hecho; reescribe solo kaggle/dataset_slice, kernel_slice_*, approval_slice.json, SHA256SUMS_slice.txt)
$PY kaggle/build_bundle_slice.py
# humo de una corrida por problema (CPU)
$PY artifact_slice/run_slice.py --problem ns --arms B,E --seeds 4201 --out OUT --adam 12 --lbfgs-iters 5 --pre-every 6 --pre-n 64 --smoke-pool 128 --light-every 6
$PY artifact_slice/run_slice.py --problem ac --check --out OUT --adam 12 --lbfgs-iters 5 --pre-every 6 --pre-n 64
# subida y lanzamiento (a ejecutar solo por orden del investigador; en este orden de prioridad)
kaggle datasets create -p kaggle/dataset_slice
kaggle kernels push -p kaggle/kernel_slice_ns ; kaggle kernels push -p kaggle/kernel_slice_dfg ; kaggle kernels push -p kaggle/kernel_slice_ackpp ; kaggle kernels push -p kaggle/kernel_slice_bu
# tras descargar las salidas
$PY artifact_slice/analyze_slice.py --dir OUT_ns --dir OUT_dfg --dir OUT_ackpp --dir OUT_bu --out results/slice
$PY wandb_tools/sync_wandb.py OUT_ns            # dry-run; --apply para sincronizar (acto del investigador)
```
