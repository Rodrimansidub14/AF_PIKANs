# EXP-0028, bloque final de piezo (converso y directo): presupuesto y comandos. NO LANZADO ni subido

Receta (paper): Adam 1e-3, 20 000 pasos, float64, sin L-BFGS, 2×2048 interiores + 256 por frontera/interfaz, remuestreo cada 200; RBF16 de 11 bases (4-16-16-8) con las entradas/salidas del paper; brazos B, F, J, E, I; semillas pareadas **20260728-20260732** (cohorte del paper) **por problema**; recálculo del bloque antes del paso 1 y cada 1000 pasos (20 por corrida de J/E/I); GN con 2048 puntos por clave del pool actual (3 328 puntos: interior 4 096 → 2 048, cuatro fronteras y la interfaz 256). 50 corridas (25 por problema) + 2 de referencia **MLP-paper** (MLP 3×50 SiLU, semilla 20260728, una por problema, baja prioridad, bloque `k26`; ≈ 4 000 s converso y ≈ 2 700 s directo en T4, +1.9 GPU-h y un lanzamiento de ≈ 1.1 h). Registro de error (u, v, φ, flecha, φ del electrodo) cada 500 pasos. Métrica primaria `l2re_u` al paso 20 000.

## Tiempos: qué es medido y qué es estimado
| Concepto | CPU (8 núcleos, 2-6 hilos, **máquina compartida/cargada**) | T4 float64 |
|---|---|---|
| Adam, s/paso, RBF16 converso | 3.9-4.0 | **0.47 medido** (EXP-0009: 9 222-9 510 s / 20 000, `daniel_arm.py`) |
| Adam, s/paso, RBF16 directo | 2.63 (0.67× el converso) | 0.31 **estimado** (misma razón) |
| Adam, s/paso, MLP 3×50 | 1.57 converso / 1.01 directo | ~0.19-0.20 converso (EXP-0009: ~3 900-4 030 s) |
| Recálculo E, 3 328 pts, chunk 16 | 296-387 s converso / 329 s directo | **estimado** 36-47 s (CPU ÷ 8.2, razón de Adam); rango 36-120 s |
| Recálculo I, chunk 2 | 357 s converso (directo 2 331 s medido con la CPU saturada por otro proceso: descartado) | igual que E ±50 %; **el chunk 2 solo está probado en CPU** (ver riesgo 1) |
| Recálculo J | no medido por separado (≈ E: mismo J, solo cambia la reducción) | ≈ E |
| Puntuación (4 221 pts) | 0.05 s | despreciable |

Supuesto declarado (el mismo del presupuesto de DFG): la razón CPU/T4 del recálculo es la de Adam (8.2×); si el recálculo está limitado por lanzamientos pequeños la ganancia es menor (hasta 3×). La medición real la hace el kernel (`--timing`, bloque `t`, minutos) antes de gastar nada.

## Estimación por corrida (T4×1, float64) y total
| | converso | directo |
|---|---|---|
| B | 9 400 + 60 = 9 460 s | 6 300 + 60 = 6 360 s |
| F (+5 % en el paso) | 9 930 s | 6 680 s |
| J / E / I (+20 × 45 s) | 10 830 s (rango 10 600-12 400) | 7 580 s (7 300-9 000) |
| suma por semilla (B+F+J+E+I) | 51 900 s | 35 600 s |
| 5 semillas | 259 500 s = 72 GPU-h | 178 000 s = 49 GPU-h |

**Total ≈ 437 000 GPU-s = 121 GPU-h**; con dos ranuras (T4×2) ≈ **61 h de pared**, repartidas en **25 lanzamientos** (`k01`…`k25`) más `k26` (referencia MLP-paper) (, dos trabajos por lanzamiento, uno por GPU): los de converso duran ~2.7-3.4 h, los de directo ~1.8-2.5 h. Cada lanzamiento cabe bajo el corte de despacho de 3.4 h y el kill de 3.9 h (peor caso converso J/E/I con recálculo lento: 12 400 s = 3.4 h: al límite). No cabe un trabajo adicional por ranura (dos corridas seguidas de converso suman 5.3 h). Cuota semanal de Kaggle: a verificar antes de lanzar; 61 h de sesión es un orden de magnitud sobre lo usado en 1c/1d.

**Opciones para reducirlo (decisión del investigador, no tomadas aquí):** (a) 3 semillas (4001-4003): 15 lanzamientos, ~37 h; (b) un problema primero (el directo, más barato: 13 lanzamientos, ~26 h); (c) omitir brazos que `predictions.md` declare sin predicción (v1.3 permite correrlos como exploratorios, no es obligatorio); (d) probar dos procesos por GPU en el bloque `t` (no medido). Recortar semillas sacrifica la comparación pareada de 5.

## Riesgos conocidos
1. **El bloque `I` y torch en CPU:** con float64 y 8 salidas, `GramPreconditioner('I')` hace que el proceso muera con segfault (código 139 / 0xC0000005) si el chunk del GN es ≥ 4 *en CPU* (J y E no). Se aísla a la capa con layout por entrada: el GN sin precondicionar y el layout de E funcionan. `run_piezo.py` usa chunk 2 para `I` en CPU y 16 en GPU, **sin verificar en GPU**; el kernel mide `I` con chunks 8, 16 y 2 en `--timing` (J y E primero; el JSON se escribe tras cada fila, así que un cierre del proceso deja lo medido) y las corridas de producción de `I` pueden fallar con `failed` en el ledger. Si falla en GPU, relanzar con `--pre-chunk 2` (la ruta probada) y contar ~8× más llamadas por recálculo.
2. **Sustituto del GN en la carga neta cero (directo):** bajo troceo el bloque es una cota superior del exacto (DIFERENCIAS.md, fila 9).
3. **GN con enrutamiento (converso):** J con el mismo `detach` que ve el optimizador, no el exacto (DIFERENCIAS.md, fila 8).
4. Línea base RBF congelada obtenida con el gancho de instantáneas de `daniel_arm.py` (Adam reiniciado cada 2000 épocas); la receta continua del puerto puede dar otro nivel (DIFERENCIAS.md, fila 12).

## Comandos (NINGUNO ejecutado salvo los de humo y medición en CPU)
```
PY="C:\Program Files\Python312\python.exe"; cd experiments/mechanisms/EXP-0028-failure-attribution
# pruebas (CPU)
$PY -m pytest tests/test_piezo_problem.py tests/test_residual_terms.py -q
# regenerar referencias FEM (scikit-fem 12.0.2; pip install --user scikit-fem==12.0.2)
$PY piezo/make_fem_reference.py
# humo CPU de la validación (hecho: ver smoke_baseline.json) y de una corrida
$PY piezo/smoke_baseline.py 200 100
$PY piezo/run_piezo.py --problems converse,direct --arms B,E --seeds 20260728 --adam 20 --eval-every 10 --smoke-pool 32 --pre-every 10 --pre-n 64 --out OUT
# propiedades a la inicialización (hecho): piezo/properties_piezo.md
$PY piezo/measure_properties_piezo.py --problems converse,direct --seeds 20260728-20260730 && $PY piezo/measure_properties_piezo.py --report

# VALIDACIÓN COMPLETA, a lanzar solo por orden del investigador (GPU, no se ha lanzado)
#  MLP del paper, semilla 20260728, n = 1, 20 000 pasos: debe dar 0.107/0.150/0.045 (converso) y 0.117/0.153/0.067 (directo)
$PY piezo/run_piezo.py --problems converse,direct --arms MLP --seeds 20260728 --adam 20000 --device cuda:0 --out OUT_val
#  línea base RBF16 (B), n = 5, 20 000 pasos, semillas del paper (comparar con 0.092 ± 0.021 en u del converso congelado)
$PY piezo/run_piezo.py --problems converse,direct --arms B --seeds 20260728,20260729,20260730,20260731,20260732 --adam 20000 --device cuda:0 --out OUT_base

# bundle y kernel (construir no sube nada)
$PY piezo/build_bundle_piezo.py --block t        # solo B == ID con calentamiento + medición; --block k01..k26 | all
# kaggle datasets create -p kaggle/dataset_piezo   y   kaggle kernels push -p kaggle/kernel_piezo   (NO ejecutados; requieren orden y predictions.md firmado)
```
Dataset `rodrimandub/af-pikans-exp0028-bundle-piezo`, kernel `rodrimandub/af-pikans-exp-0028-piezo`, topes 3.4 h (despacho) / 3.9 h (kill); antes de despachar el kernel corre `--check-id --warmup-steps 50` (B == ID bit a bit en los dos problemas) y `--timing`, y no despacha si B ≠ ID.

## Resultados del humo en CPU (hechos; `smoke_baseline.py`, `smoke_baseline.json`)
- **MLP del paper, semilla 20260728, 200 pasos de Adam reducido:** la historia de pérdida del puerto iguala la publicada (`results/runs/paper20k_*/loss_*.npy`, corrida de A100) con error relativo máximo **6.4e-13 (converso)** y **1.3e-11 (directo)**. Con los pesos entrenados del paper el puntaje contra el FEM regenerado da 0.107/0.150/0.045 (converso) y 0.117/0.153/0.067 (directo), flecha −29.3/−175.7 µm y φ flotante −2.92 V: las cifras del paper. (Los 20 000 pasos completos del MLP no se han corrido; comando arriba.)
- **RBF16 (B), 100 pasos, semilla 20260728:** en la solución casi nula (u 1.00 / v 1.00 converso; ~0.99-1.02 directo), como cualquier corrida corta de este problema (EXP-0009: no sale de ahí en pocos cientos de pasos). La comparación con **0.092 ± 0.021 (u, converso, n = 5)** exige el comando de 20 000 pasos y n = 5 (no lanzado en GPU).

## Artefactos por corrida y registro W&B (offline)
Cada corrida escribe junto al JSON `piezo-<problema>-<brazo>-seed<s>-artifacts.npz` (comprimido) y, una vez por problema, `fem_<problema>_reference.npz`; módulo reutilizable `piezo/artifacts.py` (`RunRecorder`, `save_run_artifacts(model, prob, tag, out_dir, P, pre, step)`), solo lectura (prueba: la trayectoria es idéntica al bit con y sin registro). Contenido: (1) `state_dict` completo (coeficientes, centros, anchos, base, `pre_P`, c' reparametrizados) en el paso 0, cada 2000 y al final; (2) campos u, v, φ, σ11, σ22, σ12, D1, D2 en la malla 201×21 (float32) cada 5000 y al final; (3) residuos por punto y por término (constitutivos, equilibrio, Gauss, borde) sobre el pool y sobre la malla (interior + sus cuatro bordes), mismos pasos; (4) pérdida total y por término (con peso) y normas de gradiente por tensor (`layers.k.coefficients`, `layers.k.base`) cada 100 pasos; (5) por recálculo: traza por arista y cond de cada bloque B, valores singulares de P; (6) ocupación de cada base RBF por capa (activación > 0.1) en los pasos de campos y asentamiento de coeficientes efectivos por arista cada 1000 pasos; (7) consistencia constitutiva: discrepancia en unidades físicas entre las 5 salidas mixtas y las reconstruidas desde ∂u, ∂v, ∂φ, y su L2 relativo (malla y pool); (8) semilla, versiones, hash SHA-256 del código y los datos, commit de git, configuración. El FEM de referencia solo trae u, v, φ.

**Tamaño (medido en una corrida de humo y extrapolado a los puntos reales):** estado ≈ 0.47 MB por instantánea con `pre_P` (≈ 0.04 MB sin él) × 11; campos + residuos + consistencia + ocupación ≈ 0.7 MB × 4 instantáneas pesadas; recálculos ≈ 23 KB × 20; historias < 0.1 MB. **≈ 8 MB sin comprimir (≈ 5-7 MB comprimido) por corrida con bloque, ≈ 2-3 MB para B**; por kernel (2 trabajos) ≈ 12-15 MB, más ≈ otro tanto en la carpeta `wandb/` offline (copia del artefacto) y los dos FEM (0.2 MB): **≈ 30 MB por kernel, ≈ 0.4 GB los 50 trabajos**: muy por debajo del tope de ~100 MB por kernel, así que no hace falta recortar (si hiciera falta, el orden es (1), (2), (3), (5)). Costo de cómputo: las 4 instantáneas pesadas cuestan ≈ el equivalente de 1-2 recálculos cada una (≈ 1 % de la corrida en T4); las normas de gradiente y la historia son despreciables.

**W&B, siempre OFFLINE:** `wandb.init(project="AF_PIKANS", entity="man22611-uvg", group="EXP-0028-piezo", job_type="converse"|"direct", mode="offline", dir=<salida>)`; no lee ni escribe `WANDB_API_KEY` (los kernels no tienen internet y el bundle no lleva ninguna clave). Registra la configuración completa, pérdida total y por término y normas de gradiente cada 100 pasos, error vs FEM en cada puntuación (cada 500), cond mediana de P en cada recálculo y el `.npz` como `wandb.Artifact` (tipo `run-state`, si pesa ≤ 60 MB). Si `wandb` no está en la imagen del kernel, se omite (estado `unavailable` en el JSON) y los mismos datos quedan en el `.npz`. `--no-artifacts` / `--no-wandb` los apagan.
**Sincronizar (acto manual del investigador, en local, nunca en Kaggle):** bajar la salida del kernel y, con la clave del `.env` cargada solo en memoria, `python wandb_tools/sync_wandb.py <carpeta_de_salida>` (script aparte de otro agente; equivale a `wandb sync <salida>/wandb/offline-run-*` con `WANDB_ENTITY=man22611-uvg WANDB_PROJECT=AF_PIKANS`). Nada de esto se ha ejecutado.
`artifact_slice/artifacts_core.py` (el módulo genérico de otro agente) **no existe aún**: `piezo/artifacts.py` es autocontenido y su contrato (`grid_points`, `reference_fields`, `boundary_sets`, `consistency`, `residual_terms`, `loss`) es lo que los demás problemas deberán exponer para reutilizarlo.

## Cambios por la alineación con el paper (ver `ALINEACION_PAPER.md`)
Semillas 20260728-20260732; brazo de referencia `MLP` (bloque `k26`, baja prioridad); puntuación adicional de tensiones y D contra el FEM P3 refinado (`score_flux`, `reference/fem_flux_*.npz`, +0.2 MB en el bundle; la solución P3 de 293 619 DOF tarda 40-60 s y se hace una sola vez con `make_fem_flux_reference.py`); RMSE, MAE y máximo absoluto de u, v, φ en `score`. El kernel de producción no se ve afectado en costo (la puntuación de tensiones es una evaluación de 4 221 puntos al final).
**Pruebas:** `tests/test_piezo_problem.py` completo sin caída (el segfault de torch en CPU aparecía en un recálculo de chunk 16 con el estado de la sesión; las pruebas de artefactos usan chunk 2); el registro W&B offline se prueba en un subproceso aparte con timeout y se omite con nota si el sistema lo cierra (en Windows el servicio de wandb puede hacerlo; por defecto `artifacts.py` no arranca wandb en win32 salvo `X28_WANDB_FORCE=1`).
