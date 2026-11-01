# Estado consolidado para la redacción (2026-10-26)

Para escribir Resultados y Discusión desde `ProyectoGrad/` sin reabrir cómputo. El cómputo está cerrado por decisión del investigador (sin crédito en Modal); se reporta lo encontrado. Este documento apunta a las fuentes; las cifras viven en los archivos citados, no aquí.

Rutas relativas a `AF_PIKANs/`. `EXP28` = `experiments/mechanisms/EXP-0028-failure-attribution`.

## 1. Qué responde cada fuente

| Para escribir sobre | Leer | Datos crudos |
|---|---|---|
| Línea base (objetivo 1) | `evals/baselines/` (v1 2026-10-03, v2 2026-10-04), `experiments/*/RESULTS.md` | `registry/runs.jsonl` (EXP-0008, EXP-0009) |
| Familias de mecanismo cerradas (muestreo y otras) | `experiments/INDEX.md` (EXP-0011 a 0018) y cada `decision.md` | `registry/failures.jsonl`, `registry/proposals.jsonl` |
| Precondicionamiento E, F, I, J en el banco (objetivo 2) | `mechanisms/EXP-0023-gram-preconditioning/` (`decision-round1B.md`, `results/c1-V/RESULTS_V.md`, `results/c1-P/RESULTS_P.md`), `EXP-0025-observer-v0/results/block-T1/RESULTS.md`, `EXP-0026-e-burgers-diagnostic/results/RESULTS.md`, `EXP-0027-seed-predictor/results.md` | JSON por corrida en cada `results/` |
| Atribución de fallos y fases 1a a 1e | `EXP28/spec.md`, `predictions.md`, `results/phase*_scored.md`, `results/phase1e_all_scored.md` | `EXP28/results/phase1*/` |
| Piezo: predicciones firmadas frente a resultado | `EXP28/results/piezo_vs_predictions.md`, `EXP28/results/F_diagnosis_piezo.md`, `EXP28/ANALISIS_PIEZO_PROPIEDADES.md`, `EXP28/analisis_piezo/` | `EXP28/results/piezo-modal-run/`, `piezo-s1a`, `piezo-s1b` |
| Mecanismo final M (objetivo 3) | `EXP28/EXP-0029_spec.md` (firma), `EXP28/DISENO_FINAL.md`, `EXP28/results/EXP0029_vs_predicciones.md`, `EXP28/LECCIONES_EXP0029.md`, `EXP28/results/T1_T2_resultados.md` | `EXP28/results/piezo-exp0029-final/` (651 MB: pérdida por paso, estados cada 1 000 pasos, bloques GN) |
| Caracterización de B, E y M | `mechanisms/EXP-0030-characterization-M/results/RESUMEN.md` y los `.md` de la misma carpeta; `CIERRE_OBJETIVO3.md`, `CARACTERIZACION_PLAN.md` | datos de EXP-0029 (sin corridas nuevas) |
| Diseño (araña, ejecución) | `EXP28/DISENO_ARANA.md` (vigente), `DISENO_FINAL.md`, `DISENO_EJECUCION.html` | — |
| Literatura de piezo | `EXP28/REVISION_LITERATURA_PIEZO.md` (referencias por verificar; ninguna se abrió completa) | — |

## 2. Resultados que sostienen el texto (con su grado de confianza)

- **E sale antes del llano en 5 de 5 semillas del converso** (mediana 5 500 frente a 7 000 pasos). Hallazgo posterior a ver los datos, no predicción previa.
- **Error en el último paso:** B y E quedan iguales con n = 5 (E 0.158, B 0.152 de media). La ventaja de E está en la cola: error robusto (mediana de `l2re_u` en los pasos 18 000 a 20 000) E 0.033 a 0.102 frente a B 0.039 a 0.391. **La métrica robusta se definió después de ver los resultados: declararla como post hoc.**
- **Predicción firmada antes de piezo:** F y E mejoran modestamente. Resultado: E fue el mejor y F el peor.
- **El voltaje flotante mal ajustado no explica el error del directo** (ya es el minimizador exacto; `EXP28/analisis_piezo/check_Vf.md`).
- **Con M** (converso 28 a 30): 0.336, 0.166, 0.165 frente a E 0.050, 0.077, 0.033. Directo (n = 3): B 0.208, E 0.193, M 0.180, dentro del ruido.
- **La parte eléctrica casi no pesa en el converso** (≈ 0.015 % de la traza en un estado de una semilla; `analisis_piezo/por_termino.py`). La composición de la pérdida depende de cómo se agregue (por arista de la capa 0 o sobre todas las aristas): `EXP-0030/results/RESUMEN.md` corrige las cifras de `LECCIONES_EXP0029.md`; la dirección se mantiene.

## 3. Salvedades que condicionan lo que se puede afirmar

1. **M frente a E está confundido con el hardware.** M corrió solo en A100; E de las semillas 28 a 30 corrió en T4 y H100, y las únicas semillas de E en A100 (31 y 32) tampoco muestran la ventaja sobre B. Con estos datos no se puede atribuir «M pierde la ventaja de E» a las patas, al hardware o a las semillas (`EXP-0030/results/A2_mismo_hardware.md`). Hay dos diferencias de entorno más: B y E de la semilla 28 corrieron con torch 2.11 y Python 3.13 (Kaggle), el resto con torch 2.14.1 y Python 3.12.
2. **Muestras pequeñas:** 3 a 5 semillas por brazo, sin p-valores. En subconjuntos de semillas, E gana a B en más de la mitad de las semillas en 0 de 5 subconjuntos de 4 y en 3 de 10 de 3.
3. **Corridas que faltan:** M converso 31 y 32 (OOM en A100-40GB), B y E directo 31 y 32, Burgers con receta completa (solo 2 semillas), DFG (cortado por un apagón).
4. **No calculado:** simetrías de geometría y polarización como criterio de admisibilidad (EXP-0030 las sustituyó por invariantes, `I_invariantes.md`), Morris (diferido), cuotas por grupo en los artefactos viejos de B y E directo.
5. **Idea abierta, sin correr:** M2 (P5 disparado por cambio de cuotas, P2 solo en grupos con cuota ≥ 1 %). Probarla requeriría corridas cortas del converso (8 000 pasos).

## 4. Lo que solo el investigador puede cerrar

- ~~`decision.md` de EXP-0028 y EXP-0029~~: **firmadas el 2026-10-26** (`accepted_exploratory` y `not_supported`; la de EXP-0029 es `decision-EXP-0029.md`). EXP-0030: `inconclusive`.
- ~~`registry/claims.jsonl`~~: **C1 a C7 aceptados el 2026-10-26** (ver `docs/plans/2026-10-26-claims-candidatos.md`). Cada uno exige citarse con su límite. El campo `thesis_section` está pendiente hasta que existan los capítulos.
- Redacción del objetivo 2: «focalización adaptativa» frente a «ajuste adaptativo de la escala» (precondicionamiento). Está planteado en el brief.
- Revocar el token de Modal que se pegó en el chat.

## 5. Trazabilidad y registro

- `registry/runs.jsonl`: completo hasta EXP-0030. Las 24 entradas del 2026-10-26 tienen `recorded_by: agent_backfill_2026-10-26` y `scientific_accepted: false`; fechas, semillas, brazos y rutas salen de los JSON, y los backends se infirieron de los kernels y scripts.
- `registry/proposals.jsonl`: 109 entradas (más recientes: EXP-0030 bloques I a L, cierre de cómputo).
- **W&B:** proyecto `AF_PIKANS`, entidad `man22611-uvg`. Herramientas y reglas de la clave en `EXP28/wandb_tools/README.md`; bitácora de envíos en `EXP28/wandb_tools/sync_log.json`. Las corridas de piezo de EXP-0029 son instrumentadas (24 `offline-run` en `piezo-exp0029-final/piezo_exp0029/wandb`); las del banco son reconstruidas desde JSON (sin pesos, campos ni residuos por punto) y así están etiquetadas.
- Evidencia pesada (`*.npz`, `results/piezo-exp0029-final/`) está fuera de git por `.gitignore`; se referencia por ruta.

## 6. Verificación

```bash
"C:\Program Files\Python312\python.exe" -m pytest tests/test_piezo_problem.py tests/test_residual_terms.py tests/test_gram_preconditioning.py tests/test_mechanism_isolation.py tests/test_arm_S.py -q
```
