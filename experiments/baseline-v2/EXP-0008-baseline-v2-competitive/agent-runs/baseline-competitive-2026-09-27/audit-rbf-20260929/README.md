# Auditoría por semilla del control RBF-Sobol — 2026-09-29

Estado: ejecución local, solo lectura sobre artefactos archivados. Sin GPU, sin Kaggle, sin cuota, sin cambios en `evals/`, `registry/` ni umbrales. Nada aquí acepta una referencia, congela un baseline o constituye un claim.

Archivos:

- `build_audit.py` → `rbf-audit-per-seed.csv` (35 filas: Heat 10, Burgers 10, AC 15) y `rbf-audit-summary.json`.
- `rescore_heat_stages_cpu.py` → `heat-s2-stage-rescore-cpu.json` (re-score post hoc, CPU float64, de snapshots Adam-only de Heat S2; ejecutar con `py -3.13`).

## Resultados

### Heat α=0.1 — el enigma del ~81× queda resuelto

Las dos cohortes RBF (v4 500 s y S2) son las mismas semillas 901–905, mismo `model_id=rbf_fixed`, mismos pools (SHA-256 de pool idéntico por semilla en las 5). Difieren solo en la etapa/presupuesto entrenado:

| Snapshot | Puntuado por | Media ± sd (n=5), L2 relativo 512×256 |
|---|---|---|
| v4, 500 s Adam (≈17.9k–18.1k updates, sin L-BFGS) | archivado | `1.677869e-4 ± 1.338478e-4` |
| S2 `checkpoint-adam-final.pt` (20 000 Adam, sin L-BFGS) | **re-score CPU nuevo** | `2.197255e-4 ± 1.802112e-4` |
| S2 `checkpoint-500s.pt` (snapshot a 500 s) | re-score CPU nuevo | `2.056045e-4 ± 2.777356e-4` |
| S2 `checkpoint-final.pt` (20 000 Adam + L-BFGS parcial) | archivado; re-score CPU coincide a ≤7.3e-13 relativo | `2.060803e-6 ± 2.306606e-7` |

Conclusiones verificadas:

1. **El `2.060803e-6` es el checkpoint post-L-BFGS** (`confirmation-cost.json` apunta a `checkpoint-final.pt`; `run_metadata.stage=lbfgs_final`). No es Adam-only. El Adam-only de esas mismas corridas es `2.197e-4`.
2. La cifra `2.197255e-4 ± 1.802112e-4` es idéntica, a todos los dígitos, a la fila `RBF-PIKAN_archived` de `scorecard-current.csv`: esa fila es el endpoint Adam-20k de S2.
3. Los dos controles Adam-only (v4 1.68e-4 y S2 2.20e-4) son consistentes dado el ruido entre semillas; el salto ~100× es atribuible a L-BFGS, incluso con 22–73 updates en 4 semillas (cociente final/Adam por semilla 0.004–0.033). El cap de 500 s no explica la brecha.
4. El error de Heat en etapa Adam es muy ruidoso entre snapshots de la misma corrida (p. ej. semilla 902: 1.6e-4 en 20k, 6.8e-4 a 500 s). Un mecanismo evaluado sobre el endpoint Adam necesita superar ese ruido: usar n≥5 pareado y reportar la etapa.

Decisión para el B0 de Heat (para el investigador): elegir la etapa antes de elegir la cohorte. Adam-only (tiempo/pasos iguales al candidato) → cohorte v4 o S2 `adam-final`, ya con score y pools idénticos. Adam+L-BFGS → S2 `final`, pero L-BFGS parcial (4 paradas tempranas, 1 cap 680 s) es un procedimiento distinto y el mecanismo debe recibir el mismo. No mezclar.

Pendiente Heat: `heat_v4_500s` no tiene `run_status` de completitud en el ledger (`gpu_seconds_cap_during_adam` es el cap esperado; el ledger global `incomplete_validation` viene del target de 1M updates, no de un fallo).

### Burgers canónico (EXP-0006)

Reproducido desde `run_summary.json` de los diez jobs: coincide con `paired-runs.csv` en los diez; checkpoint `checkpoint-020000.pt` coincide con `evidence_hashes.json` en los diez; los diez con `bundle_id` único `4393e173…3938`; 20 000 pasos en los diez. Sobol `3.954980e-2 ± 2.564915e-2`, RAD `2.126864e-1 ± 7.526204e-2`, cociente RAD/Sobol media geométrica `5.865147`, RAD peor en 5/5. Tiempos (`elapsed_sec`): Sobol media 1062 s, RAD 1049 s (rango 1038–1077); con otro reloj (`seconds` del CSV) ~1039/1024. En ambos RAD no es más lento, así que el paso igualado no favorece a Sobol. Hashes de `pools.npz` por job están en el CSV. No se necesita relanzar.

### Allen–Cahn

Adam: 600 s por semilla con tope de tiempo (27 930–29 288 pasos, float64, sin selección por referencia); no es paso igualado. Los cinco hashes de checkpoint del score-only coinciden con el preflight S3. Continuación L-BFGS: 240 s por semilla, todas `time_cap_reached` (1 507–1 997 updates).

| Etapa | Referencia | Media ± sd (n=5) |
|---|---|---|
| Adam endpoint | FD8192 candidata | `1.650741e-2 ± 2.435627e-3` |
| + L-BFGS 240 s | FD8192 candidata | `9.163726e-3 ± 2.664378e-3` |
| + L-BFGS 240 s | ETDRK4 candidata | `9.163711e-3 ± 2.664377e-3` |

Hallazgo que corrige la auditoría previa: el score Adam se calculó contra la referencia con SHA `8ce9fb20…` = **FD8192**, no ETDRK4 (así lo dicen `score_only_manifest.json` y el `source_preflight.references` de S3). Por tanto Adam y L-BFGS se compararon contra la misma referencia candidata y la mejora ×1.80 por L-BFGS es limpia en ese sentido. Matiz de malla (leído del código, no medido): el score-only evalúa sobre la malla nativa de la referencia FD8192 (`evaluate_field` sobre `reference["x"], ["t"]`), mientras S3 interpola primero la referencia a 512×256 (`load_reference`). Las dos etapas no están en malla idéntica; con FD8192 vs ETDRK4 coincidiendo a 1.4e-8 en 512×256 el efecto debería ser despreciable frente a 9e-3 y 1.65e-2, pero no está cuantificado. Cuantificarlo requiere re-puntuar el checkpoint Adam en 512×256 (local, sin entrenar, si hay torch; no hecho).

## Errores encontrados en los documentos revisados

- `baseline-closure-audit-rbf.md`, fila AC: dice que el endpoint Adam se puntuó "contra ETDRK4 candidata" y el post-L-BFGS "contra FD/RK4". Ambos usan FD8192 (y S3 además reporta ETDRK4). El plan (`baseline-closure-plan-rbf.md`) ya lo dice bien.
- `baseline-closure-audit-rbf.md` y plan, Heat: "score 2.06e-6 no se puede llamar Adam-only" — correcto, y ahora está resuelto: es post-L-BFGS y el Adam-only es `2.197e-4`.
- `baseline-closure-audit-rbf.md`, Burgers: tiempos 1 024/1 039 s son del CSV pareado; con `elapsed_sec` son 1 049/1 062. Mismo sentido, distinto reloj.

## Lo que sigue necesitando decisión humana o cuota

- Referencias AC (FD8192 vs ETDRK4 vs BDF/`.mat`): gate `scientific_claim`.
- Elegir etapa del B0 (Adam vs Adam+L-BFGS) por problema y firmar el freeze: gate `eval_integrity`.
- Ningún Tier 2 es necesario para cerrar el B0 de Heat, Burgers ni AC con lo archivado. El único hueco de datos es el efecto de la malla nativa vs 512×256 en el score Adam de AC (re-score local sin entrenar).
