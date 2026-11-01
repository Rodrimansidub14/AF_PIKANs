# Allen–Cahn EXP-0003: resultados medidos

Bundle `7d9fafd823f3e3150907ed029be678103e23c129fcab265aba61da9bff233338`; kernel `rodrigomansilla/af-pikans-allen-cahn-exp003`. Las tres semillas (101–103) completaron 20,000 pasos cada una y pararon por límite de pasos. El entrenamiento y los diagnósticos sumaron 2260.0 s; cada semilla quedó por debajo del límite de 1,200 s.

| Semilla | GPU / slot | Pasos | Entrenamiento (s) | Residuo normalizado medio | RMSE residual held-out | Energía inicial → final | Utilización GPU media |
|---:|---|---:|---:|---:|---:|---:|---:|
| 101 | Tesla T4 / 0 | 20,000 | 773.68 | 0.00258299 | 0.00380412 | 2.00327 → 0.247727 | 35.7% |
| 102 | Tesla T4 / 1 | 20,000 | 772.04 | 0.000782594 | 0.00108728 | 2.00327 → 0.242265 | 36.1% |
| 103 | Tesla T4 / 0 | 20,000 | 714.27 | 0.000454461 | 0.000599674 | 2.00327 → 0.251379 | 37.5% |

La energía bajó en las cinco mediciones por semilla; no se registraron aumentos. Cada validación usó 16,384 puntos held-out. Pools por semilla: interior 8,192, diagnóstico 2,048, held-out 16,384, frontera 1,024 e inicial 1,024 puntos.

La evaluación local reporta `execution_valid=true`, sin fallos. Los 715 archivos del manifiesto completo SHA-256 coinciden; las 223 entradas de hash de cada semilla también coinciden después de W&B sync. El primer output de Kaggle omitió nueve logs secundarios de W&B, recuperados desde el ZIP completo. La CLI devolvió un código no cero después de escribir resultados, pero la campaña y sus artefactos pasaron la validación local.

`thresholds_pass=false` y `eligible_for_review=false`: el único gate requerido por semilla es `allen_cahn/normalized_residual_mean`, pero el manifiesto no define su umbral (`threshold_not_defined`). No hay aceptación científica.

Los `validation.html` (residuo) y `energy.html` (energía) están presentes y hasheados para las tres semillas. El error de campo/L2 sigue bloqueado: `field_error_status=locked_pending_reference`; no existe métrica de error de campo y `field_reference_used=false`. `final_test_used=false` en JSON y NPZ de las tres semillas; el test final no se abrió. Los errores de traza inicial/frontera sí están medidos, pero no son error de campo.

Los perfiles confirman CUDA y Tesla T4 en slots 0 y 1; la telemetría registró 73–79 muestras por semilla, utilización media de 35.7–37.5% y memoria mediana de 327 MiB. El resumen guarda `device=cpu` después del entrenamiento; se deja anotado como discrepancia del campo post-run frente a la asignación GPU y la telemetría.

W&B sincronizó los tres runs offline (3/3, exit code 0) desde `download/pikan-7d9fafd823f3e315/verified-extract/runs`. El ZIP y el árbol inicial de descarga no fueron destinos de sync. Estado y conteos sin credenciales: `wandb-sync.json`.

La tabla completa por semilla está en `quantitative-signals.csv`.
