# Verificación final

Fecha: 2026-09-20.

- Suite Python completa: 66 pruebas aprobadas, incluidas 18 de infraestructura.
- Suite Senpi JavaScript: 23 pruebas aprobadas.
- Smoke final: prepare -> run (semillas 11 y 17) -> evaluate -> compare.
- Resultado: execution_valid=true; scientific_accepted=false.
- Revisión independiente: dos hallazgos corregidos y revisados nuevamente.
- Regresiones cubiertas: transporte modificado incluso al regenerar su índice
  de hashes; artefactos anidados obsoletos en reintentos.
- Configuración MCP: Kaggle y Colab desactivados, sin credenciales.

Esta carpeta sustituye la demostración preliminar `../infra-smoke/`.
No hubo entrenamiento PIKAN, uso de GPU remota ni conexión real a Kaggle/Colab.
Las pruebas del transporte remoto usan un sustituto de la llamada CLI, mientras
que el notebook, los subprocesos y el evaluador se ejecutan realmente en local.

Artefactos: `evaluation.json`, `comparison.json`, `bundle/experiment.ipynb` y
`runs/seed-*/run_summary.json`.
