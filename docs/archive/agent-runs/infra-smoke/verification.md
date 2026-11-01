# Verificación de infraestructura remota

Fecha: 2026-09-20.

- Suite Python completa: 63 pruebas aprobadas.
- Suite Senpi JavaScript: 23 pruebas aprobadas.
- Incluidas 15 pruebas de infraestructura remota.
- CLI: prepare, run, evaluate y compare ejecutados localmente.
- Smoke: semillas 11 y 17 completadas; hashes y snapshots validados.
- Notebook autocontenido ejecutado como subproceso real en tests.
- Timeout, resultados no finitos, archivos modificados y duplicación de envío cubiertos.
- TOML de MCP parseado: Kaggle y Colab desactivados.

Los resultados de esta carpeta son de infraestructura. No entrenan PIKAN, no
demuestran mejora científica y no autorizan claims. El adaptador Kaggle se probó
con transporte simulado; no hubo autenticación, subida, uso de GPU ni descarga
desde una cuenta real. La política de escrituras externas permanece intacta.

Ver `evaluation.json`, `comparison.json` y `runs/seed-*/run_summary.json`.
