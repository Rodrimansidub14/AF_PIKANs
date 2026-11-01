# Entrega de baselines — 27 de septiembre de 2026

Abrir `index.html` para las tablas y gráficas; `baselines-report.md` contiene la revisión y los límites de cada resultado. Los HTML incluyen Plotly y funcionan sin conexión después de extraer el ZIP.

Notebooks de resultados: `heat-results.ipynb`, `burgers-results.ipynb`, `allen-cahn-results.ipynb`, `dfg-results.ipynb`; resumen: `baselines-results.ipynb`. `notebook-execution.json` registra sus ejecuciones efectivas. Los CSV contienen métricas por semilla, medias, desviaciones estándar e intervalos descriptivos.

Los entrenamientos en Kaggle ya estaban terminados; esta entrega ejecuta inferencia de checkpoints y posprocesado. No repite entrenamiento ni ejecuta tests, preflights, validaciones de integridad o auditorías de convergencia locales. Las evaluaciones técnicas citadas son las previamente archivadas por la campaña.

Para recalcular dentro de la copia original del repositorio, desde `C:/Users/rodri/Documents/AF_PIKANs`:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
& 'var/agent_runs/sciml-env/Scripts/python.exe' 'var/agent_runs/baseline-close-2026-09-27/delivery/run_notebooks.py'
```

Se requieren los archivos originales de la campaña en las rutas conservadas en los CSV. Este ZIP incluye el análisis y sus salidas, pero no duplica todos los checkpoints, campos y logs W&B. Los originales permanecen en `var/agent_runs/`.

No se cambia el entrenamiento, la PDE, el muestreo, los umbrales ni los archivos protegidos de `evals/`. `scientific_accepted=false` y `baseline_frozen=false`. Allen–Cahn conserva referencia provisional; DFG se reporta mediante QoIs y residuos, sin inventar error de campo FEM.
