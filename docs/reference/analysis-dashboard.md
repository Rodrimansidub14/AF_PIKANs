# Dashboard de análisis posterior

La app Panel lee artefactos descargados, ofrece filtros por problema/señal,
presenta la tabla cuantitativa y las curvas Plotly. PyVista se activa únicamente
si el directorio contiene mallas VTK (`.vtu`, `.vtk`, `.vtp`, `.vts` o `.vtr`).
La exportación crea PNGs con `Plotter(off_screen=True)` y no abre ventanas.

El run Heat/Burgers actual representa campos escalares sobre una malla tensorial
de validación, no una geometría/malla VTK; para esos mapas la libreta usa Plotly.
El viewer de PyVista queda listo para Navier–Stokes DFG y otros problemas con malla.

La acción `fetch` copia la libreta de resultados a `notebooks/<problema>/<EXP>/`,
verifica los hashes y extrae el ZIP. El paquete conserva las corridas W&B offline;
si encuentra `WANDB_API_KEY` en `.env`, las sincroniza desde el equipo local con
`wandb sync`. La clave solo se entrega al proceso en memoria. Sin ella, los runs
siguen disponibles localmente y `wandb-sync.json` marca la sincronización pendiente.

En PowerShell, desde la raíz del repo:

```powershell
& 'var/agent_runs/remote-env/Scripts/python.exe' -m pip install -r requirements-analysis.txt
$env:AF_PIKAN_RUNS = 'var/agent_runs/heat-burgers-002-t4x2/download-extracted/runs'
panel serve src/research_pipeline/remote/analysis_dashboard.py --show
```

El nombre `<bundle-id>` se registra en `bundle.json` y en el reporte de descarga.
La app no lee `.env`, claves de Kaggle ni claves de W&B.
