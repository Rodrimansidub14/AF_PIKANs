# Autorización registrada y envío: EXP-0003 Allen–Cahn

Estado: autorización de la campaña registrada desde la instrucción explícita del investigador del 2026-09-23; el bundle privado fue enviado a Kaggle y su estado más reciente fue `RUNNING`. `approval.json` conserva la autorización ligada al hash.

Bundle verificado localmente:
`7d9fafd823f3e3150907ed029be678103e23c129fcab265aba61da9bff233338`.

Revisar `notebooks/experiments/allen-cahn-rbf-sobol-exploratory.ipynb` y
`experiments/EXP-0003-allen-cahn-rbf-sobol/parameter-decisions.md`. La corrida
está limitada a tres semillas, dos T4 en paralelo, 20.000 pasos o 1.200 s de
entrenamiento por semilla. W&B queda offline. Error de campo y aceptación
científica continúan bloqueados.

Bundle: `var/agent_runs/allen-cahn-exp003/bundle-final`  
Notebook SHA-256: `64b911ddcfdeecc66b372276de0a0cea1db7c0ede60bfcdf2c67d07b52026b98`

La aprobación registrada contiene estos campos, fuera del bundle como `approval.json`:

```json
{
  "bundle_id": "7d9fafd823f3e3150907ed029be678103e23c129fcab265aba61da9bff233338",
  "approved_by": "Rodrigo Mansilla",
  "scope": "approved",
  "hypothesis": "approved",
  "code_surface": "approved",
  "remote_execution": true
}
```

El token se usó solo en memoria para el CLI; no se incluyó ni persistió en el bundle o en Git.
