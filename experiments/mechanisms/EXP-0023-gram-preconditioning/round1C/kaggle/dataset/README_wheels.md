# wandb para el kernel de Kaggle de 1-C (sin internet) — preparado, NO subido

`wheels/` contiene `wandb==0.30.0` (la versión de los manifiestos) y todas sus dependencias, descargadas para Linux x86_64, Python 3.13 (`manylinux2014/2_17/2_28`; el log de 1-B indica python3.13 en Kaggle). 26 archivos, ≈ 31 MB; hashes en `wheels.SHA256SUMS.txt`.

Se verificó que el conjunto resuelve sin red: `pip install --dry-run --no-index --find-links wheels --platform manylinux2014_x86_64 --python-version 3.13 --only-binary=:all: wandb==0.30.0` → «Would install … wandb-0.30.0 …». La instalación real en Kaggle no se ha probado (necesita ese entorno).

Uso previsto en el kernel, antes de importar `arnes_round1c`:

    pip install --no-index --find-links <ruta-al-dataset>/wheels wandb==0.30.0

`arnes_round1c.note_run` fuerza `mode="offline"`: nada se sube desde Kaggle ni desde el equipo local; la sincronización es un `wandb sync` manual y autorizado. No se incluye ninguna clave.

Este directorio es el borrador del dataset de 1-C: todavía no existe el dataset en Kaggle ni el kernel, y no se subió nada.
