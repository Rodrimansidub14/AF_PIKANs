# wandb_tools (EXP-0028)

Herramientas para registrar EXP-0028 en Weights & Biases sin exponer la clave.

## Flujo

1. **Kernel de Kaggle (sin internet)**: registrar con `wandb.init(mode="offline", dir=<salida>)`.
   Las carpetas `wandb/offline-run-*` viajan dentro de la salida del kernel.
2. **Descarga** de la salida del kernel al equipo local.
3. **Sync local**, con la clave solo en memoria:

   ```powershell
   python wandb_tools/sync_wandb.py <ruta-descargada>            # dry-run (por defecto)
   python wandb_tools/sync_wandb.py <ruta-descargada> --apply    # sincroniza
   ```

   `sync_wandb.py` lee `.env` dentro del proceso Python, pasa `WANDB_API_KEY` por el
   entorno del subproceso `wandb sync` (nunca por argv), no la imprime y escribe en
   `sync_log.json` solo carpetas, códigos de retorno y las últimas líneas de salida
   (con la clave censurada). Las carpetas ya sincronizadas se saltan (`.synced`).

## Runs reconstruidos desde JSON

`backfill_from_json.py` crea runs offline a partir de los JSON por corrida en `results/`
(`phase1a`, `phase1b`, `phase1c-t1`, `phase1c-dfg`, `phase1d-ac`, `phase1d-kpp`,
`phase1d-bu1`, `phase1e-ns1` y cualquier `phase1e-*`).

```powershell
python wandb_tools/backfill_from_json.py --limit 3     # humo
python wandb_tools/backfill_from_json.py               # todo, offline en wandb_tools/offline
python wandb_tools/backfill_from_json.py --mode online # sube directo (requiere sesión W&B)
```

Cada run lleva `group=<fase>`, `job_type=<problema>`, `name=<problema>-<brazo>-seed<seed>`,
tags `[fase, problema, brazo, reconstruido_desde_json]`, `config.reconstruido_desde_json=true`,
`config.instrumentado=false` y una nota explícita. Series: `adam/*` (pérdida vs paso),
`lbfgs/*` (si existe `lbfgs_traj`), `pre/*` y `regrid/*` (diagnósticos). No existen en los JSON, y
por tanto no se reconstruyen: pesos, campos, residuos por punto. La clase del criterio uniforme
(`clase_*_agregada_brazo`) es del brazo agregado, no de la corrida individual. El id del run es
determinista (fase + nombre), así que repetir el backfill no duplica carpetas (`--force` rehace).

## Cambiar proyecto o entity

Por defecto salen de `.env` (`WANDB_ENTITY`, `WANDB_PROJECT`; proyecto `AF_PIKANS` si falta).
Para otro destino: editar el `.env`, o `sync_wandb.py ... --entity X --project Y`
(`wandb sync` reasigna el destino al subir).
