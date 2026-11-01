# Qué se migró de `var/agent_runs/` y qué se quedó

`var/` está ignorado por Git y no viaja con el repositorio. El 2026-11-01 se comparó por suma SHA-256 todo el texto de `var/agent_runs/` con lo que ya estaba versionado y se **copió** (sin mover ni borrar) lo que faltaba. `var/` queda intacto como respaldo local.

## Dónde quedó cada campaña

| Origen en `var/agent_runs/` | Destino |
|---|---|
| `heat-burgers-001` | `experiments/heat/EXP-0001-heat1d-baseline/agent-runs/` |
| `heat-burgers-002-t4x2` | `experiments/heat/EXP-0002-heat-burgers-sobol/agent-runs/` |
| `heat-burgers-architecture-screen` | `experiments/heat/EXP-0005-architecture-screen/agent-runs/` |
| `allen-cahn-exp003` | `experiments/allen-cahn/EXP-0003-allen-cahn-rbf-sobol/agent-runs/` |
| `burgers-rad-exp006`, `-v2`, `-v3`, `-v4` | `experiments/burgers/EXP-0006-burgers-sobol-rad/agent-runs/` |
| `baseline-competitive-2026-09-27`, `baseline-close-2026-09-27`, `baseline-rbf-2026-09-22`, `exp0008`, `rbf-signal-baseline-20260924` | `experiments/baseline-v2/EXP-0008-baseline-v2-competitive/agent-runs/` |
| `piezo-ref-20260930` | `experiments/piezo/EXP-0009-piezo-indirect-rbf/agent-runs/` |
| `exp0013-fields` | `experiments/diagnostics/EXP-0013-run-diagnosis/agent-runs/` |
| humo de infraestructura, borradores de escritura, barridos de literatura, informes, briefs y scripts sueltos | `docs/archive/agent-runs/` |

Cada carpeta `agent-runs/<campaña>/` conserva la estructura original: aprobaciones, manifiestos, informes de cierre, scripts, resúmenes y registros de corrida.

## Qué NO se migró

- **Pesos y estados** (`*.pt`, `*.npz`, `*.pth`): pesados y regenerables; se distribuyen aparte con suma de verificación (ver `REPRODUCIR.md`).
- **Entornos y cachés** (`kaggle-cli-env`, `remote-env`, `uv-cache`, `sciml-env`, `node-test-tmp`, `wandb`).
- **Descargas de terceros**: `literature-pdfs` (PDF de artículos) y el repositorio de referencia de Daniel (`piezo-ref-20260930/daniel-ref*`), por la regla de no versionar archivos de terceros sin licencia revisada.
- **Archivos de más de 3 MB** (97) y **registros regenerables** (`*.log`, `*.aux`, `*.out`, `*.pdf`, salidas de pytest).
- **Instantáneas de código** enviadas a Kaggle (`bundle/`, `vendor/`, `adapters/`), que se regeneran con el script de bundle de cada experimento (ver `.gitignore`).

## Tamaños que siguen solo en `var/`

| Campaña | Tamaño aproximado |
|---|---|
| `baseline-competitive-2026-09-27` | 2.6 GB |
| `heat-burgers-architecture-screen` | 1.9 GB |
| `burgers-rad-exp006-v4` | 1.5 GB |
| `heat-burgers-002-t4x2` | 353 MB |
| `burgers-rad-exp006-v3` | 352 MB |
| `allen-cahn-exp003` | 297 MB |
| `baseline-close-2026-09-27` | 272 MB |
