# Convención de nombres y trazabilidad

Vigente desde 2026-10-26. Lo existente se respeta (ver «Heredado»); lo nuevo sigue estas reglas, y `tools/check_repo.py` las comprueba.

## 1. Experimentos

- Carpeta: `experiments/<problema>/EXP-NNNN-<slug>/`. `NNNN` de cuatro dígitos, **nunca se reutiliza** (el 0019 está reservado). `<slug>`: minúsculas, números y guiones.
- Los experimentos de mecanismo van en `experiments/mechanisms/`; los transversales o de diagnóstico en `diagnostics/` y `baseline-v2/`.
- Cada carpeta tiene `spec.md` con front matter (`experiment_id`, `slug`, `status`, `preregistered_at`, `hypothesis_id`) y, cuando se cierra, `decision.md` con `decision` y `claims`.
- `status`: `draft` → `preregistered` → `running` → `scored` → `decided`. `decision`: `accepted`, `rejected`, `not_supported`, `inconclusive`, `superseded`, `closed_*`.
- Un experimento derivado que comparte código con otro puede vivir como spec dentro de la carpeta del primero (EXP-0029 dentro de EXP-0028); entonces se declara en `experiments/INDEX.md` y en `registry/runs.jsonl`.
- Un borrador de decisión del agente se llama `decision-borrador*.md`. **Solo `decision.md` es una decisión.**

## 2. Resultados

- `results/<fase-o-bloque>[-<AAAAMMDD>]/`, minúsculas con guiones. Dentro: un JSON por corrida `<problema>-<brazo>-seed<N>.json` y, aparte, los pesados (`*-artifacts.npz`, `*.pt`), que no se versionan.
- Puntuación y lectura: `results/<nombre>_scored.md|json`, `RESULTS.md` o `results.md`; el resumen para humanos nunca vive solo en un `.log`.
- Cada bloque con resultados tiene una entrada en `registry/runs.jsonl` con el mismo identificador que su carpeta (`run_id`).

## 3. Notebooks

| Dónde | Nombre | Para qué |
|---|---|---|
| `notebooks/<problema>/EXP-NNNN-<slug>/` | `expNNNN-<rol>-<tema>[-vN].ipynb` | Exploración y análisis interactivo. `rol` ∈ {`exploration`, `baseline`, `candidate`, `analysis`, `reporting`} (contrato en `context/notebook-contract.toml`) |
| `experiments/<problema>/EXP-NNNN-<slug>/kaggle/<corrida>-<AAAAMMDD>/notebook.ipynb` | fijo: `notebook.ipynb` | Lo que se despacha a Kaggle; la carpeta lleva el nombre y la fecha |
| `experiments/<...>/colab/` | `<tema>_colab.ipynb` → se renombra a `<tema>-colab.ipynb` al tocarlo | Corridas en Colab |
| `notebooks/warmup/<problema>/` | `<tema>.ipynb` | Calentamiento previo al proceso; sin derecho a evidencia |

Reglas duras: todo en minúsculas, guiones, sin espacios, paréntesis ni acentos; sin sufijos `(1)` ni `Copy_of_`. Copias: `-copia-N`. Respaldos: `-pre-<motivo>-AAAAMMDD`. Los notebooks se versionan **sin salidas** (filtro `nbstrip`, `python tools/setup_git.py` una vez por clon); las salidas viven en el archivo local y en la evidencia de `results/`.

## 4. Código y datos

- Módulos de mecanismo: `src/mechanisms/<nombre>.py`, con prueba `tests/test_<nombre>.py`.
- Scripts de análisis de un experimento: `analyze_<bloque>.py` junto a su `results/`; scripts de corrida: `run_<fase>.py`.
- Referencias externas (soluciones exactas, FEM, `Allen_Cahn.mat`): se descargan o regeneran con el script que las produce; no se versionan archivos de terceros sin licencia revisada (`docs/REPRODUCIR.md`).

## 5. Registro y trazabilidad

- `registry/` es append-only. `run_id` único. `recorded_by` distingue `agent`, `agent_backfill_AAAA-MM-DD` y humano.
- `experiments/TRAZA.md` se **genera** (`python tools/build_trace.py`); no se edita a mano. Cruza experimento → estado → decisión → bloques de corridas → notebooks → grupo en W&B.
- W&B: proyecto `AF_PIKANS`; `group` = fase o bloque, `job_type` = problema, `name` = `<problema>-<brazo>-seed<N>`; las reconstruidas desde JSON llevan la etiqueta `reconstruido_desde_json`.

## 6. Git

- Ramas: `main` estable; trabajo en `exp/EXP-NNNN-<slug>` o `docs/<tema>`. Un commit por unidad lógica; mensaje `tipo(EXP-NNNN): resumen` con `tipo` ∈ {`feat`, `fix`, `docs`, `exp`, `data`, `chore`}.
- No se commitea: pesos, `.wandb`, `.npz` de corrida, `.env`, `var/`. El criterio automático es `.gitignore` más `tools/check_repo.py`.
- Las etiquetas (`tag`) marcan los hitos citados en la tesis: `tesis-v1`, `baseline-v2` y similares.

## Heredado (permitido, no ejemplar)

`notebooks/heat/EXP-0002…`, `EXP-0003…`, `EXP-0005…` y `baseline-v2/EXP-0008…` conservan los nombres con que se citaron (`heat-burgers-architecture-screen-…`); no se renombran para no romper referencias. Las rutas viejas ya movidas están en `docs/MOVED.md`.
