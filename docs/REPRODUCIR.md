# Reproducir y verificar

Referencia de forma: los repositorios de resultados matemáticos obtenidos con agentes. Los que pude comprobar (DeepMind, [alphaevolve_repository_of_problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems)) publican una página por problema, el *prompt*, el código de verificación y los programas inicial y final, un `status.json` con el estado de cada problema, licencia y cita, y **dicen qué no incluyen** (el sistema que generó los resultados). Sobre los repositorios de OpenAI con certificados Lean, solo hallé cobertura secundaria sin confirmar; no se tomó nada de ahí como hecho. Aquí se adoptó la misma disciplina, adaptada a un proyecto empírico en el que no hay prueba formal sino datos guardados.

| Idea de la referencia | Equivalente en este repo |
|---|---|
| Una página por problema, con su *prompt* | `experiments/<problema>/EXP-NNNN-<slug>/spec.md` (pre-registro) y `decision.md` |
| Código de verificación | `evals/` (solo lectura) y `tools/reproduce_numbers.py` |
| `status.json` y rejilla de estado | `experiments/status.json` y `experiments/TRAZA.md` (generados por `tools/build_trace.py`) |
| Reconstruir los certificados con un comando | `python tools/reproduce_numbers.py`: recalcula las cifras citadas desde los JSON versionados y falla si una no sale |
| Registrar también los fracasos | `registry/failures.jsonl`, `decision.md` con `rejected` / `not_supported`, y `LECCIONES_EXP0029.md` |
| Licencia y cita | `CITATION.cff`, `LICENSE` (Apache-2.0) y `LICENSE-docs.md` (CC BY 4.0) |
| Lo que el repo no contiene | Sección siguiente |

## Qué NO está en el repositorio (y cómo conseguirlo)
- **Pesos, estados cada 1 000 pasos, bloques GN, trazas `.wandb`** (≈ 1.3 GB solo en EXP-0023 y EXP-0028). Están fuera de git (`.gitignore`). Copia de las curvas y la configuración: proyecto W&B `AF_PIKANS`, entidad `man22611-uvg`. El resto se conserva local y se reparte aparte (Release, Zenodo o disco), con suma de verificación.
- **`var/`** (12 GB de descargas de Kaggle, entornos): regenerable, ignorado.
- **Referencias externas** (`Allen_Cahn.mat`, soluciones COMSOL/FEATFLOW, FEM de piezo): se regeneran con los scripts de `evals/problems` o se descargan; no se redistribuyen archivos de terceros sin licencia revisada.
- **`.env`** (claves de W&B, Kaggle, Modal): nunca. Se crea a mano desde un `.env.example`.

## Entorno
Python 3.12 y `requirements.txt` (versiones fijadas). Hardware de las corridas: cada JSON de piezo trae `device`, `dtype` y `run_card`. El hardware fue mixto (T4, H100, A100) y está confundido con el brazo en la comparación M frente a E: no pasarlo por alto al citar.

## Verificación en cuatro comandos
```bash
python tools/setup_git.py          # una vez por clon: filtro que guarda los notebooks sin salidas
python tools/check_repo.py         # nombres, registro, tamaños, secretos
python tools/reproduce_numbers.py  # las cifras citadas salen de los datos guardados
python -m pytest tests -q          # pruebas del código (algunas se saltan sin el repo de escritura)
```
`experiments/TRAZA.md` y `experiments/status.json` se regeneran con `python tools/build_trace.py`.

## Decidido el 2026-10-26
- **Licencia:** Apache-2.0 para el código y CC BY 4.0 para documentos y datos derivados (`LICENSE`, `LICENSE-docs.md`).
- **Decisiones firmadas:** `experiments/mechanisms/EXP-0028-failure-attribution/decision.md` (EXP-0028, `accepted_exploratory`) y `decision-EXP-0029.md` (EXP-0029, `not_supported`). Los borradores quedaron en `_borradores/`.
- **Claims aceptados:** C1 a C7 en `registry/claims.jsonl`.

## Publicación
El repositorio es público en https://github.com/Rodrimansidub14/AF_PIKANs. La versión citada en la tesis es la etiqueta `tesis-v1` (commit `4d04917`).

## Pendiente
Dónde alojar la evidencia pesada (pesos, estados y trazas), con suma de verificación.
