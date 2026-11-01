# specs/ — el contrato que dirige el trabajo

Reemplaza a `reports/`. La diferencia no es de nombre: un reporte describe
algo que ya pasó, un spec dirige algo que todavía no pasa. En un repositorio
agent-first el artefacto durable es el segundo.

| Ruta | Contenido |
|---|---|
| `proceso-de-ingenieria.md` | **El proceso que gobierna el repo. Empezar aquí** |
| `tooling/` | Specs SDD del tooling determinista (harness, pipeline, remoto, LaTeX) |
| `tooling/plans/` | Planes de implementación asociados |
| `conventions/` | Convenciones transversales (figuras, notación) |

Los pre-registros de experimentos **no** viven aquí: viven junto a su
experimento, en `experiments/<ID>/spec.md`, porque spec, manifiesto y decisión
son la unidad de trabajo que un agente toma completa.

| | `specs/tooling/` | `experiments/<ID>/spec.md` |
|---|---|---|
| Régimen | determinista | empírico |
| Pregunta | ¿qué debe construirse? | ¿qué debe medirse? |
| Lo cierra | los tests | el investigador, sobre evidencia |
