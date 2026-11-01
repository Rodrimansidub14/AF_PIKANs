# AF_PIKANs agent entrypoint

## Arquitectura: proposer / verifier

Este repositorio opera un loop de investigación asistida por agentes con una
frontera de escritura en el centro:

```
Investigador → objetivo · hipótesis · presupuesto
     ↓
PROPOSER   literatura · hipótesis · mecanismo · manifiesto · análisis
     ↓     (escribe en src/mechanisms/, experiments/, var/agent_runs/)
INFRA      local CPU · Kaggle · Colab
     ↓
VERIFIER   evals/ — problemas, graders, suites, baselines congelados
     ↓     (SOLO LECTURA para todo agente)
REGISTRY   registry/ — runs · proposals · failures · claims
     ↓
Investigador → gate científico → claim → escritura
```

**La regla que gobierna todo lo demás:** un agente puede poner el build en
verde; solo un humano puede hacer que un claim sea verdadero.

## Superficies

| Ruta | Agente |
|---|---|
| `evals/` | **solo lectura, siempre.** Es el verificador |
| `registry/claims.jsonl` | **solo lectura.** Los claims los acepta el humano |
| `context/`, `workflow/`, `.senpi/`, `.git/`, `var/context/` | solo lectura |
| `src/mechanisms/` | escribible por `mechanism_proposer`, sin gate |
| `experiments/` | escribible por `hypothesis_planner`; firmar el pre-registro es humano |
| `var/agent_runs/`, `artifacts/` | escribible |
| `PIKAN-Vault/PIKAN-Vault` | solo `literature_scout`, con aprobación humana explícita |

`evals/` está protegido porque un agente que puede editar sus graders, sus
umbrales o su malla de evaluación puede aprobar cualquier mecanismo cambiando
la vara en vez del resultado. Si un resultado parece exigir mover la vara,
detente y pide el gate `eval_integrity`.

Un mecanismo en `src/mechanisms/` **no puede importar `evals`**: un candidato
que conoce su métrica la optimiza en vez del problema.
`tests/test_mechanism_isolation.py` lo verifica.

## Tiers de ejecución

| Tier | Costo | Autoridad |
|---|---|---|
| 0 smoke | segundos, CPU | libre |
| 1 regresión | minutos, CPU | libre |
| 2 candidato | horas, GPU | presupuesto aprobado |
| 3 tesis | caro | gate `scientific_claim` |

La restricción vinculante de este proyecto es GPU-hora, no throughput de
agentes. El loop solo es viable si Tier 0/1 absorbe la mayoría de iteraciones.

## Routing

Usa `context/` como fuente de control y descubre la ruta activa desde
`workflow/index.toml`. Los prompts se enrutan automáticamente por lenguaje
natural; no pidas al usuario que pegue reglas, use `/workflow` o declare un
rol para tareas ordinarias. Conserva `research-role=...` únicamente como
override explícito.

## Límites

Conserva `workflow/` y `var/context/` como rutas de control y no modifiques
ninguna ruta fuera de este repositorio. Los archivos de política y guardia
(`context/agent-policy.toml`, `AGENTS.md`, `.senpi/extensions/` y `evals/`)
solo pueden actualizarse para un cambio aprobado explícitamente por el
usuario. `PIKAN-Vault/PIKAN-Vault` solo puede modificarse para promover notas
con aprobación humana explícita, usando la superficie declarada para
`literature_scout`. La aprobación humana sigue siendo necesaria para gates
científicos, claims y conocimiento canónico. Las lecturas externas solicitadas
por la tarea están permitidas en modo solo lectura, pero deben anunciarse con
la ruta exacta antes de ejecutarse. Las escrituras de archivos fuera del
repositorio siguen bloqueadas. Por aprobación explícita del investigador
(2026-09-20), se permite conectar Kaggle, subir notebooks privados y ejecutar
campañas previamente aprobadas conforme a `context/agent-policy.toml`.
Esta excepción no habilita otros servicios ni concede gates científicos.
Las credenciales y la autenticación se completan al final de la preparación.

Por aprobación explícita del investigador (2026-09-22) se estableció la
arquitectura proposer/verifier descrita arriba, con `evals/` protegido y
`src/mechanisms/` como superficie escribible.

## Proceso

El proceso de ingeniería está en `specs/proceso-de-ingenieria.md`.
Léelo antes de abrir un experimento.
