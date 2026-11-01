---
experiment_id: EXP-NNNN
decision: pending        # accepted | rejected | inconclusive | deferred
decided_at: 
decided_by: 
supersedes: 
---

# EDR — EXP-NNNN

## Scorecard

<Tabla pegada desde el gate. Valores por semilla, no solo agregados.>

## Gate técnico

`technical_pass`: <true|false|inconclusive>

## Lectura

<Qué muestran los números. Solo los números.>

## Decisión

<Aceptado / rechazado / no concluyente, y por qué. Esta sección solo la
escribe el investigador: es el punto del loop donde un agente no tiene
autoridad.>

## Amenazas de validez que siguen abiertas

## Claim derivado

Si se aceptó, la línea que entra en `registry/claims.jsonl`:

```json
{"claim_id": "", "experiment_id": "EXP-NNNN", "run_id": "", "artifact": "", "latex_section": "", "accepted_by": "", "accepted_at": ""}
```

## Siguiente acción
