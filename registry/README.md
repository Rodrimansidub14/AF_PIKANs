# registry/ — memoria del loop

Append-only. Nunca se edita una línea existente; se añade una nueva.

| Archivo | Contenido |
|---|---|
| `runs.jsonl` | Toda corrida ejecutada: manifiesto, suite, semillas, scorecard, costo |
| `proposals.jsonl` | Qué propuso el agente, por qué, y qué pasó |
| `failures.jsonl` | Clusters de fallo con su causa diagnosticada |
| `claims.jsonl` | Claim aceptado por el humano -> run -> artefacto -> sección LaTeX |

`proposals.jsonl` es lo que convierte esto en un loop y no en una serie de
corridas sueltas: es la memoria que impide que el agente vuelva a proponer lo
que ya falló.

`claims.jsonl` es el puente evidencia->escritura. Sin él, en tres meses no se
sabe de qué corrida salió cada figura de la tesis.

Solo un humano escribe en `claims.jsonl`.
