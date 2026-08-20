# ProyectoGrad agent entrypoint

Usa `context/` como fuente de control de research automation y descubre la
ruta activa desde `workflow/index.toml`. Los prompts se enrutan
automáticamente por lenguaje natural; no pidas al usuario que pegue reglas,
use `/workflow` o declare un rol para tareas ordinarias. Conserva
`research-role=...` únicamente como override explícito.

No modifiques `context/`, `workflow/`, `PIKAN-Vault/PIKAN-Vault`, `.senpi/`,
`var/context/` ni ninguna ruta fuera de este repositorio. La aprobación humana
sigue siendo necesaria para gates científicos, claims y conocimiento
canónico.
