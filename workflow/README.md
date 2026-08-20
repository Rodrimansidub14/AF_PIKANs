# Workflows de research automation

Senpi descubre automáticamente `index.toml` al iniciar un prompt dentro de
`ProyectoGrad`. El router selecciona el flujo y rol más coherentes con el
lenguaje natural; no es necesario escribir `/workflow` ni `research-role=...`
para tareas ordinarias.

Flujos disponibles:

- `literature`: fuentes, conceptos y marco teórico.
- `hypothesis`: preguntas, hipótesis falsables y manifiestos.
- `notebook`: creación o adaptación acotada de notebooks.
- `run`: validación y ejecución reproducible.
- `evaluation`: métricas, comparaciones y gates.
- `synthesis`: borradores de informes, LaTeX y claims sustentados.

La sintaxis `research-role=...` permanece como override explícito cuando la
intención automática debe corregirse. El registro es configuración canónica:
los agentes lo leen, pero no lo modifican.
