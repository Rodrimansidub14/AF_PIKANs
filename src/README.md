# Código fuente

| Ruta | Responsabilidad | Escritura del agente |
|---|---|---|
| `mechanisms/` | Mecanismos de focalización | **sí, sin gate** |
| `research_pipeline/` | Validación, ejecución local, gates, remoto | no |
| `harness_loop/` | Manifiestos y ejecución controlada de corridas | no |
| `graph_engineering/` | Validación de contratos y grafo de PIKAN-Vault | no |

`mechanisms/` es la única superficie de código que el proposer puede escribir
sin aprobación humana. Un mecanismo **no puede importar `evals`**: un candidato
que conoce su métrica la optimiza en vez del problema.
`tests/test_mechanism_isolation.py` lo verifica de forma estática.

Los comandos se invocan desde la raíz usando rutas `src/...`. Las salidas de
corridas van a `artifacts/` o `var/`, nunca a los módulos de fuente.
