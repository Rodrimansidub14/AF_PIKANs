# Proyecto de graduación

Repositorio principal del trabajo de graduación sobre mecanismos de focalización
adaptativa en arquitecturas PIKAN. El documento vigente se compila desde
`z-main.tex` con la plantilla UVG 2019.

## Límites del repositorio

| Ruta | Responsabilidad |
|---|---|
| `z-main.tex` y módulos `0-` a `o-` | Fuente principal del documento LaTeX |
| `figuras/` y `plantilla/` | Figuras del documento y recursos gráficos UVG |
| `m-bibliografia.bib` | Bibliografía del documento vigente |
| `src/` | Código ejecutable, pipelines y harness de experimentos |
| `notebooks/` | Exploración interactiva y experimentos en Jupyter |
| `PIKAN-Vault/` | Base de conocimiento y notas de investigación |
| `context/` | Ontología, contrato de notebooks y política de agentes |
| `.senpi/extensions/` | Adaptador local que conecta los contratos con cada prompt de Senpi |
| `presentaciones/` | Material de exposición y sus salidas |
| `recursos/` | Imágenes, referencias históricas y prototipos auxiliares |
| `reports/` | Informes; el material cerrado vive en `reports/archive/` |
| `tests/` | Pruebas del pipeline y graph engineering |
| `artifacts/` | Salidas regenerables, ignoradas por Git |
| `var/` | Estado temporal y context packs, ignorados por Git |

## Compilación local

Con MiKTeX instalado, use la receta de LaTeX Workshop `pdflatex → biber →
pdflatex ×2` sobre `z-main.tex`. Las salidas se escriben en
`artifacts/latex/`.

Los nombres de asesor y tribunal que aún no han sido confirmados están
marcados explícitamente en `0-datos_estudiante.tex`.

## Flujo de investigación con Senpi

El proyecto incluye una extensión local que valida la ontología, actualiza el
grafo de `PIKAN-Vault` y añade el contexto de research automation antes de cada
prompt. El router descubre `workflow/index.toml` y selecciona automáticamente
el flujo y la superficie según el lenguaje natural:

```text
resume las fuentes relevantes para el marco teórico
adapta el baseline Heat en el notebook y audítalo
ejecuta el manifiesto en dry-run
compara las métricas de las corridas
redacta un borrador del apartado usando evidencia aceptada
```

No es necesario usar `/workflow` ni `research-role=...` en el uso normal. La
directiva `research-role=...` queda disponible como override cuando se necesita
forzar una ruta concreta. Los gates científicos y la promoción de conocimiento
canónico siguen requiriendo decisión humana.

En Senpi, confía el proyecto si se solicita y ejecuta `/reload`. Luego
`/research-context` permite comprobar el estado del contexto. Las salidas
derivadas viven en `var/context/`; no se modifica `obs-data`.
