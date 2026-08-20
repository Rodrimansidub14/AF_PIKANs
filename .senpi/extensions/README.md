# Integración Senpi del proyecto

Esta extensión es local a `ProyectoGrad`. Senpi la descubre desde
`.senpi/extensions/` cuando el proyecto está confiado.

## Activación

1. Abrir Senpi con este repositorio como directorio de trabajo.
2. Aceptar la confianza del proyecto si Senpi la solicita.
3. Ejecutar `/reload`.
4. Ejecutar `/research-context` para comprobar el grafo y los contratos.

## Enrutamiento automático

La extensión lee `workflow/index.toml` y selecciona el workflow y rol por
intención. No es necesario escribir `/workflow`, `research-role=...` ni una
lista de reglas en cada prompt:

```text
resume las fuentes del marco teórico
adapta el baseline Heat en el notebook
ejecuta el manifiesto en dry-run
compara las métricas de las corridas
redacta un borrador con evidencia aceptada
```

El router usa `literature_scout` como fallback conservador. La directiva
`research-role=...` sigue disponible como override explícito cuando se necesita
forzar una ruta. La selección no concede gates científicos: las superficies
declaradas en `context/agent-policy.toml` y la aceptación de hipótesis,
corridas, claims y cambios de conocimiento canónico permanecen bajo control
del investigador.

## Seguridad y derivados

El adaptador ejecuta los comandos locales de `src.graph_engineering`, lee
`PIKAN-Vault/PIKAN-Vault` y escribe únicamente artefactos regenerables en
`var/context/`. Las escrituras de Senpi a `context/`, `workflow/`, el vault, `.senpi/`,
`.git/`, `var/context/` y cualquier ruta fuera del repositorio son rechazadas.
`obs-data` queda fuera del proyecto y fuera del alcance del adaptador.
