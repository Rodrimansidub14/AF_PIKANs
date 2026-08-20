# Enrutamiento automático de workflows para Senpi

## Objetivo

Permitir que un prompt normal en `ProyectoGrad` active automáticamente el
workflow y el rol de research automation que corresponden a su intención. El
usuario no tendrá que anteponer `/workflow`, escribir `research-role=...` ni
enumerar reglas operativas en cada solicitud. La directiva explícita seguirá
existiendo como override para casos excepcionales.

## Alcance y restricciones

- El cambio vive únicamente en `C:/Users/rodri/Documents/ProyectoGrad`.
- `obs-data` permanece fuera de alcance y no se lee ni modifica.
- `PIKAN-Vault/PIKAN-Vault` continúa siendo corpus canónico de solo lectura
  para agentes; sus derivados siguen en `var/context/`.
- La instalación global de Senpi no se modifica.
- `context/` y `workflow/` son configuración canónica: los agentes pueden
  leerlos, pero no escribirlos.
- Se conserva `L2_human_verified_ai_execution`, la autoridad humana y todos
  los gates científicos existentes.
- El router selecciona contexto y superficie declarada; no concede por sí
  mismo aprobación para modificar una PDE, el modelo central, un claim o
  conocimiento canónico.

## Diseño

### Registro declarativo

Se crea `workflow/index.toml` como catálogo único de workflows. Cada entrada
declara:

- `id`, descripción y prioridad;
- rol de agente asociado;
- frases o tokens de intención que la activan;
- exclusiones para evitar colisiones entre workflows;
- fases operativas que el agente debe seguir;
- gates que debe mostrar antes de ejecutar acciones sensibles.

`workflow/README.md` documenta el catálogo para humanos, pero no es necesario
copiarlo al prompt. El adaptador proyecta del registro solo los campos seguros
y compactos que necesita el runtime.

El registro inicial cubrirá estas rutas:

| Workflow | Rol | Intención principal |
|---|---|---|
| `literature` | `literature_scout` | fuentes, conceptos, marco teórico y estado del vault |
| `hypothesis` | `hypothesis_planner` | pregunta, hipótesis, falsación y manifiesto |
| `notebook` | `notebook_engineer` | crear o adaptar notebooks dentro de su superficie |
| `run` | `experiment_runner` | validar o ejecutar manifiestos y registrar corridas |
| `evaluation` | `evaluator` | métricas, comparación y gates de resultados |
| `synthesis` | `research_synthesizer` | informe, borrador LaTeX y claims sustentados |

### Selección automática

La extensión local mantiene el ciclo actual de validación y actualización del
grafo, pero cambia la selección de rol:

1. Carga el `senpi-context.json` generado por `src.graph_engineering`.
2. Lee las rutas declaradas del catálogo de workflows proyectadas en ese pack.
3. Si el prompt tiene `research-role=...` o `/research ...`, valida y usa ese
   override.
4. Si no hay override, normaliza el prompt y calcula una puntuación
   determinista por coincidencia de triggers, exclusiones y prioridad.
5. Selecciona el workflow ganador y su rol, incluyendo sus fases y gates en el
   contexto del agente.
6. Si no hay coincidencia clara, usa `literature` como fallback conservador.

La puntuación y el workflow seleccionado quedan visibles en el contexto
inyectado para que la decisión sea auditable. El agente recibe el workflow
activo, no una copia completa de todos los workflows.

Para prompts con varias intenciones, el registro permite declarar fases. El
router conserva un workflow primario y entrega sus fases ordenadas; no fusiona
permisos arbitrariamente entre roles. Si una fase posterior requiere otra
superficie o gate, el agente debe detenerse y solicitar la decisión humana.

### Integración con los contratos

`src/graph_engineering/context.py` incluirá en el paquete derivado:

```json
{
  "workflow": {
    "default": "literature",
    "ambiguity_policy": "conservative",
    "items": [
      {
        "id": "notebook",
        "role": "notebook_engineer",
        "priority": 30,
        "triggers": ["notebook", "adapta", "baseline"],
        "excludes": ["solo explica"],
        "stages": ["inspect_contract", "edit_declared_surface", "audit"]
      }
    ]
  }
}
```

Los permisos efectivos siguen viniendo de `context/agent-policy.toml`; el
registro no duplica `read_paths` ni `write_paths`. El core de Senpi verifica
que el workflow referencie un rol existente antes de usarlo.

`workflow/` se añade a la huella de fuentes para que cualquier cambio
invalide el pack durante la sesión y fuerce una regeneración validada.

### Seguridad y comportamiento ante errores

- Un workflow desconocido o mal formado no habilita escritura.
- Si falla la validación de contratos, el manifiesto o el build del grafo, se
  conserva el estado fail-closed actual.
- Un prompt ambiguo puede leer el contexto bajo `literature_scout`, pero no
  obtiene automáticamente una superficie de edición de notebooks ni permisos
  de ejecución costosa.
- La política de archivos trata `workflow/` igual que `context/`: lectura
  permitida para el contexto del agente y escritura denegada.
- El guard de Bash mantiene los bloqueos de `obs-data`, `.git`, `.senpi`,
  `context/`, `workflow/`, el vault y `var/context/`.

### Compatibilidad

Se conserva el formato actual de directivas explícitas para no romper prompts
existentes. `AGENTS.md` será un punto de entrada mínimo para agentes que
respetan convenciones de repositorio: indicará que deben descubrir y seguir
`context/` y `workflow/`, sin duplicar allí la ontología ni los permisos.

## Flujo de datos

```text
prompt natural
  → extensión Senpi
  → contratos + manifiesto + grafo PIKAN
  → workflow/index.toml proyectado
  → selector determinista
  → workflow + rol + fases + gates
  → system prompt compacto
  → herramientas con política de archivos
```

## Pruebas y aceptación

El cambio se acepta cuando:

1. Un prompt de literatura selecciona `literature_scout` sin directiva.
2. Un prompt de notebook selecciona `notebook_engineer` sin directiva.
3. Prompts de hipótesis, ejecución, evaluación y síntesis seleccionan sus
   respectivos workflows.
4. El override explícito sigue funcionando y un rol desconocido sigue
   bloqueando la tarea.
5. Un prompt sin coincidencia usa el fallback conservador.
6. El contexto renderizado muestra el workflow activo, sus fases y gates.
7. Las pruebas Node y Python existentes y nuevas pasan.
8. La validación de contratos, la validación del manifiesto y el build del
   contexto pasan.
9. Las comprobaciones de estado confirman que no hubo cambios en `obs-data`,
   la instalación global de Senpi, el vault canónico ni rutas fuera del repo.

