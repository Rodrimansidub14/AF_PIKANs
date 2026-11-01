# Infraestructura remota PIKAN

Esta capa prepara experimentos autocontenidos para Kaggle y Colab, registra
corridas por semilla y evalúa artefactos localmente. Los MCP oficiales se configuran
en `.codex/config.toml`; están desactivados. El transporte batch usa el CLI oficial
de Kaggle para poder probarlo sin depender de herramientas MCP dinámicas.

No se ha conectado ninguna cuenta ni ejecutado una GPU remota. No incluye un
entrenador PIKAN nuevo, búsqueda autónoma de hipótesis ni inferencia OpenRouter.
El agente Codex ya disponible puede usar estos comandos para cada candidato.

## Preparación local antes de autenticar

El entorno aislado está en `var/agent_runs/remote-env`. Incluye Kaggle CLI 2.2.4
y Colab MCP 1.0.1 desde la revisión oficial
`b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`. Todas las dependencias resueltas están
fijadas en `requirements.lock` para Windows/Python 3.13. No se usa un `uvx` que
descargue una revisión nueva al iniciar el servidor.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File src/research_pipeline/remote/setup.ps1
py -3.12 -m src.research_pipeline.remote doctor
```

El instalador requiere `uv` y Python 3.13 ya disponibles; no instala Python
globalmente. Es repetible y mantiene entorno, cachés y temporales en el repositorio.
El adaptador encuentra Kaggle dentro de ese entorno sin activar PATH.
`doctor` inspecciona metadata de paquetes sin leer credenciales, importar clientes
de proveedores o contactar servicios. `authentication=deferred` es intencional.
El servidor local de Colab completó `initialize` y `tools/list` en una prueba MCP
real sin abrir navegador ni llamar herramientas de autenticación. Esto no prueba
la conexión del navegador ni las notificaciones dinámicas del cliente Codex.

## Recorrido sin credenciales

Desde la raíz del repositorio, con Python 3.11+ y PyYAML:

```powershell
py -3.12 -m src.research_pipeline.remote prepare --manifest src/research_pipeline/examples/infra_smoke_manifest.json --out var/agent_runs/infra-smoke/bundle --kernel researcher/pikan-smoke --file src/research_pipeline/examples/infra_smoke.py --evidence-kind infrastructure --timeout 10
py -3.12 -m src.research_pipeline.remote run --bundle var/agent_runs/infra-smoke/bundle --out var/agent_runs/infra-smoke/runs
py -3.12 -m src.research_pipeline.remote evaluate --bundle var/agent_runs/infra-smoke/bundle --runs var/agent_runs/infra-smoke/runs --out var/agent_runs/infra-smoke/evaluation.json
```

`researcher/pikan-smoke` es un identificador de ejemplo para la prueba offline;
no se contacta Kaggle. Para preparar un envío real se necesita el usuario real y
un slug nuevo. `prepare` no sobreescribe paquetes existentes. `run` reanuda los
completados intactos y no reintenta fallos salvo `--retry-failed`. Los intentos
anteriores se conservan. Un lock que sobrevive a una interrupción requiere revisar
que el proceso terminó antes de eliminar manualmente `.running.lock`.

El smoke comprueba aritmética y transporte, **no entrena ni valida una PIKAN**.
Su informe siempre declara `evidence_kind=infrastructure`,
`eligible_for_review=false` y `scientific_accepted=false`.

## Contrato de un experimento real

1. Preparar un manifiesto específico en `var/agent_runs`, conservando los contratos
   de `context/` y el alcance aprobado de `workflow/index.toml`.
2. `model.implementation` identifica el entrypoint Python; incluirlo y cada módulo
   local requerido con un `--file src/...py` separado. No hay glob, copia del vault
   ni subida implícita de todo el repositorio. Revisar los archivos antes de enviar:
   una allowlist no detecta secretos escritos dentro del código.
3. Definir `training.seeds`, métricas con `direction` y `threshold`, y
   `artifacts.expected`. Los umbrales son una decisión científica humana.
4. El entrenador lee `HARNESS_MANIFEST` (JSON, también YAML válido),
   `HARNESS_RUN_DIR` y `HARNESS_SEED`; configura los RNG de Python/NumPy/PyTorch.
   Debe escribir `metrics.json`, `decision_note.md` y los artefactos declarados.
   El runtime escribe `manifest_snapshot.json` y `run_summary.json`.
5. Preparar con `--evidence-kind research`, un `--timeout` por semilla y, si
   corresponde, `--gpu`. Límites del paquete: 1–32 semillas únicas y hasta 3600
   segundos por semilla. El manifiesto y todo archivo enviado tienen SHA-256.
6. El investigador revisa el paquete y registra una aprobación explícita.

Ejemplo del formato de aprobación, guardado dentro de `var/agent_runs`:

```json
{
  "bundle_id": "SHA256 mostrado por prepare",
  "approved_by": "Nombre del investigador",
  "scope": "approved",
  "hypothesis": "approved",
  "code_surface": "approved",
  "remote_execution": true
}
```

El agente no debe completar este registro en nombre del investigador sin su
decisión explícita. Es un registro de autorización, no una firma criptográfica ni
un mecanismo de aislamiento frente a código malicioso. Un cambio de manifiesto,
código, semillas o presupuesto genera otro ID y requiere revisar esa campaña.

Los entrenadores ejecutados son código de confianza. Los límites de rutas del CLI
y el timeout no impiden que un entrenador arbitrario realice sus propias llamadas
de red o escrituras. La CPU/GPU y las dependencias deben estar disponibles en el
runtime; no se instalan paquetes automáticamente ni se habilita internet en Kaggle.

## Activación de Kaggle

El investigador aprobó explícitamente el 2026-09-20 conectar Kaggle, subir
notebooks privados y ejecutar campañas previamente aprobadas. La política y
AGENTS.md ya reflejan esta excepción restringida bajo `[scope]`:

```toml
external_write_access = "approved_services"
external_write_services = ["kaggle"]
external_write_actions = ["private_notebook_upload", "approved_campaign_run"]
```

Los demás gates se mantienen y una campaña específica aún necesita su registro
de aprobación. Las herramientas MCP directas no pasan por el guard del CLI;
permanecen desactivadas hasta completar la autenticación y revisar las herramientas expuestas.
La política por sí sola no es un sandbox universal de proveedores externos.

Como último paso de conexión, con credenciales configuradas por el usuario:

- Usar `KAGGLE_API_TOKEN` en el entorno del proceso; no guardarlo en Git o notebooks.
- El entorno y las dependencias ya están instalados mediante `setup.ps1` y su lock.
- Activar `mcp_servers.kaggle.enabled` en `.codex/config.toml` y reiniciar la conexión
  del cliente. La configuración local puede requerir confiar en el proyecto.
- Comprobar herramientas, identidad de cuenta y acceso a aceleradores. La cuota
  se consulta en la cuenta; no se supone una cantidad fija de horas.

```powershell
py -3.12 -m src.research_pipeline.remote submit --bundle var/agent_runs/CAMPAIGN/bundle --approval var/agent_runs/CAMPAIGN/approval.json
py -3.12 -m src.research_pipeline.remote status --bundle var/agent_runs/CAMPAIGN/bundle
py -3.12 -m src.research_pipeline.remote fetch --bundle var/agent_runs/CAMPAIGN/bundle --out var/agent_runs/CAMPAIGN/download
```

No hay polling continuo. `status` consulta una vez. `fetch` descarga a una carpeta
nueva y verifica el ID de campaña y artefactos; rechaza resultados de otra versión
del notebook. Conserva `evaluation.json` junto a la descarga. El CLI aísla su
directorio de configuración bajo `var/agent_runs/kaggle-config`.

### Monitor local de campaña

El dashboard de solo lectura puede consultar el estado Kaggle periódicamente,
mostrar tiempo transcurrido/límite y leer resúmenes/logs una vez descargados:

```powershell
$env:AF_PIKAN_CAMPAIGN = 'var/agent_runs/heat-burgers-002-t4x2'
$env:AF_PIKAN_BUNDLE = 'var/agent_runs/heat-burgers-002-t4x2/bundle-final-20260923'
$env:AF_PIKAN_STATUS_SCRIPT = 'var/agent_runs/run_heat_burgers_t4x2.py'
var/agent_runs/remote-env/Scripts/python.exe src/research_pipeline/remote/dashboard.py --port 8765
```

Abrir `http://127.0.0.1:8765`. Solo enlaza a localhost y nunca devuelve
credenciales. Para la campaña actual, el estado de Kaggle es la única señal
disponible durante el entrenamiento: usa W&B offline; tras la descarga, muestra
las semillas y métricas locales, además del resultado de `wandb-sync.json`.
`AF_PIKAN_CAMPAIGN`, `AF_PIKAN_BUNDLE` y `AF_PIKAN_STATUS_SCRIPT` permiten
apuntar el monitor a otra corrida. Los bundles nuevos transmiten `stdout`/`stderr`
y guardan una copia local; para gráficas W&B durante el entrenamiento también se
requiere modo online y `WANDB_API_KEY` en Kaggle Secrets. El monitor no cambia
un kernel que ya está ejecutándose.

`submission.json` evita repetir un envío. Si la respuesta es incierta, el estado es
`submission_unknown`: consultar Kaggle y reconciliar antes de actuar; nunca borrar
el registro para reintentar ciegamente. Utilizar un kernel diferente por paquete
reduce conflictos con versiones concurrentes. La campaña exploratoria
Heat+Burgers T4x2 se autenticó, subió, ejecutó y descargó con artefactos
verificados; la evaluación científica continúa pendiente de interpretación y
umbrales explícitos.

## Colab

El mismo `experiment.ipynb` es autocontenido y se puede abrir en Colab. El MCP
oficial conecta un agente local a una sesión de navegador. Antes de activarlo hay
que comprobar la compatibilidad de `notifications/tools/list_changed` del cliente.
La configuración conserva `enabled=false`. El launcher local configura logs y
temporales dentro del repositorio antes de importar el servidor oficial. Desactiva
su comprobación automática de actualizaciones. No llama herramientas de navegador.

Dependencias y revisión ya están fijadas. Cuando se autorice el uso remoto de
Colab: abrir el notebook y comprobar una celda simple antes de ejecutar
entrenamiento. La excepción actual de escrituras remotas cubre solo Kaggle.
Las salidas quedan en
`pikan-<id>/runs` dentro del runtime; descargarlas al repositorio y usar `evaluate`.
La conexión MCP no proporciona una GPU ni elimina las cuotas o interrupciones.

## Evaluación y comparación

`evaluate` comprueba semillas esperadas, procedencia, snapshots, hashes, artefactos,
estado y métricas finitas. Calcula media y desviación muestral. `thresholds_pass`
significa que todas las corridas cumplieron los umbrales declarados.

```powershell
py -3.12 -m src.research_pipeline.remote compare --baseline-bundle var/agent_runs/BASE/bundle --baseline-runs var/agent_runs/BASE/runs --candidate-bundle var/agent_runs/CAND/bundle --candidate-runs var/agent_runs/CAND/runs --out var/agent_runs/comparison.json
```

Compara semillas emparejadas, el mismo problema y referencia, entrenamiento,
presupuesto y contrato de métricas. Una mejora pareada positiva favorece al
candidato. Reporta el cociente de tiempo observado, pero deja `cost_gate_pass=null`
hasta confirmar hardware/carga comparables. Las semillas no sustituyen distintas
configuraciones físicas. No se infiere significancia estadística, SOTA ni aceptación
científica. `eligible_for_review` solo indica que existe evidencia numérica para
revisar; los checks pendientes aparecen en el informe.

## Pruebas

```powershell
py -3.12 -m unittest tests.test_remote_research -v
```

El notebook y los subprocesos se ejecutan realmente en las pruebas. Solo se simula
el transporte Kaggle. No se usan credenciales, GPU ni servicios externos.

Fuentes oficiales consultadas para los adaptadores:

- https://www.kaggle.com/docs/mcp
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- https://github.com/googlecolab/colab-mcp
- https://developers.openai.com/codex/mcp/
