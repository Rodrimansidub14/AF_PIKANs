# Preparación local y permiso Kaggle

Fecha: 2026-09-20.

El investigador pidió dejar credenciales/autenticación al final y después aprobó
explícitamente conectar Kaggle, subir notebooks privados y ejecutar campañas
previamente aprobadas. La excepción quedó en AGENTS.md y context/agent-policy.toml.
No se modificaron los gates científicos ni se autorizó escritura remota en Colab.

Instalación local:

- Python 3.13.5, entorno `var/agent_runs/remote-env`.
- Kaggle CLI 2.2.4.
- Colab MCP 1.0.1, revisión b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1.
- 116 dependencias fijadas en requirements.lock.
- setup.ps1 ejecutado y repetido sin reinstalaciones innecesarias.
- doctor: local_dependencies_ready=true, kaggle_permission_approved=true.

Verificación:

- 71 pruebas Python completas aprobadas y 23 JavaScript.
- Contratos: cero errores y cero advertencias.
- Prueba del comando exacto de .codex/config.toml: MCP initialize y tools/list
  exitosos, herramienta open_colab_browser_connection disponible.
- Un lanzamiento intermedio mediante py/execve cerró stdio en Windows; sustituido
  por ejecución directa del Python local y verificado nuevamente.
- Sin procesos Colab residuales tras la prueba.
- Authlib emite avisos de deprecación no fatales; no impiden el handshake.

No se llamó la herramienta que abre el navegador. No se leyeron credenciales,
no se autenticó ninguna cuenta y no se lanzó un experimento remoto.
MCP Kaggle y Colab permanecen desactivados hasta el paso final de conexión.
La compatibilidad Codex con cambios dinámicos de herramientas de Colab y la
ejecución GPU requieren la futura sesión autenticada.

El registro `colab-probe.json` conserva el resultado de la última prueba local.
