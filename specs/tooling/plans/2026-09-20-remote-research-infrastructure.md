# Infraestructura de investigación remota

## Diseño y alcance autorizado

Implementar la arquitectura propuesta en la conversación: Codex prepara cambios,
Kaggle ejecuta lotes, Colab permite inspección interactiva y el harness conserva
evidencia. La autorización es para infraestructura local; no cambia la política,
no activa hipótesis, no publica código ni consume GPU remota.

Se conserva el workspace actual porque contiene trabajo del usuario sin registrar.
No se tocan AGENTS.md, context/, workflow/, .senpi/ ni el vault.

## Entregables

1. Tests de paquetes reproducibles, confinamiento de rutas, timeout, reanudación,
   validación de resultados y bloqueo de escrituras remotas.
2. `src/research_pipeline/remote/`: CLI prepare/run/status/submit/fetch/evaluate,
   adaptador Kaggle CLI y runtime portable de biblioteca estándar.
3. Paquete explícito de archivos Python (sin copiar el repositorio), manifiesto,
   hashes, notebook portable y metadata privada para Kaggle.
4. Corridas por semilla con snapshot, logs, estado durable y límite de tiempo.
   Una ejecución interrumpida no se vuelve a enviar automáticamente a Kaggle.
5. Evaluación numérica provisional: valores finitos, umbrales explícitos,
   artefactos y trazabilidad; ninguna aceptación automática de claims.
6. Configuración MCP oficial desactivada en `.codex/config.toml`, sin secretos;
   guía operativa de activación y dependencias opcionales.

## Verificación

Primero ejecutar tests que fallen por ausencia del nuevo CLI/comportamiento.
Después implementar y ejecutar los tests de infraestructura y la suite existente.
Realizar un smoke end-to-end local con el mismo runtime incluido en los notebooks.
No usar credenciales ni red en tests. Las pruebas del adaptador simulan solamente
el borde externo; el empaquetado y la ejecución se prueban con archivos/procesos reales.

## Límites

No se integra un LLM API ni una búsqueda autónoma de nuevas hipótesis. El agente
existente puede preparar candidatos y comparar artefactos mediante el CLI.
MCP es acceso interactivo; el adaptador CLI deja el contrato batch verificable.
Una conexión remota real y sus cuotas quedan pendientes de credenciales y permiso
explícito para escrituras externas. No se promete compatibilidad Colab/Codex sin
una prueba real del intercambio dinámico de herramientas.

## Resultado de implementación

Implementado el CLI portable, comparación pareada, adaptador Kaggle con guard de
política y registro antienvío duplicado, configuración MCP desactivada y guía.
Verificado con 66 pruebas Python y 23 pruebas JavaScript. Smoke final y evidencia
en `var/agent_runs/infra-smoke-final/verification.md`. La revisión independiente
detectó dos casos de integridad/reintento, corregidos con pruebas de regresión.
La integración con cuentas remotas permanece pendiente de autorización y credenciales.
