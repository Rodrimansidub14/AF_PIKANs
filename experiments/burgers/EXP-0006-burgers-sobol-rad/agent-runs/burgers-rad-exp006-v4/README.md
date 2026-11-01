# EXP-0006 · relanzamiento de cuenta Kaggle

- Estado al preparar este registro: **enviado; kernel RUNNING**.
- Kernel privado: `rodrimandub/af-pikans-burgers-sobol-rad-exp006-rerun-v4`.
- Bundle ID: `4393e173686298ee83759eb20d320a1f65d320db0ac6bb2ecdf6aad073563938`.
- Paquete: `bundle-v6-gpu-20260926`.
- Reautenticación: consulta Kaggle de solo lectura satisfactoria para el usuario configurado localmente. No se copia ni registra ningún token.
- Motivo: la cuenta anterior se eliminó y la salida del kernel previo quedó inaccesible/incompleta.

## Diseño conservado

Se reutilizaron byte por byte todos los archivos de investigación listados en `bundle.json` del paquete v5. La verificación de esos hashes fue satisfactoria. Solo se regeneraron la identidad del bundle, el notebook de transporte, los metadatos privados del kernel y sus hashes para apuntar al nuevo usuario/slug. No se modificaron PDE, referencia, arquitectura, muestreo, semillas, umbrales, métricas, optimizador ni presupuesto.

- Burgers 1D canónico; RBF-PIKAN fija.
- Sobol fijo frente a RAD; cinco pares reales (201–205), diez jobs totales.
- `NvidiaTeslaT4` (T4×2), hasta dos jobs concurrentes; cada job limitado por la configuración del manifiesto.
- Registro W&B offline por job y sincronización local al recuperar los artefactos.
- La comparación primaria, tablas cuantitativas y visualizaciones Plotly están en el notebook y el protocolo del bundle.
- La corrida sigue siendo exploratoria: `scientific_accepted=false`; no equivale por sí sola a evidencia SOTA.

## Salida y seguimiento

- Aprobación ligada al bundle exacto: `approval-gpu-v6-20260926.json`.
- Estado automatizado: `watcher-status.json`.
- Descargas/reintentos: `download-attempts/`.
- Monitor local: http://127.0.0.1:8765/.
- Decisiones completas de parámetros: `bundle-v6-gpu-20260926/manifest.json` y `approval-gpu-v6-20260926.json`.

El intento anterior permanece preservado en `burgers-rad-exp006-v3/analysis-recovery-20260926/`; contenía cuatro jobs RAD completos y no contenía los artefactos Sobol requeridos para el análisis pareado.
