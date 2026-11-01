# Siguiente secuencia de problemas

| Orden | Problema | Estado | Alcance inicial |
|---|---|---|---|
| Actual | EXP-0005: Heat + Burgers canónico | Screening completado; interpretación y preregistro siguiente pendientes | RBF fija, MLP PINN y B-spline PIKAN; tres semillas, Sobol fijo, 18/18 runs W&B sincronizados. Burgers es Raissi/PINNacle y conserva un ID de evaluación distinto del EXP-0002 estacionario. [Resultados y visualizaciones](../var/agent_runs/heat-burgers-architecture-screen/analysis/architecture-screen-comparison.html). |
| 1 | Allen–Cahn 1D | Borrador EXP-0003 | Extender trainer escalar; campo bloqueado hasta referencia BDF convergida. |
| 2 | DFG 2D-1 Re=20 | Borrador EXP-0004 | Extender red a 2 coordenadas/3 campos, geometría y salida; QoI/campo bloqueados hasta auditoría FEM. |
Pendiente | Poisson 2D | Conservado como pendiente | No descartar; priorizar después de definir dominio y condiciones del protocolo. |

## Orden de ingeniería

1. Implementar Allen–Cahn con hard transform y evaluación solo física; obtener
   tiempo real por semilla y revisar interfaz/energía antes de desbloquear campo.
2. Implementar DFG separando interior y cinco componentes de frontera; auditar
   las pérdidas de entrada, paredes/cilindro y salida en un smoke de precisión.
3. EXP-0005 confirmó asignaciones de workers independientes a los slots GPU 0
   y 1 en la configuración Kaggle de dos T4. Verificar nuevamente la forma del
   kernel en cada campaña; `devices=1` por proceso no reparte trabajo.
4. Diseñar las variantes RAD/RAR-D como comparaciones contra estos controles
   fijos, con pool held-out y presupuesto idéntico.
5. Preparar referencia numérica convergida (BDF Allen–Cahn; FEM DFG) en una
   pista aparte, sin desbloquear métricas de campo por disponibilidad de datos.

Los bundles descargados pasaron integridad ZIP y verificación de hashes; los
resultados incluyen `gpu_slot` por semilla y se sincronizaron 18/18 runs con
W&B. Esto verifica la asignación observada en EXP-0005; no sustituye la
verificación de hardware por campaña.

## Regla de adaptación entre problemas

EXP-0003 y EXP-0004 son **propuestas iniciales**, no especificaciones cerradas.
Tras Heat+Burgers, revisar sus pools, ejemplos procesados, pérdida por paso,
perfiles de tiempo, asignación GPU, ocupación/saturación de RBF y distribución
espacial del residuo diagnóstico antes de firmar los siguientes preregistros.
El resultado exploratorio mostró menor error de Heat con B-spline y menor error
global/frente de Burgers con RBF; no selecciona un ganador general. Usar las
señales y su variación para decidir la siguiente hipótesis, con el Sobol fijo
como control y sin alterar retrospectivamente los manifests.

- Transferir hallazgos de infraestructura y del modelo común: throughput,
  dtype, batching PINA, instalación, logging, precisión numérica y señales
  RBF que sí midan lo esperado.
- No transferir automáticamente densidad de puntos, normalizadores, pérdidas,
  transformación dura, geometría ni umbrales de una PDE a otra.
- Si el residuo de entrenamiento/diagnóstico muestra concentraciones
  reproducibles, convertir esa observación en una hipótesis explícita
  (p. ej., RAD/RAR-D o estratificación). Mantener el Sobol uniforme como
  control; evaluar el adaptativo como nueva variante con el mismo presupuesto.
- No escoger los siguientes parámetros mirando la evaluación final bloqueada.
  Si una métrica held-out exploratoria guía una nueva hipótesis, declararlo,
  congelar el conjunto de comparación y reservar la evaluación final.
- Cada cambio material produce un nuevo manifiesto/versionado y una entrada
  en `parameter-decisions.md` con evidencia, fecha y motivo. Conservar la
  propuesta original para comparar qué cambió y por qué.

Así, los resultados previos pueden modificar los siguientes experimentos sin
reescribir retrospectivamente los anteriores ni mezclar control y adaptación.
