# EXP-0003 — Allen–Cahn con RBF-PIKAN

**Estado:** configuración v1 para exploración; no es preregistro confirmatorio,
no está ejecutada y no establece un baseline congelado.  
**Rol:** exploración de señales, sin claim científico ni baseline congelado.

Esta versión fija los parámetros para que el bundle sea reproducible. Las
señales de EXP-0002 (batches reales, costo, uso de GPU, gradientes,
ocupación/saturación RBF y residuo diagnóstico) podrán motivar otra versión
antes de observar resultados de Allen–Cahn. Cualquier variante adaptada se
registrará por separado, con Sobol fijo conservado como control.

## Pregunta

¿Qué señales de muestreo, derivadas, residuo, bases y estabilidad aparecen al
entrenar el problema Allen–Cahn 1D con la configuración RBF-PIKAN fija del
proyecto?

## Problema y condiciones

Se conserva el contrato `allen_cahn_deepxde_1_12_2_dirichlet`:

\[
u_t - 0.001u_{xx} - 5(u-u^3)=0,\qquad (x,t)\in[-1,1]\times[0,1],
\]

con `u(x,0)=x² cos(πx)` y `u(−1,t)=u(1,t)=−1`. La transformación dura
`u=x² cos(πx)+t(1−x²)N(x,t)` impone las tres trazas exactamente. Por ello,
las pérdidas de frontera e inicial no se añaden como términos blandos.

## Protocolo inicial propuesto

- Pool fijo, scrambled Sobol 2D, **8.192 puntos interiores**; stream y matriz
  exacta guardados por semilla.
- Evaluación de residuo en **16.384 puntos Sobol held-out** de un stream
  independiente. Sus coordenadas se guardan; no se usan para optimizar.
- Tres semillas independientes; batch 1.024; Adam, `lr=1e-3`.
- Presupuesto por semilla: **20.000 pasos o 20 minutos**, lo que ocurra primero.
- Sin RAD/RAR-D en este control. La comparación adaptativa vendrá luego bajo
  el mismo presupuesto y en una campaña distinta.
- Verificación held-out con un segundo Sobol independiente; residuo físico,
  concentración espacial/temporal, energía Allen–Cahn y señales internas.
- Medir energía `E[u]=∫[D|u_x|²/2 + 5(u²−1)²/4]dx` en 513 nodos simétricos
  (incluyen `x=0` y ambos bordes), para `t=0,0.25,0.5,0.75,1`. La curva es
  descriptiva; no es un criterio de aceptación.
- Un mapa Plotly de 257×129 es solo para inspección visual; los valores de
  residuo confirmatorios del piloto provienen del Sobol held-out, no de ese mapa.
- **No reportar error de campo ni aceptar calidad científica:** la referencia
  numérica BDF está bloqueada hasta completar estudio de convergencia.

El manifiesto de campaña fija todas estas cantidades. La ejecución exploratoria
fue autorizada con tres semillas y un máximo de dos workers; Kaggle exige además
revisión humana ligada al hash del bundle final. El presupuesto agregado
es un máximo de 3.600 GPU-segundos de entrenamiento (tres semillas × 1.200 s),
con un máximo de dos workers simultáneos en dos T4; setup y cola no cuentan como
tiempo de entrenamiento. W&B opera offline dentro de Kaggle y se sincroniza
posteriormente desde el repositorio.

## Señales y cierre

Registrar loss del residuo, residuo normalizado y sus cuantiles, gradiente,
activaciones/saturación, ocupación y norma de bases, sensibilidad física,
tiempo por paso/diagnóstico, ejemplos procesados y punto de parada. Guardar
campos en tiempos fijos y energía diagnóstica, sin umbral de aprobación.

El cierre compara semillas y coste; produce una tabla de fallos y preguntas
para la siguiente iteración. No usa el error de la malla final como mecanismo
para extender presupuesto. Finaliza con revisión humana.

## Falsación y límites

Una señal interna no reproducible, residuos no finitos o trazas violadas por
la transformación invalidan la corrida técnica. Un residuo bajo por sí solo
no demuestra solución correcta ni superioridad; el campo permanece sin
referencia desbloqueada.

## Fuentes

- [DeepXDE Allen–Cahn example](https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html): ecuación, datos y hard transform del caso.
- [DeepXDE reference implementation](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Allen_Cahn.py): 8.000 dominio, 400 frontera, 800 iniciales como escala de referencia; nuestra transformación dura elimina los pools de traza.
- [Wu et al., sampling study](https://arxiv.org/abs/2207.10289): RAD/RAR-D son candidatos para una comparación posterior, no se mezclan en el control fijo.
