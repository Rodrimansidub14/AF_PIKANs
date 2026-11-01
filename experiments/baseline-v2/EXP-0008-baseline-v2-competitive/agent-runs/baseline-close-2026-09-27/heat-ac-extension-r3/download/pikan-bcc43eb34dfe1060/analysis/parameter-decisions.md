# Decisiones de parámetros — EXP-0003 Allen–Cahn

Estado: configuración versionada para el piloto exploratorio EXP-0003. El contrato
del runner, el presupuesto y el smoke CPU están implementados; queda revisión del
bundle exacto antes de cualquier despacho remoto. No es preregistro confirmatorio,
baseline congelado ni aceptación científica.

| Decisión | Propuesta | Motivo / limitación |
|---|---|---|
| PDE | `D=0.001`, reacción `5(u-u³)` | Conserva el caso DeepXDE 1.12.2 ya declarado por el verifier. |
| Dominio | x∈[-1,1], t∈[0,1] | Caso canónico del proyecto. |
| Condiciones | Hard transform `x²cos(πx)+t(1−x²)N` | Impone IC y Dirichlet sin penalizaciones blandas. Comparación no equivale a PINN con trazas blandas. |
| Red inicial | RBF fija [2,16,16,1], 11 bases por arista | Control común con Heat/Burgers; no se afirma óptima para interfaces. |
| Puntos de entrenamiento | 8.192 interiores Sobol scrambled, pool fijo por semilla | Potencia de dos; cercana a los 8.000 interiores del ejemplo de referencia. Se guarda la matriz exacta y ningún punto held-out entra a la loss. |
| Residuo held-out | 16.384 puntos Sobol independientes, stream `10*seed+5` | Más puntos que entrenamiento para reducir ruido al describir el campo residual; sin uso para optimización ni selección de checkpoints. Guardar matriz y cuantiles espaciales/temporales. |
| Trazas duras | 512 tiempos por borde y 1.024 coordenadas iniciales Sobol | Solo medir desviación numérica de `u(±1,t)=-1` y `u(x,0)=x²cos(πx)`; no se añaden condiciones blandas ni puntos de pérdida. |
| Batch | 1.024 | Continuidad con campaña anterior; confirmar semántica de batching al implementar el runner. |
| Optimizador | Adam, `lr=0.001` | Primer control comparable dentro del proyecto; sin scheduler ni L-BFGS en el run inicial. |
| Presupuesto | 20.000 pasos o 1.200 s por semilla | Tope de 20 min y tres semillas: máximo 60 min de entrenamiento. |
| Streams | `10*seed+1` interior, `+2` diagnóstico, `+3` frontera, `+4` inicial, `+5` held-out | Streams reproducibles independientes; la inspección de cobertura usa coordenadas guardadas, sin remuestrear después de ver métricas. |
| Validación | Scrambled Sobol de 16.384 puntos distinto del entrenamiento | Diagnóstico físico held-out; no hay etiqueta de campo verificada. |
| Métricas | Residuo, trazas, energía y señales RBF/AD | La referencia BDF sigue `pending_convergence`; error L2 queda bloqueado. |
| Energía | `E=∫[D|u_x|²/2 + 5(u²−1)²/4]dx`, 513 nodos simétricos, `t={0,.25,.5,.75,1}` | 512 intervalos trapezoidales en `[-1,1]`, con `x=0` y extremos incluidos. Reportar variación y conteo de incrementos; no convertir monotonía en gate para una aproximación PINN. |
| Mapa visual | 257×129 en `[-1,1]×[0,1]` | Solo visualización Plotly; la métrica de residuo se estima en los 16.384 puntos held-out. No usar el mapa para escoger duración o checkpoint. |
| Semillas | 101, 102, 103 | Repeticiones piloto para dispersión; no potencia confirmatoria. |
| Precisión | Heredar inicialmente float64 del trainer; medir coste | Coherencia numérica con la primera campaña; T4 puede penalizar FP64. Revisar tras obtener tiempo real; cambiar implica nueva versión. |

No hay umbrales definidos. No ajustar la campaña mirando el score de evaluación.
La corrida queda en modo W&B offline; las credenciales se mantienen fuera del
notebook y del bundle.

## Puerta de actualización con EXP-0002

Revisar antes de firmar: costo por paso y puntos realmente vistos, criterio de
parada, inestabilidad/no finitos, saturación y cobertura de bases, sensibilidad
al dtype, residuo en regiones de entrenamiento y segundo T4 disponible. Estos
datos pueden cambiar batch, precisión, presupuesto y plan de diagnóstico.
No copiar la densidad local de puntos ni un umbral de residuo de Heat/Burgers a
Allen–Cahn: su interfaz temporal requiere una hipótesis propia. Si las señales
motivan más resolución espacial/temporal, crear una variante con idéntica
evaluación independiente y control Sobol conservado.


## Reproducibility note for r2

The r1 kernel reached Kaggle with 2xT4 and failed in the notebook's sampling-report cell before training: Allen-Cahn pools use boundary_probe/initial_probe keys, unlike Heat pools' left/right/initial keys. This r2 uses problem-specific pool keys in the reporting cell. Training seeds, PDEs, model, fixed Sobol streams, optimizer and compute cap are unchanged.
