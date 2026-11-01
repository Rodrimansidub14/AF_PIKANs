# Decisiones de parámetros — EXP-0003 Allen–Cahn

Estado: piloto exploratorio EXP-0003 ejecutado con el bundle indicado al final de
este documento. No es preregistro confirmatorio, baseline congelado ni aceptación
científica; la métrica primaria no tenía umbral y el error de campo siguió bloqueado.

| Decisión | Elección ejecutada | Motivo / limitación |
|---|---|---|
| PDE | `D=0.001`, reacción `5(u-u³)` | Conserva el caso DeepXDE 1.12.2 ya declarado por el verifier. |
| Dominio | x∈[-1,1], t∈[0,1] | Caso canónico del proyecto. |
| Condiciones | Hard transform `x²cos(πx)+t(1−x²)N` | Impone IC y Dirichlet sin penalizaciones blandas. Comparación no equivale a PINN con trazas blandas. |
| Red inicial | RBF fija [2,16,16,1], 11 bases por arista | Control común con Heat/Burgers; no se afirma óptima para interfaces. |
| Puntos de entrenamiento | 8.192 interiores Sobol scrambled, pool fijo por semilla | Potencia de dos; cercana a los 8.000 interiores del ejemplo de referencia. Se guarda la matriz exacta y ningún punto held-out entra a la loss. |
| Residuo held-out | 16.384 puntos Sobol independientes, stream `10*seed+5` | Más puntos que entrenamiento para reducir ruido al describir el campo residual; sin uso para optimización ni selección de checkpoints. Guardar matriz y cuantiles espaciales/temporales. |
| Trazas duras | 512 tiempos por borde y 1.024 coordenadas iniciales Sobol | Solo medir desviación numérica de `u(±1,t)=-1` y `u(x,0)=x²cos(πx)`; no se añaden condiciones blandas ni puntos de pérdida. |
| Batch | 1.024 | Continuidad con la campaña anterior; el costo y número de pasos se midieron por semilla. |
| Optimizador | Adam, `lr=0.001` | Primer control comparable dentro del proyecto; sin scheduler ni L-BFGS en el run inicial. |
| Presupuesto | 20.000 pasos o 1.200 s por semilla | Tope de 20 min y tres semillas: máximo 60 min de entrenamiento. |
| Streams | `10*seed+1` interior, `+2` diagnóstico, `+3` frontera, `+4` inicial, `+5` held-out | Streams reproducibles independientes; la inspección de cobertura usa coordenadas guardadas, sin remuestrear después de ver métricas. |
| Validación | Scrambled Sobol de 16.384 puntos distinto del entrenamiento | Diagnóstico físico held-out; no hay etiqueta de campo verificada. |
| Métricas | Residuo, trazas, energía y señales RBF/AD | La referencia BDF sigue `pending_convergence`; error L2 queda bloqueado. |
| Energía | `E=∫[D|u_x|²/2 + 5(u²−1)²/4]dx`, 513 nodos simétricos, `t={0,.25,.5,.75,1}` | 512 intervalos trapezoidales en `[-1,1]`, con `x=0` y extremos incluidos. Reportar variación y conteo de incrementos; no convertir monotonía en gate para una aproximación PINN. |
| Mapa visual | 257×129 en `[-1,1]×[0,1]` | Solo visualización Plotly; la métrica de residuo se estima en los 16.384 puntos held-out. No usar el mapa para escoger duración o checkpoint. |
| Semillas | 101, 102, 103 | Repeticiones piloto para dispersión; no potencia confirmatoria. |
| Precisión | float64 (`64-true`) | Coherencia numérica con la primera campaña; el costo observado quedó entre 714 y 774 s por semilla. Cambiar dtype exige una nueva versión experimental. |

No hay umbrales definidos. No ajustar la campaña mirando el score de evaluación.
La corrida se inició en modo W&B offline; los tres runs se sincronizaron después
desde la copia de resultados verificada. Las credenciales se mantienen fuera del
notebook y del bundle.

## Evidencia de ejecución y resultados medidos

- Bundle remoto: `7d9fafd823f3e3150907ed029be678103e23c129fcab265aba61da9bff233338`.
- Kaggle kernel: `rodrigomansilla/af-pikans-allen-cahn-exp003`.
- Semillas `101`, `102`, `103`: 3/3 completadas; 20.000 pasos por semilla,
  criterio de parada `step_limit`; duración media 753,3 ± 33,8 s (media ± DE
  muestral), sin llegar al límite de 1.200 s.
- W&B: 3/3 runs offline sincronizados, códigos `g7w2635j`, `hc0vt58r`,
  `8eerje4z`; los 223 artefactos con hash por semilla verificaron correctamente.
- Residuo normalizado medio en el conjunto independiente: 1,273×10⁻³ ±
  1,146×10⁻³; rango 4,545×10⁻⁴–2,583×10⁻³. RMSE absoluto del residual held-out:
  1,098×10⁻² ± 1,036×10⁻². Son señales descriptivas de tres semillas, no una
  comparación ni evidencia de precisión de campo.
- Energía en cinco tiempos: E(0)=2,0033; E(1)=0,2471 ± 0,0046; variación relativa
  −87,66 ± 0,23 %. No se observaron incrementos entre los cuatro intervalos
  muestreados; esto no demuestra monotonía continua ni se usó como gate.
- La transformación dura dejó error máximo de traza inicial de 7,6×10⁻¹⁷ y error
  de frontera cero. Es una verificación de implementación/precisión, no una prueba
  de generalización.
- Hardware: se asignaron dos workers T4 en paralelo para las semillas 101/102 y
  se reutilizó una ranura para 103. `hardware.json` confirma CUDA/T4 visibles,
  la telemetría registró utilización GPU media de 35,7 %, 36,1 % y 37,5 %; sin
  embargo, el resumen final de cada trainer etiqueta `device=cpu`. Se conserva
  esta discrepancia y no se atribuye el cálculo de entrenamiento a GPU solo a
  partir de estos campos.

## Puerta de actualización para la siguiente fase

Los resultados observados aportan el costo por paso, cobertura de bases, saturación,
sensibilidad de derivadas y dispersión entre semillas. Antes de una campaña
comparativa aún falta: resolver la discrepancia de dispositivo, converger/auditar
la referencia BDF, definir la métrica y sus umbrales antes de mirar nuevos scores,
y conservar una evaluación independiente. No copiar la densidad local de puntos
ni un umbral de residuo de Heat/Burgers a Allen–Cahn. Si las señales motivan mayor
resolución alrededor de la interfaz, proponer una variante preregistrada que
conserve este control Sobol.
