# EXP-0004 — DFG 2D-1 Re=20 con RBF-PIKAN

**Estado:** borrador de diseño; no firmado ni ejecutado.  
**Rol:** exploración técnica, con validación de campo y QoI cerrada hasta
implementar referencias independientes.

Este protocolo es provisional y depende de EXP-0002/0003. Antes de firmar,
ajustar el presupuesto al throughput medido, la precisión numérica y el uso
real de GPU; adaptar regiones de diagnóstico solo con evidencia del problema.
Los controles uniformes y cada política de focalización tendrán manifiestos
separados.

## Pregunta

¿Puede el trainer RBF-PIKAN modelar el flujo estacionario alrededor del
cylinder DFG 2D-1 y qué señales del residuo, frontera y optimización revelan
sus zonas difíciles?

## Problema y condiciones

Se mantiene la geometría del contrato existente:

\[
\Omega=[0,2.2]\times[0,0.41]\setminus B_{0.05}(0.2,0.2),
\]

con `ν=0.001`, `ρ=1` y Navier–Stokes estacionario en forma laplaciana para
`(u,v,p)`. Perfil de entrada parabólico `u=4·0.3·y(0.41−y)/0.41²`, `v=0`;
no-slip en paredes y cilindro; salida natural de la forma declarada en
`evals/problems/dfg_2d/reference.py`. Mantener la convención de presión
cinemática y documentar cualquier cambio de outlet antes de entrenar.

## Protocolo inicial propuesto

- Red de tres entradas `(x,y)` físicos más salida `(u,v,p)`; geometría RBF fija.
- 16.384 puntos interiores por Sobol aleatorizado sobre el rectángulo, con
  rechazo reproducible del disco; guardar candidatos aceptados y pool exacto.
- Muestrear por separado cada frontera; círculo por ángulo equiespaciado sin
  duplicar extremo, segmentos por Sobol 1D. No usar el evaluador como pool.
- Tres semillas; Adam `lr=1e-3`; batch inicial 2.048; **20.000 pasos o 20 min
  por semilla** como tope, sujeto a un smoke de memoria/costo.
- Evaluación independiente por pools Sobol: momento-x, momento-y, continuidad;
  condiciones de entrada, no-slip y salida; errores por región near-cylinder
  y wake. Reportar coste y hardware.
- No reportar RMSE contra campo FEM ni error de drag/lift/Δp aún: la referencia
  numérica local y las integrales de fuerza deben auditarse primero.

La primera comparación posterior será muestreo fijo contra una variante RAD o
RAR-D, con los mismos pools de evaluación, red, semillas y presupuesto. La
estratificación near-cylinder/wake se evaluará como factor separado, no se
introduce en el control base.

## Falsación y cierre

Invalida la corrida cualquier NaN, falta de conservación, incumplimiento de
condiciones de frontera, inconsistencia del orden `(x,y)` o presión, o
referencias de hardware incompletas. Residuos pequeños no validan QoI ni
superioridad. Cierre con dispersión entre semillas, campos y streamline plots,
lista de fallos y revisión humana.

## Fuentes

- [FEATFLOW DFG benchmark 2D-1 Re=20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html): geometría, ecuaciones y cantidades canónicas drag/lift/Δp.
- [Wu et al., sampling study](https://arxiv.org/abs/2207.10289): muestreo fijo versus RAD/RAR-D; motiva una comparación controlada posterior.
- [Turek et al., benchmark computations](https://www.mathematik.tu-dortmund.de/papers/TurekMierkaHysingKuzmin2010.pdf): refinamiento de malla y referencias CFD.
