# experiments/ — pre-registros

Un experimento es una campaña contra la suite de evals que prueba **una**
hipótesis. La unidad de trabajo que un agente toma es una carpeta completa.

```
EXP-NNNN-slug/
  spec.md        pre-registro. Se escribe ANTES de correr. Lo firma el humano.
  manifest.yaml  contrato máquina que ejecuta el harness
  decision.md    EDR: veredicto humano tras ver el scorecard
  runs/          gitignored; la evidencia vive en var/ y se indexa en registry/
```

## La regla eval-first

No se abre un experimento sin declarar antes, en `spec.md`:

1. qué suite lo evalúa,
2. qué métrica tiene que moverse,
3. cuánto tiene que moverse para contar,
4. qué resultado lo falsaría.

Sin esas cuatro, no hay experimento: hay una corrida exploratoria, y esas van
a `notebooks/` con rol `exploration`, sin derecho a producir evidencia.

## Por qué el pre-registro va antes

Si el umbral se fija después de ver los resultados, el umbral no mide nada:
mide la corrida que ya ocurrió. Es la misma razón por la que `evals/` está
protegido contra escritura de agentes.

## Estados

`draft` -> `preregistered` (firmado) -> `running` -> `scored` -> `decided`

Solo el humano mueve `scored` -> `decided`.

## Próxima preparación

EXP-0002 queda como evidencia histórica: Heat y Burgers estacionario con
RBF-PIKAN fija. EXP-0005 prepara el screening Heat + Burgers canónico con
RBF-PIKAN, MLP PINN y PIKAN B-spline; vuelve a entrenar Burgers bajo la
definición transitoria de Raissi/PINNacle, sin mezclarla con EXP-0002. El estado
de ejecución se mantiene en
[`var/agent_runs/heat-burgers-architecture-screen`](../var/agent_runs/heat-burgers-architecture-screen/).
Los borradores [EXP-0003 Allen–Cahn](EXP-0003-allen-cahn-rbf-sobol/spec.md) y
[EXP-0004 DFG Re=20](EXP-0004-dfg-re20-rbf-sobol/spec.md) se adaptan tras
revisar señales y costo de EXP-0005. Poisson sigue pendiente mientras se
formaliza cualquier cambio del conjunto objetivo en el
[roadmap](roadmap-next-problems.md).
