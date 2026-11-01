# P4: solución Navier--Stokes DFG 2D-1

`solve_dfg.py` resuelve el contrato DFG 2D-1 estacionario con elementos
Taylor--Hood `P2/P1`, malla triangular Delaunay con cilindro poligonal explícito,
continuación de Picard y correcciones de Newton. La salida usa exactamente
`nu=0.001`, entrada parabólica de máximo `0.3` y media `0.2`, no deslizamiento
en paredes y cilindro, y salida natural `nu d_n u - p n = 0` sin presión fijada.

## Reproducir

Desde la raíz del repositorio, con el Python incluido en la evidencia:

```powershell
$py = 'C:\Users\rodri\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'var\agent_runs\teoria-visual-v2\vendor'
& $py 'var\agent_runs\teoria-visual-v2\dfg\solve_dfg.py' 2>&1 |
  Tee-Object 'var\agent_runs\teoria-visual-v2\dfg\run-full.log'
```

La ejecución completa usa tres niveles (`coarse`, `medium`, `fine`). `--quick`
usa los dos primeros para una comprobación inicial. La prueba interna de
interpolación escalar `p(x,y)=x+2y` se guarda en `evidence.json`.

## Archivos

- `p4-solution.pdf`, `.png`, `.svg`: velocidad, presión y líneas de corriente
  evaluadas desde el campo FEM resuelto; el sólido está enmascarado.
- `p4-illustration.pdf`, `.png`, `.svg`: geometría y contrato físico, sin campo
  simulado.
- `p4-solution.npz`: malla, grados de libertad y grillas de visualización.
- `evidence.json`: residuales, flujos, normales, funcionales y comparación de
  refinamiento.
- `method.tex` y `caption.tex`: fragmentos para la integración en la tesis.

La evidencia queda marcada `not_certified`: los balances y el residuo débil
son pequeños, pero el funcional de sustentación cambia de signo y
`p_diff` aún no satisface el umbral de convergencia declarado.
