# Qué falta para pasar a NS/DFG y piezo (2026-10-20)

Solo inventario: no se lanzó nada. Es el insumo de la regla de salida firmada el 2026-10-20 22:01 (si la ronda 1-B no pasa T1, la tesis se plantea con el diagnóstico de condicionamiento, la cadena negativa 0015-0018 y la dependencia del problema, "validado en un problema real (NS/DFG, piezo)"). Lo marcado **[verificado]** lo vi en archivos; **[inferido]** es lectura mía y hay que confirmarlo.

## Lo que ya existe

| | NS2d-CG (PINNacle) | DFG 2D-1, Re = 20 | Piezo (bimorfo PVDF) |
|---|---|---|---|
| Problema en código | `NSProblem` en `pikan_campaign.py`, 3 salidas (u, v, p), pérdida con 8 términos (momento x/y, continuidad, 5 de borde) **[verificado]** | `src/sciml/dfg.py` (`DFGRBFPIKAN` 2-16-16-3, `steady_ns_residual`, QoI Cd/Cl/Δp), campaña en `dfg_campaign.py` **[verificado]** | **No hay `PiezoProblem` ni modelo propio en `src/` ni en la campaña.** Solo el repo de Daniel González clonado en solo lectura (`var/agent_runs/piezo-ref-20260930/daniel-ref*`) **[verificado]** |
| Referencia | `ns2d-cg-20260930/reference` (tabla x y u v p, la usa `NSProblem.score`) **[verificado]** | FEM Taylor–Hood L3 propio (DOLFINx 0.11, `reference/dfg_fem/solve_dfg.py`), anclado a FEATFLOW (Cd 2.9e-5, Δp 1.2e-4 relativo); aceptada como referencia **de trabajo**, no para claims **[verificado]** | FEM propio de Daniel (scikit-fem, P2, 0.4–3 s por solución); pasa dos chequeos de mano; **sin contraste de terceros** (COMSOL o analítica del bimorfo) **[verificado]** |
| Línea base | cohortes 301–305 / 501–505 (7–9 % en u, v; 5–8 % en Cd); receta larga (`dfg-long`, 20 k Adam + L-BFGS con tope de 600 s): RBF16 0.191 en u, Cd 35 % de error, **L-BFGS terminó siempre por tope de tiempo** **[verificado]** | ídem | congelada: 0.092 / 0.136 (MLP y RBF16 con el código de Daniel); **no existe etapa L-BFGS** para piezo ni para NS2d-CG **[verificado, EXP-0013]** |
| Capas RBF compatibles con el mecanismo | `RBFEdge` igual que en Burgers/AC | `RBFEdgeLayer` con la misma forma (`coefficients` (O,I,G), `centers`, `width`, `base`) **[verificado]** | MLP tanh de Daniel; el RBF-PIKAN de piezo existe solo como cohorte de EXP-0013 **[inferido: falta ubicar el código que lo entrenó]** |

## Lo que falta, por pieza

### A. Para cualquier problema nuevo: el mecanismo hoy solo entiende Burgers y Allen–Cahn
1. **Residuo vectorial.** `gram_preconditioning.make_residual` (y `gn.py` del 0022) devuelven un residuo escalar y solo para `burgers_pinnacle_1d` y `allen_cahn`. NS/DFG tienen 3 ecuaciones y términos de borde en la pérdida; el GN del 0023 (J^T J / N de la pérdida real) necesita **todos** los términos que entran en `prob.loss`, con sus pesos. Hay que definir una interfaz `prob.residual_terms(model_fn, X)` por problema.
2. **Forward independiente del modelo.** `_forward` replica `RBFKAN`; DFG multiplica por `output_scales`, piezo tiene 8 salidas y bordes duros. Conviene calcular J con `torch.func.functional_call` sobre el modelo real (con P temporalmente en identidad) en vez de reescribir el forward.
3. **Qué puntos.** El 0022/0023 usa 2 048 puntos del pool de colocación; en NS/DFG el pool tiene dominio, entrada, salida, cilindro y paredes. Decidir (y preregistrar) cuántos puntos de cada tipo entran en B.
4. **Entrenador.** `pre_trainer.py` y `t1_trainer.py` solo tienen `burgers` y `ac`. Falta un entrenador por problema que replique la receta de su campaña (DFG: float32, Adam 20 k + L-BFGS con tope de segundos; hoy el L-BFGS de DFG se corta por reloj, no por iteraciones, lo que rompe la comparabilidad "mismo presupuesto" de T1).
5. **Presupuesto.** Medido solo para Burgers/AC. El Jacobiano de un residuo con 3 ecuaciones + 5 de borde es varias veces más caro; medir antes.

### B. NS/DFG en particular
- **Elegir uno.** NS2d-CG ya tiene `NSProblem` (más barato, referencia tabulada) pero DFG es el problema "real" que nombra la regla de salida y tiene QoI (Cd, Cl, Δp). Decisión del investigador.
- **L-BFGS por iteraciones.** Hace falta una variante con tope de iteraciones (como T1) o declarar que el reloj es el presupuesto.
- **Referencia de DFG:** ya está aceptada solo como "de trabajo". Para un claim hace falta el gate `scientific_claim` (pendiente) y mantener la salvedad: el campo lo generó un agente; su respaldo de terceros son los QoI de FEATFLOW. Cl solo se valida con el cálculo nativo de DOLFINx.
- **Presión:** sin gauge fijo; comparar con gauge en (2.2, 0.2).

### C. Piezo en particular (lo más lejos)
1. **Problema sin implementar en nuestro código:** falta `PiezoProblem` (8 salidas en formulación mixta: u, v, φ, σx, σz, τxz, Dx, Dy; bordes duros; pesos adaptativos entre PDE y borde en el código de Daniel). Decidir si se adopta su parametrización de salidas o se compara con su "caso A estándar".
2. **Licencia y permiso:** el repo declara MIT en `pyproject.toml` pero no tiene archivo LICENSE; confirmar con Daniel antes de reutilizar código.
3. **Referencia de terceros:** el FEM es del mismo grupo que la PINN. Para sostener el criterio "de terceros" falta contrastarlo con COMSOL o la solución analítica de bimorfo (se puede hacer sin GPU).
4. **Convenciones sin fijar:** signo de polarización (el default de su FEM da el sentido opuesto al de sus pesos) y signo de e31/e33 (tu tabla `tab:pvdf` los lista con el mismo signo; el repo, opuesto). Sin esto el error de referencia no es interpretable.
5. **Formulación directa no usable:** los pesos del paper dan amplitud ≈ 0.11× y φ con correlación 0.50 frente al FEM; solo la indirecta es comparador utilizable (u, v ≈ 0.19 L2).
6. **Etapa L-BFGS inexistente** para piezo; el precondicionamiento actúa en las dos fases.
7. **Float64** en el código de Daniel; el resto de la campaña corre en float32. Decidir dtype.

## Orden sugerido (para decidir, no ejecutado)
1. Generalizar `gram_preconditioning` al residuo vectorial + `functional_call` y probarlo en NS2d-CG con CPU (la interfaz es lo barato).
2. DFG antes que piezo: el problema, la referencia y la línea base ya existen.
3. Piezo solo después de cerrar convención de signos, licencia y contraste de terceros; sin eso la comparación no tiene vara.

Nada de esto se lanza sin autorización explícita (GPU/Kaggle) ni toca `evals/` ni `registry/claims.jsonl`.
