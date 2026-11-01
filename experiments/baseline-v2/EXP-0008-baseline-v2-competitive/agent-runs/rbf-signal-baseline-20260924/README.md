# Baseline exploratorio de señales RBF-PIKAN frente a métodos frontera

**Corte:** 2026-09-24. **Alcance:** análisis de artefactos ya ejecutados; no inicia entrenamiento.
**Autoridad:** descriptivo. No hay aceptación científica ni claim SOTA.

## Propósito del baseline

Este baseline caracteriza señales observables en RBF-PIKAN y determina qué hipótesis de focalización o diagnóstico merecen una corrida controlada. El MLP tanh es una referencia contextual. B-spline se conserva como control de otra base explícita, necesario para probar si la geometría RBF añade valor interpretativo más allá de que el modelo tenga cualquier base local.

La motivación de RBF es estructural y medible: con centros y anchos fijos, cada activación puede enlazarse con una región del dominio y con distancias normalizadas a centros. Esto vuelve medibles ocupación, huella, redundancia, sensibilidad y estabilidad entre semillas. No demuestra superioridad de precisión ni interpretación. B-spline también tiene bases explícitas y soporte local; un MLP puede analizarse por sensibilidad aunque no exponga bases geométricas equivalentes.

## Resultados locales por semilla

Los valores de EXP-0005 son medias y desviaciones estándar entre tres semillas; las celdas y tiempos dentro de una corrida no cuentan como muestras independientes. Allen–Cahn reporta tres semillas, pero el error de campo sigue bloqueado por la referencia numérica pendiente; su residual y energía no son sustitutos del error de solución.

| Problema | Arquitectura | Señal medida | n semillas | Media | DE muestral | Rango | Unidad |
|---|---|---|---:|---:|---:|---:|---|
| heat | RBF-PIKAN fija | relative field L2 | 3 | 8.875e-05 | 4.543e-05 | 4.622e-05–0.0001366 | dimensionless |
| heat | RBF-PIKAN fija | PDE residual RMSE | 3 | 0.0009225 | 0.0001163 | 0.0007934–0.001019 | PDE residual units |
| burgers | RBF-PIKAN fija | relative field L2 | 3 | 0.05405 | 0.03934 | 0.02567–0.09895 | dimensionless |
| burgers | RBF-PIKAN fija | front-region field RMSE | 3 | 0.292 | 0.2293 | 0.1309–0.5545 | dimensionless |
| burgers | RBF-PIKAN fija | PDE residual RMSE | 3 | 0.06064 | 0.01867 | 0.04414–0.0809 | PDE residual units |
| heat | PINN MLP tanh | relative field L2 | 3 | 0.000252 | 8.872e-05 | 0.0001915–0.0003539 | dimensionless |
| heat | PINN MLP tanh | PDE residual RMSE | 3 | 0.001546 | 0.000245 | 0.001316–0.001803 | PDE residual units |
| burgers | PINN MLP tanh | relative field L2 | 3 | 0.1159 | 0.024 | 0.09271–0.1406 | dimensionless |
| burgers | PINN MLP tanh | front-region field RMSE | 3 | 0.3506 | 0.2264 | 0.1914–0.6097 | dimensionless |
| burgers | PINN MLP tanh | PDE residual RMSE | 3 | 0.1389 | 0.01263 | 0.1243–0.1467 | PDE residual units |
| heat | PIKAN B-spline cúbica | relative field L2 | 3 | 2.3e-05 | 9.487e-06 | 1.476e-05–3.337e-05 | dimensionless |
| heat | PIKAN B-spline cúbica | PDE residual RMSE | 3 | 0.0003409 | 8.908e-05 | 0.0002873–0.0004438 | PDE residual units |
| burgers | PIKAN B-spline cúbica | relative field L2 | 3 | 0.221 | 0.05098 | 0.1639–0.2618 | dimensionless |
| burgers | PIKAN B-spline cúbica | front-region field RMSE | 3 | 0.9432 | 0.1147 | 0.8163–1.04 | dimensionless |
| burgers | PIKAN B-spline cúbica | PDE residual RMSE | 3 | 0.1478 | 0.1111 | 0.08042–0.2761 | PDE residual units |
| allen_cahn | RBF-PIKAN fija | training_training_and_diagnostics_sec | 3 | 753.3 | 33.84 | 714.3–773.7 | s |
| allen_cahn | RBF-PIKAN fija | validation_heldout_normalized_residual_rmse | 3 | 0.00183 | 0.001727 | 0.0005997–0.003804 | adimensional |
| allen_cahn | RBF-PIKAN fija | validation_heldout_normalized_residual_abs_p95 | 3 | 0.003984 | 0.004066 | 0.00116–0.008645 | adimensional |
| allen_cahn | RBF-PIKAN fija | validation_energy_relative_change | 3 | -0.8766 | 0.00229 | -0.8791–-0.8745 | adimensional |
| burgers | RBF-PIKAN fija | abs residual vs absolute field error Spearman | 3 | 0.2164 | 0.09049 | 0.1179–0.2959 | dimensionless |
| burgers | RBF-PIKAN fija | top residual decile field-error enrichment | 3 | 3.428 | 1.231 | 2.595–4.842 | dimensionless |

### Lectura de las señales actuales

- En el screening pareado de tres semillas y parámetros casi iguales, B-spline logra menor error relativo L2 medio en Heat (2.30e-5 frente a 8.88e-5 de RBF), mientras RBF queda delante en Burgers (5.40e-2 frente a 1.16e-1 del MLP contextual y 2.21e-1 de B-spline). Es evidencia exploratoria por contrato, no una clasificación universal ni la razón única para elegir la arquitectura.
- En Burgers, el piloto estima Spearman entre residual absoluto y error absoluto de 0.118–0.296; el decil de mayor residual contiene 2.60–4.84 veces el error medio. Son asociaciones descriptivas con n=3 semillas, no una prueba causal ni señal confirmatoria para guiar puntos.
- En Allen–Cahn, la energía baja en las cinco muestras temporales de cada semilla y la energía relativa cambia aproximadamente −0.88; el residual held-out RMS varía entre semillas. Eso valida instrumentación física básica, pero no localiza errores de interfaz porque no hay campo de referencia habilitado.
- La pantalla de arquitecturas igualó parámetros aproximadamente (RBF 3,648; B-spline 3,648; MLP 3,655). RBF y B-spline registran bases/ocupación explícitas; MLP queda como orientación contextual. Estas métricas describen lo que el sistema puede inspeccionar, no prueban que RBF explique mejor una salida.
- Un resultado útil de interpretabilidad requerirá tres chequeos: repetibilidad entre semillas, predicción de error en validación independiente, y ablación/perturbación de las bases señaladas con cambio de campo/residual en la dirección predicha.

## Contraste con resultados SOTA/frontera publicados

Las cifras publicadas se conservan en filas separadas. No se calculan deltas entre artículos y AF_PIKANs salvo que coincidan ecuación, dominio, condiciones, parámetros, métrica, malla de evaluación, arquitectura, optimizador, puntos y presupuesto. Esta puerta evita comparar el valor de una métrica bajo contratos diferentes.

| Problema | Método publicado | Score publicado y señal local | Comparabilidad con AF_PIKANs | Señal que motiva explorar |
|---|---|---|---|---|
| Heat 1D | [Free-RBF-KAN](https://arxiv.org/abs/2601.07760) (2026) | Publicado: No score copied into the local leaderboard: the paper's high-frequency 2D heat contract differs from the project's smooth 1D heat equation. Local: EXP-0005 fixed-RBF relative field L2 = 8.88e-5 ± 4.54e-5 (n=3); the B-spline screen mean is 2.30e-5 ± 9.49e-6. | `different_problem_contract`; context_only | Band widths/grid adaptation and high-frequency modal coverage are candidate signals; the project's smooth sin(pi*x), alpha=0.1 heat run is a different PDE instance. |
| Burgers 1D | [RAMS](https://doi.org/10.1002/aidi.202500214) (2026) | Publicado: relative L2: 0.181 (random) → 0.010 (RAMS), 10 independent runs in the paper. Local: EXP-0005 fixed-RBF relative field L2 = 5.40e-2 ± 3.93e-2 (n=3). | `same_problem_family_protocol_unmatched`; context_only_no_delta | Measure whether points moved by residual gradients add information beyond residual magnitude, while recording movement overhead and independent held-out field error. |
| Burgers 1D | [vRBA](https://doi.org/10.1038/s44387-026-00084-4) (2026) | Publicado: relative L2: 8.25e-9 (vRBA+SSBroyden); prior RAD+Fourier Features+SSBroyden: 2.90e-8. Local: EXP-0005 fixed-RBF relative field L2 = 5.40e-2 ± 3.93e-2 (n=3). Same canonical equation, domain, initial condition and zero endpoint data; protocol/evaluation details are not aligned. | `same_pde_different_protocol`; same_problem_context_not_head_to_head | Residual weighting selects a functional: log/linear/quadratic/exponential potentials test different error-tail behavior. Track distribution weights and gradient signal-to-noise, not only mean residual. |
| Allen-Cahn 1D | [vRBA](https://doi.org/10.1038/s44387-026-00084-4) (2026) | Publicado: relative L2: 1.88e-6 (best listed vRBA+SSBroyden result for its Allen-Cahn contract). Local: No local field error: reference is locked pending convergence. EXP-0003 has only held-out PDE residual and energy signals. | `different_problem_contract`; not_comparable | Use energy dissipation, interface position/width, residual concentration and reference error together; residual alone cannot certify interface accuracy. |
| DFG cylinder Re=20 | [FEATFLOW DFG 2D-1 benchmark plus PINNacle/cPIKAN methods as future learned comparators](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html) (benchmark + literature) | Publicado: Official FEM reference/QoIs exist; learned-model score pending contract-matched selection. Local: No local run. | `reference_available_learned_comparator_pending`; no_local_run | Prioritize continuity, momentum terms, wall shear, pressure and drag/lift errors; validate against converged FEM QoIs before comparing architectures. |

**Lectura numérica del SOTA:** el artículo vRBA publica 8.25e-9 de error relativo L2 para Burgers canónico con vRBA+SSBroyden; AF_PIKANs midió 5.40e-2 ± 3.93e-2 con RBF fija. Ambas cifras usan la misma familia de PDE y métrica, pero difieren en optimizador, features, muestreo, presupuesto y protocolo de evaluación; el cociente entre ellas no es un efecto experimental ni un head-to-head. RAMS reporta 0.181→0.010 con movimiento de muestras en diez repeticiones, igualmente como referencia de método, no como delta frente al RBF local. Para Allen–Cahn, el artículo publica 1.88e-6, pero usa otra difusividad/condición de frontera y el piloto no habilita error de campo. Heat tiene solución analítica local; DFG carece aún de corrida local.

## Señales que pasan a la siguiente exploración

| PDE | Fenómeno dominante | Señales prioritarias | Decisión habilitante |
|---|---|---|---|
| heat | `heat_1d_alpha_0.1_sin_pi` | modal_decay, thermal_energy_decay, u_xx_scale, initial_boundary_trace | Descomponer u(x,t) en modos seno y correlacionar la cola modal con activación RBF en validación fija. |
| burgers | `burgers_1d_raissi_pinnacle_transient` | front_position_and_width, shock_gradient, viscous_dissipation, initial_boundary_trace | Mantener la evaluación independiente y medir residual/error por bandas del frente, además de correlación y enriquecimiento por semilla. |
| allen_cahn | `allen_cahn_deepxde_1_12_2_dirichlet` | interface_and_energy_dissipation, interface_position_and_width, double_well_balance, initial_boundary_trace | Converger y desbloquear la referencia de campo; medir error de interfaz y energía frente a Sobol en puntos held-out. |
| dfg_ns | `dfg_2d1_re20_incompressible` | divergence_and_momentum_balance, cylinder_wall_shear, drag_lift_pressure_drop, wake_length | Primero fijar solver FEM convergido, malla, normalización, QoIs DFG y descomposición de residuo por ecuación. |
| poisson | `scope_pending` | flux_balance, corner_singularity_or_smoothness, boundary_trace | Conservar como problema candidato; reabrir solo tras decidir el alcance de la tesis. |

## Recomendación de la siguiente campaña

1. Cerrar el baseline RBF fijo y su scorecard descriptiva con los pools Sobol ya documentados; preservar RBF como arquitectura primaria para observar señales, MLP como referencia y B-spline como control de base local.
2. En Burgers, preregistrar una sola intervención contra Sobol. Priorizar una reproducción de RAD/RAR-D o RAMS/vRBA bajo presupuesto definido y conservar arquitectura RBF; medir cobertura de bases, residuales/error por banda del frente, gradiente/SNR por partición, error held-out, overhead y distribución de puntos.
3. En Allen–Cahn, completar convergencia y revisión de la referencia de campo; medir error y anchura/posición de interfaz junto a energía y residual. Añadir SNR/estados de entrenamiento porque la literatura informa fases de fitting, transición y difusión que el resumen agregado actual no mide.
4. Pasar a DFG/NS después de fijar FEM, malla y QoIs. No usar el score de calor o Burgers para transferir umbrales a fluidos.
5. Confirmar cualquier señal elegida con al menos cinco semillas y un presupuesto igualado antes de afirmar generalidad o ventaja frente a resultados publicados.

## Medición y linaje

Los artefactos tabulares se generan con `python -m src.research_pipeline.sciml.rbf_signal_baseline`. `observed-signals.csv` conserva métricas medidas por run; `literature-crosswalk.json` conserva contexto y estatus de comparabilidad; `exploratory-signals.html` muestra solo resultados locales, sin superponer puntuaciones incompatibles de papers.

Entradas verificadas por SHA-256:

- `screen_primary`: `783a60f142adaa4e0f9619ce8ebebdebe1c7c9b838ab6cc2a475ce30135e2ce5`
- `screen_signals`: `a0355820ac9219273de6e4c40529002ff7b28780b92d50d5107c84d40d5c35ea`
- `burgers_residual_error`: `f3ece994a40050e6b25c868f85db0e47974b068817dc9e56db92c366b98f1b0b`
- `allen_cahn`: `e13aa82d2aa54a1601e5c3605360f8f79f185c8aa3ebf49334a4b7ad09c52aa2`

No se leyó `evals/`, no se cambió ningún grader o umbral y `scientific_accepted=false`.
