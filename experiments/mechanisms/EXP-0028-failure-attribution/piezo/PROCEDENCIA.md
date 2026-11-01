# Procedencia y licencia del código portado (piezo, EXP-0028)

**Origen.** Paper *Enhanced Multiphysics Simulation via a Tailored PINN Architecture for Piezoelectric Cantilever Beam Energy Harvesters*, Math. Comput. Appl. 2026, 31, 214 (Gonzalez Carrillo et al.), y su repositorio `Daniel14gonc/PINNs_piezoelectricity`, commit `e7856f1` («CLEAN: reduce repository to the paper's 20k campaign», 2026-10-02). Clon de solo lectura en `var/agent_runs/piezo-ref-20260930/daniel-ref-20261003/` (no se modifica; contiene además los pesos entrenados y las historias de pérdida del paper en `results/runs/paper20k_*`).

**Licencia.** `pyproject.toml` del repo declara **MIT**, pero **no hay archivo `LICENSE`** (ni aviso de copyright) en el árbol. Estado: *MIT declarada sin LICENSE*. Hasta tener el archivo o la confirmación de los autores, este registro la trata como «reutilización con atribución, pendiente de confirmar»: 
- el puerto (`piezo_problem.py`) es una obra derivada línea por línea de `src/pinn_piezo/{indirect,direct}/{model,losses,sampling}.py`, `scaling.py`, `materials.py` y `geometry.py`; conserva su estructura, nombres de términos y orden de operaciones (la prueba de la pérdida contra su código lo exige);
- la atribución está en la cabecera de `piezo_problem.py` y aquí; **no se redistribuye código del repo** (el FEM, `fem.py`, solo se importa para regenerar `reference/fem_*.npz` con `make_fem_reference.py`; las pruebas importan el repo para comparar y se omiten si falta);
- pendiente humano: pedir a los autores el archivo `LICENSE` (o una nota de permiso) antes de publicar `piezo_problem.py` fuera del repositorio privado o dentro del documento de tesis, y antes de subir el bundle (`af-pikans-exp0028-bundle-piezo`) si el dataset dejara de ser privado. El dataset de Kaggle es privado.

**Qué se tomó y de dónde.**
| Elemento | Fuente | Dónde vive aquí |
|---|---|---|
| Material, escalas características, normalización constitutiva represented/dominant | `materials.py`, `scaling.py` (paper ecs. 44-62) | `piezo_problem.py` (`_material`, `converse_scales`, `direct_scales`, `dominant_scales`) |
| Modelo: entradas (ξ, η, \|η−1/2\|, p), evaluación N±, construcción por fase, bordes duros | `indirect/model.py`, `direct/model.py` (ecs. 30-35, 72-80) | `PiezoNet` |
| Pérdida converso (enrutamiento) y directo (λ_BC = 10, carga neta cero) | `indirect/losses.py`, `direct/losses.py` (ecs. 36-71) | `PiezoProblem.loss` y `piezo_residual_terms` |
| Muestreo por capa y remuestreo cada 200 | `indirect/sampling.py` | `sample_pools`, `run_piezo.adam_piezo` |
| FEM de referencia (P2, 200×8, 20 451 DOF) | `fem.py` (scikit-fem 12.0.2, BSD-3) | `reference/fem_converse.npz`, `fem_direct.npz` vía `make_fem_reference.py` |
| Tronco RBF-PIKAN | propio (`pikan_campaign.RBFEdge`; el intercambio en `model.net` es el de `EXP-0009/daniel_arm.py`) | `RBFEdge`, `PiezoNet(trunk='rbf')` |

El paper declara que la construcción por fase, las escalas y el código de entrenamiento se desarrollaron con asistencia de modelos de lenguaje (ChatGPT, Codex); no cambia la licencia pero conviene citarlo.

**Referencia FEM: no es de terceros.** El FEM es del mismo grupo que la PINN (el paper lo valida con mallas P3 y fórmulas de viga: 198.5 µm vs 200 µm Euler-Bernoulli, 32.9 vs 33.0 µm bimorfo, 3.10 vs 3.11 V circuito abierto). Pendiente el contraste con COMSOL o solución analítica (no bloquea el uso interno).
