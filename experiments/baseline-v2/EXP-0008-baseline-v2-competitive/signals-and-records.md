# Señales internas por problema y registros (punto 6)

Estado: borrador de EXP-0008. Define qué se mide, cuándo, con qué costo y
dónde se guarda, para que el baseline competitivo deje trazabilidad completa
y para que el estudio de señal SS-1 (`rad-gate-and-v2.md`) se pueda hacer
**sin GPU adicional**.

## 0. Estado de implementación (2026-09-28)

La batería ya existe, está probada y se ejecutó sobre checkpoints reales. Es aditiva:
no modifica `model.py` ni `train.py`.

| Pieza | Ruta | Cubre |
|---|---|---|
| Batería 1D-tiempo | `src/sciml/signal_battery.py` | Heat, Burgers, Allen–Cahn; modelos RBF, B-spline y MLP (las señales de bases se omiten si el modelo no tiene bases). Rasgos densos (`u, u_x, u_t, u_xx`, residual, `‖∇residual‖`, sensibilidad, entropía/pico de bases capa 0, saturación por capa, distancia a interfaz), espectro del Jacobiano, conflicto de gradientes, ocupación, influencia por ablación de base, señales físicas por problema, estadística del residual en el pool de entrenamiento |
| Batería DFG | `src/sciml/dfg_signals.py` | residuales por región (global, cerca del cilindro, estela), no-slip, perfil de entrada, tracción de salida, **balance de caudal**, descomposición presión/viscosa de Cd y Cl, cortante de pared, longitud de recirculación, QoI σn |
| Pruebas | `tests/test_signal_battery.py` (7 pasan) | claves, finitud, ausencia de etiquetas, residual vs. diferencias finitas, restauración de parámetros tras ablación, MLP sin bases |
| Continuación + señales | `var/agent_runs/exp0008/tools/exp0008_continue.py` (Burgers), `exp0008_ac_continue.py` (Allen–Cahn) | L-BFGS desde checkpoint existente; traza por paso externo; batería antes y después; `run_record.json` con el esquema de `registry` |
| Scoring separado | `exp0008_score.py`, `exp0008_score_ac.py`, `exp0008_dfg_rescore.py` | único código que toca referencias; escribe etiquetas densas |
| Consolidación | `exp0008_collect.py`, `exp0008_ss1.py`, `build_notebook.py` | tablas por fase, SS-1, notebook |
| Notebook | `notebooks/baseline-v2/EXP-0008-baseline-v2-competitive/exp0008-baseline-v2-fast-track-signals.ipynb` | fases 0–9, rol `exploration` |

**Aún no cubierto:** señales de conflicto de gradientes y SNR por particiones *durante* el
entrenamiento Adam (solo se miden en los puntos de fase, porque el entrenamiento v1 no se
reejecuta en la ruta acelerada); batería del B-spline en Heat; brazos MLP de Allen–Cahn
(otra clase de modelo, `DeepXDEMLPAC`).

### Primeros hallazgos de la batería (orientan el mecanismo; no son causalidad)

| Señal | Observación | Lectura |
|---|---|---|
| `dfg/flux_imbalance_rel` | RBF post-L-BFGS: −10 % a −16 % (301–305 y 501–505); MLP: −1.5 % a −2 % | el RBF pierde ~11–15 % del caudal entre entrada y salida; es consistente con un Cd por debajo de la referencia |
| `dfg/residual_mass_near_cylinder` | 35 % del residual en el 2.4 % de los puntos (semilla 301) | el residual se concentra donde está el cilindro |
| `dfg/cd_pressure_part` vs `cd_viscous_part` | 3.14 / 1.91 (semilla 301, post) | la componente viscosa pesa ~38 % del Cd; es donde un error de gradiente se ve |
| `basis/layer1/dead_fraction` (Burgers) | 18 % (2 de 11 bases) en capa 1, sin cambio tras L-BFGS | capacidad desperdiciada; candidata a explicar el techo de R0 |
| `grad/conflict_cos_ic` (Burgers, semilla 201) | +0.07 en `c_adam` → −0.71 tras L-BFGS | tras converger, el gradiente de la CI se opone al de la PDE: compromiso entre términos |
| `burgers/antisym_err` (semilla 201) | 0.044 → 0.007 con L-BFGS | la simetría impar de la solución (señal sin referencia) mejora con el error de campo |
| `pool/region_residual_mass` (Burgers) | 4.7–7.9 % del residual en el 1.1 % de los puntos del frente | el frente tiene baja densidad de puntos y masa de residual moderada |

## 1. Regla de separación: rasgos vs. etiquetas

| Categoría | Definición | Quién la calcula |
|---|---|---|
| **Rasgo (feature)** | cantidad calculable **durante el entrenamiento y sin referencia** (residual, derivadas AD, activaciones, ocupación, sensibilidad, geometría) | job de entrenamiento |
| **Etiqueta (outcome)** | cantidad que necesita la referencia (error de campo, posición real del frente, QoI contra FEM/FEATFLOW) | job de scoring, **solo** después |

Etiquetas nunca entran a la pérdida, al muestreo, al early stopping ni a la
selección de checkpoint. Ausencia de medición no es cero: se registra `null`
con razón (regla 5 del proceso).

## 2. Señales existentes que se conservan (`Recorder`, `train.py`)

Pérdida por condición; norma de gradiente global y por capa; norma de
parámetros por capa; ocupación de bases (mín., media, p95, conteo <0.01), número
efectivo de bases (`exp(entropía)`), normas de coeficientes/base; estadística de
activaciones y saturación `|tanh|>0.99`; estadísticas AD de `u, u_x, u_t, u_xx`
y residual (media, RMS, p95, máx.); sensibilidad `‖du/dθ‖` en 32 puntos;
señales de muestreo (discrepancia Sobol centrada, distancia al vecino);
ejemplos vistos, segundos, memoria, GPU. Cadencias v2: métricas cada 100
pasos, diagnóstico completo cada 1 000 y en cada frontera de fase.

## 3. Señales nuevas transversales (todas las PDE)

Pendientes desde EXP-0002 (`parameter-decisions.md`: "conflicto de gradientes,
rango efectivo y norma de actualización quedan pendientes").

| Señal | Definición | Tipo | Cuándo | Costo a medir |
|---|---|---|---|---|
| `lr`, `phase` | learning rate efectivo y fase (0/1/2) | rasgo | cada paso | nulo |
| `train/update_ratio` | `‖Δθ‖/‖θ‖` por paso (Adam) o por paso externo (L-BFGS) | rasgo | 100 pasos | nulo |
| `grad/conflict_cos_{pde,ic,bc}` | coseno entre `∇L_pde` y `∇L_ic`, `∇L_bc` | rasgo | 1 000 pasos | 1–2 backward extra |
| `grad/norm_ratio_{ic,bc}` | `‖∇L_ic‖/‖∇L_pde‖`, `‖∇L_bc‖/‖∇L_pde‖` | rasgo | 1 000 pasos | ídem |
| `grad/snr_partitions` | `K=4` particiones fijas del pool por cuartiles de `t`: `SNR = ‖ḡ‖ / √(media‖g_k−ḡ‖²)` (inspirada en vRBA; **candidata**) | rasgo | 1 000 pasos | 4 backward parciales |
| `jac/effective_rank`, `jac/cond` | valores singulares de `J = ∂u/∂θ` en 256 puntos diagnóstico; rango efectivo `exp(H(σ/Σσ))` | rasgo | `c_adam`, `c_lbfgs_100` | 256 backward |
| `lbfgs/loss`, `lbfgs/gnorm2`, `lbfgs/ginf` | pérdida y normas al final de cada paso externo | rasgo | cada paso externo | nulo |
| `lbfgs/step_len`, `lbfgs/fevals`, `lbfgs/ls_fail` | longitud de paso, evaluaciones, fallos de línea de búsqueda | rasgo | cada paso externo | nulo |
| `lbfgs/curv_pairs`, `lbfgs/curv_skipped` | pares de curvatura aceptados/saltados | rasgo | 50 pasos externos | nulo |
| `cost/phase_seconds`, `cost/fevals_total`, `cost/peak_mem` | coste por fase | rasgo | por fase | nulo |
| `basis/edge_contrib` | `‖φ_g·c_g‖` por arista y base (importancia) | rasgo | fronteras de fase | despreciable |
| `basis/ablation_influence` | al anular una base de capa 0/1: `ΔRMS(residual)` y `ΔRMS(u)` en la grilla densa dev | rasgo | solo `c_lbfgs_100` | `G·capas` forward |
| `basis/xseed_stability` | τ de Kendall del rango de ocupación entre semillas; correlación entre mapas de residual entre semillas | rasgo (análisis) | tras la etapa C | CPU |

Si una señal nueva cuesta >10 % del tiempo de paso, WP-B0 la desactiva por
defecto y se registra en `proposals.jsonl`.

## 4. Señales específicas por problema

Los rasgos se calculan del modelo y del PDE; las etiquetas, con la referencia,
en el job de scoring.

### 4.1 Heat 1D (`u_t = 0.1 u_xx`)

| Señal | Definición | Tipo |
|---|---|---|
| `heat/energy(t)` | `E(t)=∫u²dx` en 9 tiempos y razón a `E_exacta(t)` | rasgo / etiqueta (razón) |
| `heat/decay_rate` | ajuste log-lineal de `E(t)` vs. `t`; comparar con `2απ²` | rasgo / etiqueta |
| `heat/uxx_scale` | RMS de `u_xx` normalizado | rasgo |
| `heat/trace_ic_bc` | desviación de IC/BC (transformación dura ⇒ ~1e-16) | rasgo |
| `heat/err_by_t` | error L2 por slices de `t` | etiqueta |

Propósito: verificar que la receta no rompe el caso suave (H4) y medir qué
bases capturan el decaimiento.

### 4.2 Burgers canónico (`ν=0.01/π`)

| Señal | Definición | Tipo |
|---|---|---|
| `burgers/front_pos(t)` | cruce por cero de `u` cerca de `x=0` para `t ≥ 1/π` | rasgo |
| `burgers/front_width(t)` | `2‖u‖∞/max|u_x|` | rasgo |
| `burgers/ux_max(t)` | `max|u_x|` por slice | rasgo |
| `burgers/antisym_err` | `‖u(x,t)+u(−x,t)‖/‖u‖` (IC impar y PDE simétrica ⇒ solución impar) | rasgo |
| `burgers/energy_balance(t)` | residuo de `dE/dt + ν∫u_x²dx` con `E=½∫u²`, por diferencias en `t` | rasgo |
| `burgers/residual_band` | RMS del residual por banda de distancia al frente y en la región crítica `|x|≤5ν, t≥1/π` (ya definida en `problem.yaml`) | rasgo |
| `burgers/front_err` | error de posición y de anchura del frente contra CH; error en frente vs. complemento (ya existe) | etiqueta |
| `burgers/err_fem101x11` | L2RE en el FEM 101×11, más piso FEM-vs-CH | etiqueta (externa) |

Propósito: la falla conocida es el frente; estas señales dicen *dónde* y
*cuándo* deja de resolverse, y si L-BFGS reduce ese error o solo el promedio.

### 4.3 Allen–Cahn (`u_t = 0.001 u_xx + 5(u−u³)`)

| Señal | Definición | Tipo |
|---|---|---|
| `ac/interface_pos(t)` | posiciones de `u=0` por slice (número y coordenadas) | rasgo |
| `ac/interface_width(t)` | `2/max|u_x|` en cada interfaz | rasgo |
| `ac/energy(t)`, `ac/dE_dt_violations` | `E=∫[D u_x²/2 + 5(u²−1)²/4]dx` en 5 tiempos, conteo de incrementos (ya existe); no es gate | rasgo |
| `ac/evenness_err` | `‖u(x,t)−u(−x,t)‖/‖u‖` (IC par ⇒ solución par) | rasgo |
| `ac/residual_interface_band` | RMS del residual en `|u_model|<0.8` (versión sin referencia de la región `interfaces`) | rasgo |
| `ac/trace_ic_bc` | 7.6e-17 / 0 con la transformación dura | rasgo |
| `ac/err_by_t`, `ac/max_snapshot_err`, `ac/interface_err` | error por tiempo, máximo por snapshot y posición de interfaz contra la referencia | etiqueta |
| `ac/err_mat201x101` | L2RE en `.mat` DeepXDE (piso 1.6e-4) | etiqueta (externa) |

Propósito: un residual pequeño no reemplaza al campo (documentado en v1); con
interfaces bien localizadas y energía coherente se ve *qué* falla cuando el
error es alto.

### 4.4 DFG Re=20 (`u,v,p`)

| Señal | Definición | Tipo |
|---|---|---|
| `dfg/continuity_rms`, `dfg/momentum_{x,y}_rms` | global, cerca del cilindro (`0<d<0.05`) y en la estela (`0.25≤x≤1.0, |y−0.2|≤0.1`) | rasgo |
| `dfg/noslip_rms`, `dfg/inlet_err`, `dfg/outlet_traction` | condiciones de frontera | rasgo |
| `dfg/flux_balance` | `∫_in u dy − ∫_out u dy` | rasgo |
| `dfg/cd_pressure`, `dfg/cd_viscous`, `dfg/cl_*` | descomposición de fuerzas en presión y viscosa (scorer σn) | rasgo |
| `dfg/wall_shear_cyl` | tensión de pared sobre el cilindro | rasgo |
| `dfg/recirc_len` | primer `x>0.25` sobre `y=0.2` donde `u` cambia de signo | rasgo |
| `dfg/qoi_err` | Cd, Cl, Δp contra FEATFLOW | etiqueta (externa) |
| `dfg/field_err_{u,v,p}` | L2 relativo por campo y región contra FEM Taylor–Hood | etiqueta (cuando la referencia exista) |
| `dfg/scorer_id` | identificador y hash del scorer usado | metadato obligatorio |

`dfg/scorer_id` evita repetir el problema observado: cifras de cohortes con
scorers distintos indistinguibles después.

## 5. Rejillas densas de rasgos (habilitan SS-1 sin volver a entrenar)

Para cada checkpoint puntuable (`c_adam`, `c_lbfgs_100`) se guarda
`dense_signals_<tag>.npz` sobre la grilla **dev** (V: 129×65; DFG: grilla dev
fija equivalente) con solo rasgos:

`x, t, u, u_x, u_t, u_xx, residual, |∇residual|, sensibilidad ‖du/dθ‖ (1 024
puntos), entropía de ocupación de bases capa 0, saturación por punto,
distancia estimada al frente/interfaz (desde el propio `u`)`.

Las etiquetas (`|error|`, error de frente) se escriben aparte en
`scores/V_labels.npz` por el job de scoring. Tamaño: ~1–2 MB por checkpoint.

## 6. Registros internos por corrida y por campaña

### 6.1 Árbol por corrida

Ver `recipe-v2.md §6`. Además, por **tramo**:

| Archivo | Contenido |
|---|---|
| `campaign_ledger.jsonl` | una línea atómica por evento (preflight, push, RUNNING, COMPLETE, descarga, verificación de hashes) |
| `quota_ledger.md` | por sesión: slug, hora de push, tope, cuota antes/después (consulta CLI), segundos reales |
| `preflight_report.json` | resultado de D0–D5 (`recipe-v2.md §9`) firmado con hash del código |
| `deviations.jsonl` | cualquier desvío del manifiesto, **escrito antes** de actuar (ver §6.3) |

### 6.2 Esquema de `registry/runs.jsonl` (append-only)

Una línea JSON por corrida; `schema: "af.registry.run/1"`.

```json
{
  "schema": "af.registry.run/1",
  "run_id": "exp0008-A1-burgers-C2-611",
  "experiment_id": "EXP-0008",
  "stage": "A1 | A2 | C | smoke | historical",
  "evidence_kind": "research | smoke | historical_backfill",
  "problem_id": "burgers_pinnacle_1d",
  "arm": {"model_id": "rbf_fixed", "rung": "R0", "hidden": 16, "bases": 11,
          "width_scale": 1.0, "params": 3648},
  "recipe_id": "AF-R2/C2",
  "seed": 611, "seed_role": "dev | confirm | historical",
  "pool_hash": "sha256", "manifest_sha256": "sha256",
  "git_commit": "…", "code_hashes": {"recipe.py": "…", "lbfgs.py": "…"},
  "env": {"python": "…", "torch": "…", "cuda": "…", "device": "Tesla T4", "slot": 0},
  "phases": [{"name": "adam", "steps": 20000, "seconds": 0.0, "stop": "step_limit"},
             {"name": "lbfgs", "outer_steps": 0, "fevals": 0, "seconds": 0.0, "stop": "…"}],
  "checkpoints": [{"tag": "c_adam", "step": 20000, "sha256": "…"}],
  "scores": {"V": {"l2_rel": null}, "E": null,
             "external": {"fem_101x11": null},
             "null_reasons": {"E": "not_opened_in_stage_A"}},
  "cost": {"gpu_seconds": 0.0, "wall_seconds": 0.0, "peak_mem_bytes": 0},
  "status": "complete | censored_time | failed | partial",
  "technical_pass": null,
  "scientific_accepted": false,
  "artifacts_root": "var/agent_runs/exp0008/…",
  "created_utc": "…"
}
```

Reglas: `technical_pass` es `null` mientras no existan umbrales congelados
(comportamiento actual de `pipeline.py gate`: `threshold_not_defined`); un
campo sin medición es `null` con razón en `null_reasons`, nunca `0`; el
esquema se valida en D3 y en cada append (línea inválida ⇒ no se agrega y se
registra fallo).

### 6.3 `proposals.jsonl` y `failures.jsonl`

`proposals.jsonl` es la memoria que impide re-proponer lo que ya falló. Una
línea por decisión relevante **antes de ejecutarla**:
`{id, date, proposer, question, change, rationale, evidence_refs[], expected,
budget, outcome: null|"positive|negative|inconclusive", outcome_refs[]}`.
El campo `outcome` se completa por *una línea nueva* que referencia el `id`
(append-only), no editando la anterior.

`failures.jsonl`: `{id, date, run_refs[], cluster, cause, cause_status:
diagnosed|open, gpu_seconds_lost, fix_ref, recurrence_guard}` donde
`recurrence_guard` apunta a la prueba D0–D5 que lo habría detenido.

### 6.4 Backfill histórico (a ejecutar tras aprobación)

`registry/` está vacío; la trazabilidad exigida por el proceso (regla 8:
claim → run → manifiesto → preregistro) no se puede reconstruir sin él. Un
script de borrador (`var/agent_runs/exp0008/tools/backfill_registry.py`, solo
lectura sobre `var/`) enumera corridas desde los `manifest.json`,
`run_summary.json` y `metrics.json` existentes y produce líneas con
`evidence_kind=historical_backfill`, `technical_pass=null`,
`scientific_accepted=false`, deduplicadas por hash de contenido. Cobertura
prevista (conteos exactos los da el script):

| Campaña | Corridas aprox. |
|---|---:|
| EXP-0002 Heat+Burgers estacionario (101–103) | 6 |
| EXP-0003 Allen–Cahn piloto (101–103) y extensión (104–105) | 5 |
| EXP-0005 pantalla de arquitecturas (3×2×3) | 18 |
| EXP-0006 Burgers Sobol vs RAD v4 (201–205, 1201–1205) y parciales v3 | 14 |
| Cierre 2026-09-27: Heat v4 (33 registros), DFG r1–r4, AC T4×2 (10) | ~55 |
| Comparadores externos: DeepXDE Heat α=0.1/0.4 (10), PINNacle Burgers Vanilla y L-BFGS (10), DeepXDE AC (5) | 25 |
| DFG L-BFGS 501–505 (10) y 301–305 (5) | 15 |
| Cierre 2026-09-28: S1a/S1b, S2 (10), S3 (10 + intento fallido) | ~25 |

Cada línea de backfill se marca `source: reconstructed` con la ruta y hash de
los artefactos de origen; un dato que no se puede reconstruir queda `null` con
razón.

**Fallos a registrar en `failures.jsonl` (clusters iniciales):**

| ID | Cluster | Causa | Estado |
|---|---|---|---|
| F-001 | RAD-v1 degrada Burgers (×5.9) | diseño de reemplazo total con ESS inestable, sin importancia, base no convergida | hipótesis, sin aislar |
| F-002 | Heat v4 `incomplete_validation` | meta de 1 M pasos inalcanzable con 500 s | diagnosticada |
| F-003 | S2 B-spline censurado por tiempo (7.8k/20k pasos) | brazo 2.6× más lento con tope 680 s | diagnosticada |
| F-004 | Kernel S1: `400 SaveKernel`, título no resuelve al id | slug/título inconsistentes | diagnosticada |
| F-005 | S1b retry: `NameError: base64` | import ausente | diagnosticada |
| F-006 | S1b retry2: `No module named 'gmsh'` | paquete Python de gmsh no instalado (posible `python-gmsh`) | hipótesis |
| F-007 | S3 intento 1: `KeyError: seeds` | config del worker incompleta | diagnosticada |
| F-008 | DFG r4: `IPython.core.guarded_eval` (PyVista) tras entrenar | visualización bloqueante | diagnosticada |
| F-009 | Scorer DFG tangencial vs σn | reducción tangencial no coincidente con FEATFLOW | diagnosticada (parche 2026-09-28) |
| F-010 | AC score-only: reloj de scoring corrió desde el inicio de la referencia | contador mal anclado | diagnosticada |
| F-011 | EXP-0006 v3: salida sin artefactos Sobol | límite de transferencia de salida | diagnosticada |
| F-012 | DFG 301–305 (Cd 74 %) vs 501–505 (6 %) | scorer o protocolo | **abierta** (H5) |

`proposals.jsonl` recibe, como mínimo: P-001 RAD-v1 (resultado negativo),
P-002 EXP-0007 (Heat step-matched, parcial), P-003 este experimento (receta
v2), P-004 gates SS-1 para RAD.

## 7. Trazabilidad del baseline competitivo (entregable)

Cada número del scorecard final debe poder seguirse en una cadena:
`scorecard → run_id(s) → manifiesto (hash) → pool_hash → checkpoint (hash) → job de scoring → evaluador (id y hash)`.
El scorecard final es una tabla `problema × arm × evaluador → media, DE, n,
semillas individuales, costo, estado`, con `scientific_accepted=false`, y se
acompaña de un notebook de análisis (Plotly, log-eje para errores positivos) y
del informe D0–D5 de cada tramo.
