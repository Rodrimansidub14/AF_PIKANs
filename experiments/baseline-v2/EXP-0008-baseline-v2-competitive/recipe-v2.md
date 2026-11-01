# Receta v2 (`AF-R2`): Adam con warmup y decay → L-BFGS full-batch, float64, presupuesto por pasos

Estado: borrador de EXP-0008. Punto 1 del análisis. No modifica v1: EXP-0002,
EXP-0005 y EXP-0006 permanecen íntegros; v2 es una receta **nueva** con id
`AF-R2` y manifiesto propio.

## 1. Principios

1. **El presupuesto se declara en pasos** (hardware-independiente). El tiempo es
   un tope de seguridad que *censura* la corrida y se registra; una corrida
   censurada por tiempo puntúa con su último checkpoint pero no puede
   afirmar paridad de pasos (lección de Heat v4 y S2-B-spline).
2. **Nada de selección de checkpoint por error.** El checkpoint final es el
   último de la fase, no el mejor según V, E ni referencia.
3. **Cada fase deja un checkpoint puntuable**, para atribuir el efecto de
   L-BFGS y trazar curvas error-vs-coste.
4. **Una corrida no puede terminar en ERROR después de haber entrenado.**
   Visualización, W&B y empaquetado ocurren tras persistir artefactos y no son
   bloqueantes (lección DFG r4: PyVista falló *después* del entrenamiento).
5. **La receta se verifica en CPU antes de gastar GPU** (§9).

## 2. Fases

```
fase 0  init            checkpoint c0 (paso 0)
fase 1  Adam            N_A pasos, warmup + cosine, checkpoint c_adam (fin de fase 1)
fase 2  L-BFGS          full-batch sobre el pool fijo, checkpoints c_lbfgs_25/50/75/100
final   = último checkpoint de la fase 2 (o c_adam si L-BFGS aborta por no finito)
```

### 2.1 Fase 1 — Adam

| Elemento | Especificación |
|---|---|
| Optimizador | `torch.optim.Adam`, betas (0.9, 0.999), eps 1e-8, sin weight decay |
| LR | por **paso de optimizador** (no por época): callback `on_train_batch_start` fija `lr(t)`. El scheduler de época de Lightning/PINA quedaría desfasado (1 época = pool/batch pasos). Prueba obligatoria en §9 |
| Warmup | `t < W`: `lr = lr_max·(0.01 + 0.99·t/W)`, `W = 500` pasos |
| Decay | `t ≥ W`: `lr = lr_min + ½(lr_max−lr_min)(1 + cos(π (t−W)/(N_A−W)))`, `lr_min = lr_max/100` |
| `lr_max` | dev: {1e-3, 3e-3}; final: el que congele la etapa A |
| Pasos `N_A` | 20 000 (= v1) salvo el arm C4 (5 000, ver §3) |
| Batch | dev: {1024 por condición (= v1), full-batch}; final: el que congele la etapa A |
| Recorte de gradiente | ninguno (= v1). Un valor no finito aborta la corrida y crea un registro en `failures.jsonl` |
| Pesos de pérdida | todos 1, como v1 (`PhysicsInformedSingleModelSolver` promedia cada condición). El balanceo adaptativo es un módulo aparte (fuera de alcance) |

### 2.2 Fase 2 — L-BFGS full-batch

Pool completo fijo (interior + IC + BC de esa semilla), sin re-muestreo, en el
mismo dtype del modelo.

| Elemento | Especificación | Fuente/motivo |
|---|---|---|
| Optimizador | `torch.optim.LBFGS`, `lr=1.0`, `max_iter=20`, `max_eval=25`, `history_size=100` | PINNacle (`lr=1`, `max_iter=20`); historial DeepXDE |
| Línea de búsqueda | `strong_wolfe` | EXP-0007/DeepXDE; más robusto que paso fijo |
| Tolerancias | `tolerance_grad=1e-8`, `tolerance_change=0` | EXP-0007; evita parada prematura por cambio de pérdida |
| Presupuesto (pasos) | hasta **15 000 pasos externos** (`optimizer.step(closure)`), cada uno ≤ 20 iteraciones internas | paridad con PINNacle (`switch_epoch=5000` de 20 000 épocas) |
| Cotas de tiempo (censura) | dev: 600 s por corrida; final: 1 200 s (§7) | se calibra en WP-B0 |
| Paradas | (a) 15 000 pasos, (b) `tolerance_grad`, (c) tope de tiempo, (d) valor no finito → vuelve a `c_adam` y registra fallo | el motivo se guarda en `phases.jsonl` |
| Reanudación | el estado del optimizador (`state_dict`, incluidas direcciones de historial) se persiste cada 50 pasos externos con escritura atómica | permite continuar tras corte de sesión sin reiniciar historial |

**Closure.** `loss = Σ_c mean_{i∈c}(r_c(x_i))²` con los mismos residuales
normalizados y trazas que v1 (`physics`, `burgers_trace`, `allen_cahn_residual`;
Heat/AC con transformación dura). Prueba de equivalencia: sobre un pool pequeño,
la pérdida del closure difiere de la que reporta PINA en `< 1e-12` relativo.

**Comparabilidad con la continuación existente.** S3 (Allen–Cahn) y DFG L-BFGS
ya son "Adam v1 → L-BFGS" sobre checkpoints de 20 000 pasos; v2 difiere en
(i) el schedule de Adam y (ii) el presupuesto de L-BFGS. Ambos checkpoints
(`c_adam`, `c_lbfgs_100`) se puntúan siempre.

## 3. Etapa A1 — selección de receta en Burgers (semillas dev 611–613)

| Config | Adam | lr_max | Batch | Fase 2 |
|---|---|---:|---|---|
| **C1** | 20 000 const. | 1e-3 | 1024 | L-BFGS (§2.2) — *es v1 + L-BFGS* |
| **C2** | 20 000 warmup+cos | 1e-3 | 1024 | L-BFGS |
| **C3** | 20 000 warmup+cos | 3e-3 | 1024 | L-BFGS |
| **C4** | 5 000 const. | 1e-3 | full (8 192) | L-BFGS (paridad PINNacle) |

12 corridas (4 × 3), modelo RBF R0 (3 648 parámetros). Cada corrida rinde
**dos** puntos pareados: `c_adam` (solo Adam) y `c_lbfgs_100`. Lecturas
permitidas sin factorial completo:

- `C1` en `c_adam` = v1 en las semillas dev (control interno).
- `C1(c_lbfgs)` vs `C1(c_adam)` = efecto puro de L-BFGS.
- `C2` vs `C1` = efecto del schedule (mismo batch, lr_max).
- `C3` vs `C2` = efecto de `lr_max`.
- `C4` vs `C1/C2` = efecto full-batch corto (receta PINNacle).

Selección: regla de `spec.md §6`. La receta ganadora se extrapola a AC, Heat y
DFG con **verificación** (AC: 2 semillas dev en A2; Heat: gate de regresión
G1; DFG: gate G0/G1). Si en AC la receta de Burgers empeora >2× a v1+L-BFGS en
las semillas dev, no se transfiere y se abre una etapa A1' específica de AC
(requiere una nueva firma).

## 4. Parámetros por problema (valores a congelar en el manifiesto)

| Problema | Pool interior | IC / BC | dtype | Adam | Fase 2 tope pasos | Tope de tiempo de L-BFGS (dev/final) |
|---|---:|---|---|---:|---:|---|
| Heat | 4 096 | 1 024 / 512 por lado | float64 | 20 000 | 15 000 | — / 1 200 s |
| Burgers | 8 192 | 1 024 / 512 por lado | float64 | 20 000 (C4: 5 000) | 15 000 | 600 s / 1 200 s |
| Allen–Cahn | 8 192 (+16 384 held-out de diagnóstico) | trazas duras de verificación | float64 | 20 000 | 15 000 | 600 s / 1 200 s |
| DFG | 16 384 (pool filtrado, sin cilindro) | 512 por componente | **float32** (decisión ya tomada en 301–305/501–505; ver nota) | 20 000 | 15 000 | — / 900 s |

La fase 1 (Adam) se fija por **pasos**; solo lleva un tope de seguridad de
2 700 s (no debería activarse: WP-B0 excluye antes los rungs que proyecten más
de 1 800 s de Adam). Si se activa, la corrida queda `censored_time`.

Los pools son los de v1 (misma semilla → mismos puntos) para las semillas de
confirmación, verificando el hash antes de entrenar. Las semillas dev generan
pools nuevos con el mismo generador y flujos `10·seed+k`.

*Nota DFG:* la T4 tiene FP64 lento; DFG queda en float32 como en las cohortes
existentes. Antes de la etapa C, una prueba CPU compara un checkpoint en
float32 vs float64 (residual y QoI) y registra la diferencia; si `|ΔCd|/Cd > 1 %`
se abre una decisión aparte, no se cambia el dtype dentro de este experimento.

## 5. Resumen de diagnósticos

La receta no reemplaza la instrumentación existente (`Recorder` de
`train.py`); la extiende (detalle en `signals-and-records.md`). Cadencias en v2:
métricas cada 100 pasos, diagnóstico completo cada 1 000 (v1: 250) para reducir
costo, y en cada frontera de fase siempre. Los diagnósticos usan el pool
held-out de 16 384 puntos y nunca afectan al entrenamiento.

## 6. Contratos de artefacto por corrida

```
var/agent_runs/exp0008/<tramo>/<problema>/<arm>/<semilla>/
  run_config.json           configuración efectiva + hash del manifiesto
  pools.npz + pool_hashes.json
  history.jsonl             una fila por paso registrado (existente, ampliada)
  phases.jsonl              una fila por fase: pasos, segundos, motivo de parada, fevals
  lbfgs_trace.jsonl         una fila por paso externo: loss, |g|, paso de línea, fevals, |Δθ|
  checkpoints/c0.pt c_adam.pt c_lbfgs_{25,50,75,100}.pt
  dense_signals_<tag>.npz   señales sobre grilla densa dev, por checkpoint (§ signals)
  scores/{V,E,external}.json   escritos solo por el job de scoring, nunca por el de entrenamiento
  run_record.json           línea candidata para registry/runs.jsonl
  status.json               latido atómico (fase, paso, tiempo)
  stdout.log stderr.log versions.json hardware.json
```

## 7. Presupuesto de tiempo y censura

Tiempos por fase se registran por separado (Adam, L-BFGS, diagnóstico,
scoring). Estimados usados para dimensionar (extrapolaciones, **no medidos con
esta receta**):

- v1 Burgers: 20 000 pasos ≈ 1 020 s con diagnóstico cada 250 pasos.
- PINNacle Adam→L-BFGS (MLP 5×100, full-batch, T4×2): 1 949 s para 5
  semillas.
- AC S3: 240 s de L-BFGS por brazo-semilla.
- Heat S2: RBF ≤ 680 s con 20 000 Adam + L-BFGS; B-spline ≈ 2.6× más lento
  por paso.

WP-B0 (micro-benchmark, ≤ 0.25 GPU-h) mide segundos/paso de cada fase para RBF
R0/R4, MLP 58/5×100 y B-spline, y **reemplaza estos estimados** antes de fijar
topes en el manifiesto. Regla: si el estimado de un tramo supera su tope en
más de 25 % tras el benchmark, se reduce alcance (menos semillas dev o menos
rungs) *antes* de lanzar y nunca a mitad.

## 8. Lo que v2 deliberadamente NO cambia

Arquitectura R0, PDE, pools, normalizaciones, ansatz duros, pesos de pérdida,
evaluadores protegidos y semillas de confirmación. Cada uno de esos cambios es
una hipótesis separada (capacidad: `capacity-ladder.md`; pesos y causal
training: módulo D condicional; muestreo: `rad-gate-and-v2.md`).

## 9. Dry-run local en CPU (obligatorio antes de cada tramo GPU)

Entorno: `var/agent_runs/sciml-env` (torch CPU). Todo es Tier 0/1 (libre).

| Paso | Qué | Criterio de paso |
|---|---|---|
| D0 estático | `py_compile` e `import` de cada módulo del kernel; verificación de **todos** los imports declarados (`base64`, `gmsh`, `pyvista`…) | 0 errores; lista de imports = lista instalada en el manifiesto del kernel |
| D1 esquema | validar `config` del worker contra un esquema (`seeds`, `arms`, `caps`…) | ningún `KeyError`; claves desconocidas rechazadas |
| D2 unidad | schedule: valores de `lr` en t ∈ {0, W−1, W, N_A/2, N_A−1}; closure ≡ pérdida PINA (<1e-12); reanudación de L-BFGS reproduce la trayectoria de una corrida continua a `<1e-10`; censura por tiempo con reloj falso; fallo por no finito vuelve a `c_adam` | todas verdes |
| D3 humo | corrida completa de extremo a extremo por problema y familia (`rbf`, `mlp`, `bspline`) con pool 64, Adam 20, L-BFGS 3 pasos: escribe **todos** los artefactos de §6 y `run_record.json` valida contra el esquema de `registry` | artefactos completos, hashes calculados |
| D4 kernel | ejecutar el notebook/script del kernel con `SMOKE=1` (mismo código que Kaggle); validar que el `title` del kernel resuelve al `id` (error 400 de S1) y que `kernel-metadata.json` es válido | sin errores |
| D5 paridad | comparar versiones (`pip freeze`) del kernel con las del entorno local; diferencias listadas | reporte firmado en `var/` |

**Pérdidas tempranas que este dry-run habría evitado:** título/slug (D4),
`base64` ausente (D0), `gmsh` no importable (D0/D5), `KeyError: seeds` (D1),
`IPython.core.guarded_eval` de PyVista (D0 + regla 4 de §1).

Un tramo no se propone para aprobación sin el informe D0–D5 adjunto.

## 10. Desarrollo de código (régimen determinista, SDD)

Antes de implementar: spec y plan en `specs/tooling/` (tras aprobación
`code_surface`), pruebas en `tests/`, luego código. Módulos previstos:

| Módulo (nuevo) | Responsabilidad |
|---|---|
| `recipe.py` | schedule de lr por paso, definición de fases, presupuesto y censura |
| `lbfgs.py` | closure full-batch, `step` externo, persistencia/reanudación, traza |
| `preflight.py` | D0/D1/D4: imports, esquema, título/slug, versiones |
| `run_record.py` | construir y validar `run_record.json` / líneas de registry |
| cambios en `model.py` | `hidden` y `width_scale` configurables (por defecto = valores v1; regresión: pesos y salidas idénticos a v1 con configuración por defecto) |
