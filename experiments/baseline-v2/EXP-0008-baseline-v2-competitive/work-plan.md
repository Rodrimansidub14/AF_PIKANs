# Plan de trabajo, tramos de GPU y presupuesto

> **Nota (2026-09-28):** este es el diseño completo de respaldo (~17.5 GPU-h).
> La ruta recomendada es [`fast-track.md`](fast-track.md) (~1.6–6 GPU-h), que
> parte de una medición en CPU y reutiliza los checkpoints v1 existentes.

Estado: borrador de EXP-0008. Nada de esto se ejecuta sin la firma del
preregistro y la aprobación **por tramo**.

## 1. Meta y entregable

**Meta:** un baseline RBF-PIKAN competitivo, reproducible y trazable, evaluado
con evaluadores declarados, contra recetas reproducidas de la literatura, en
Heat, Burgers, Allen–Cahn y DFG Re=20 (Poisson sigue pendiente de definición de
alcance; no se toca aquí).

**Entregable "baseline competitivo v2"** (un solo paquete en
`var/agent_runs/exp0008/deliverable/`, más el resumen en `experiments/`):

1. `scorecard.csv`: `problema × arm × evaluador → media, DE, n, semillas
   individuales, costo, estado`, con veredicto por problema
   (`comparable_to_literature | screening_only | not_comparable`).
2. Curvas error-vs-pasos y error-vs-segundos por fase (`c0`, `c_adam`,
   `c_lbfgs_25/50/75/100`).
3. Tablas de señales por problema (`signals-and-records.md §4`) y
   `dense_signals_*.npz` para SS-1.
4. Informe de evaluadores: piso del FEM 101×11, tabla de scorer σn de DFG,
   estado de las referencias con fichas de aceptación.
5. `registry/runs.jsonl` (incluido el backfill) y `failures.jsonl`, más
   `proposals.jsonl` si se aprueba el rol.
6. Informes D0–D5 y `quota_ledger.md` por tramo.
7. Notebook de análisis reproducible (Plotly, eje log para errores) y un
   memo de decisión para el humano (`decision.md` lo escribe el humano).

Fuera de alcance (declarado): reproducción de vRBA (Fourier features, pesos
adaptativos, SSBroyden 60 000), reproducción de AEH-PINN, RAD/RAR-D,
bases aprendibles, balanceo de pérdidas y causal-training, operator learning,
Poisson.

## 2. Paquetes de trabajo

Fase L = local CPU (Tier 0/1, libre tras las aprobaciones indicadas).
Fase G = Kaggle T4×2 (Tier 2, presupuesto por tramo).

| WP | Fase | Contenido | Depende de | Aprobación | Esfuerzo (estimado) |
|---|---|---|---|---|---|
| **WP-R0** | L | Backfill de `registry/` y clusters F-001…F-012 (`signals-and-records.md §6`) | firma de esquema | #5 y rol de escritura | 0.5–1 día |
| **WP-E1** | L | Burgers: puntuar checkpoints v1/RAD en FEM 101×11 y medir el piso | descarga externa anunciada | lectura externa | 0.5 día |
| **WP-E3** | L | DFG: scorer σn único sobre 301–305 y 501–505; prueba analítica y de orientación | — | — | 0.5–1 día |
| **WP-E4** | L | DFG: referencia FEM Taylor–Hood, 3 niveles, exportación 440×82 | decisión de ruta (A/B/C) | #6 | 3–5 días |
| **WP-C0** | L | SDD: spec + plan + tests + `recipe.py`, `lbfgs.py`, `preflight.py`, `run_record.py`, cambios en `model.py` | `code_surface` | #2 | 3–4 días |
| **WP-C1** | L | Dry-run D0–D5 con informe firmado | WP-C0 | — | 1 día |
| **WP-B0** | G | Micro-benchmark de tiempos (≤ 0.25 GPU-h) para fijar topes | WP-C1 | tramo B0 | 0.25 h |
| **T1** | G | Etapa A1: 12 corridas Burgers dev (C1–C4 × 611–613) | WP-B0, cuota verificada | tramo T1 | ver §3 |
| **B-freeze** | — | Congelación de receta (humano; `manifest` v2) | T1 | `eval_integrity` (umbrales) | — |
| **T2** | G | Etapa A2: escalera Burgers + AC (22 corridas dev) | B-freeze | tramo T2 | ver §3 |
| **R-freeze** | — | Congelación de rung(s) `R*` (humano) | T2 | — | — |
| **T3a** | G | Etapa C Burgers (20 corridas) + scoring E/externo | R-freeze, WP-E1 | tramo T3a | ver §3 |
| **T3b** | G | Etapa C Allen–Cahn (15 corridas [+5]) + scoring | R-freeze | tramo T3b | ver §3 |
| **T3c** | G | Etapa C Heat, RBF (5 corridas) [+ opcional] | R-freeze | tramo T3c | ver §3 |
| **T4** | G | Etapa C DFG (10 corridas), condicional a WP-E3 | WP-E3, R-freeze | tramo T4 | ver §3 |
| **SS-1** | L | Estudio de señal sobre `dense_signals` | T3a, T3b | — | 1–2 días |
| **WP-D** | L | Scorecard, notebook, memo | T3*, SS-1 | — | 1–2 días |

Los WP de fase L pueden empezar en paralelo apenas se aprueban; WP-E1, WP-E3 y
WP-R0 **no dependen de la receta** y pueden hacerse ya, con costo cero de GPU.

## 3. Tramos de GPU

Estimados a partir de datos medidos (v1 Burgers ≈ 1 020 s/20k pasos; PINNacle
Adam→L-BFGS ≈ 1 949 s para 5 semillas en T4×2; S3 240 s de L-BFGS; DFG ≈ 1 200 s
de entrenamiento + 540 s de L-BFGS) y extrapolados a rungs más grandes. Se
recalibran en WP-B0. Sesión = tiempo de reloj del kernel (Kaggle cuenta T4×2 por
sesión, no por dispositivo). Dos corridas concurrentes (una por T4).

| Tramo | Corridas | Tiempo por corrida (est.) | Sesión estimada | Tope duro | Notas |
|---|---:|---|---:|---:|---|
| B0 micro-benchmark | ~6 cortas | 100–200 s | 0.25 h | 0.30 h | fija topes y decide recortes |
| **T1** A1 Burgers dev | 12 | C1–C3 ≈ 1 400 s; C4 ≈ 900 s | ~2.3 h | 3.0 h | incluye 10 % de reserva |
| **T2** A2 capacidad dev | 22 | 900–2 700 s según rung | ~4.5 h (3.7–6.4) | 6.5 h | orden de exclusión previa si B0 proyecta > tope: AC-R4, AC-R5, AC-R2, Burgers-R4 |
| **T3a** C Burgers | 20 (4 arms × 5) | RBF-R0 1 400; RBF-R* 2 000; MLP-58 900; MLP-5×100 700 | ~3.9 h | 5.0 h | + scoring E/externo |
| **T3b** C Allen–Cahn | 15 (3 arms × 5) [+5 MLP-match] | RBF-R0 1 400; RBF-R* 2 000; MLP-58 900 | ~3.0 h [+0.7 h] | 4.0 h | referencia candidata; ver §5 del spec |
| **T3c** C Heat | 5 RBF (+5 B-spline +5 FNN opcionales) | RBF ≈ 700 s | 0.5 h [+2.1 h] | 1.0 h [3.3 h] | B-spline ≈ 2 400 s, FNN ≈ 600 s |
| **T4** C DFG | 10 (RBF y MLP × 5) | ≈ 2 100 s (1 200 Adam + 900 L-BFGS) | ~3.0 h | 4.0 h | condicional a WP-E3 |
| **Núcleo** | **≈ 90** | | **≈ 17.5 h** | **≈ 23.8 h** | sin opcionales |

Cada tramo respeta el límite operativo previo de **menos de 8 h por campaña**.
El núcleo no cabe en una sola sesión de cuota; se recomienda repartirlo en
al menos dos semanas de cuota si esta se recorta, o en una si se libera el
tope semanal de 30 h.

**Cuota.** Último valor leído por CLI: 15.75 h disponibles antes de S2 (2026-09-28);
consumo posterior conocido S2+S3 = 1.326 h; estimación 14.42 h **no
verificada**; refresco anunciado 2026-10-03. Antes de cada lanzamiento se
consulta con `kaggle quota` (el comando `kernels quota` no existe en CLI
2.2.4). Reserva mínima propuesta: **5 h** de saldo tras cada tramo
(el investigador puede fijar otra; la reserva del plan anterior era 12 h).

## 4. Reglas de ejecución por tramo (go / no-go)

**Antes de lanzar:**

1. Informe D0–D5 firmado y hash del código del kernel.
2. Cuota leída por CLI en ese momento; `saldo − tope ≥ reserva`.
3. Bundle con `manifest.json`, hashes de pools verificados y `kernel-metadata`
   validado localmente (título ↔ slug).
4. Cap de sesión en el notebook (watchdog con reserva de 5 min para escribir
   artefactos) y escritura atómica de progreso y checkpoints.

**Durante:** latidos en `status.json`; cada corrida escribe `run_record.json`
aunque falle. Una corrida no puede quedar en ERROR tras haber entrenado
(visualización y W&B van al final y son no bloqueantes).

**Detenerse (stop rules):**

- Las dos primeras corridas proyectan > 125 % del tope del tramo → abortar y
  reducir alcance (según orden declarado) antes de continuar.
- ≥ 2 corridas con valores no finitos en la misma configuración → detener,
  registrar en `failures.jsonl`, no reintentar automáticamente.
- Fallo de infraestructura → registrar, corregir en local y **repetir D0–D5**
  antes de gastar la reserva de infraestructura del tramo (0.3 h, incluida
  en el tope).

**Después:** descarga y verificación de hashes antes de puntuar; scoring en
job aparte; entrada en `runs.jsonl`; `quota_ledger.md`; memo corto de cierre
del tramo; **no se lanza el siguiente tramo sin ese memo**.

## 5. Decisiones humanas en el camino

| Punto | Decide | Con qué información |
|---|---|---|
| Firma del preregistro | investigador | este paquete |
| `code_surface` para `model.py`/`train.py` | investigador | WP-C0 spec y diff previsto |
| Freeze de baselines y umbrales | investigador (`eval_integrity`) | §5.1 y §5.2 del spec |
| Ruta FEM (A/B/C) | investigador | `evaluators-and-references.md §4` |
| B-freeze (receta) | investigador | resultado de T1 aplicando la regla de selección |
| R-freeze (rung) | investigador | resultado de T2 |
| Aceptar referencia AC / DFG | investigador | fichas de §5 de `evaluators-and-references.md` |
| Claim | investigador (`scientific_claim`) | scorecard final + SS-1 |

## 6. Orden recomendado y camino crítico

```
ya (CPU, sin GPU):  WP-E1 ─┐
                    WP-E3 ─┼─► informe de evaluadores (piso FEM, σn DFG)
                    WP-R0 ─┘
                    WP-C0 ─► WP-C1 ─► [firma + tramo B0/T1]
en paralelo:        WP-E4 (FEM DFG)
tras 2026-10-03:    T1 ─► B-freeze ─► T2 ─► R-freeze ─► T3a/T3b/T3c ─► T4 ─► SS-1 ─► WP-D
```

El camino crítico es **WP-C0/C1** (código y dry-run), no la GPU. Los WP-E1 y
WP-E3 ya pueden responder H5/H6 y decidir si DFG necesita un tramo T4 completo.
