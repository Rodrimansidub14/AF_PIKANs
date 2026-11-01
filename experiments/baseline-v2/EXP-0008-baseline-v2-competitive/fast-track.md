# Ruta acelerada (recomendada): baselines competitivos en ~2 GPU-h en vez de ~17.5

Estado: enmienda de borrador a EXP-0008. Sustituye el orden de `work-plan.md`
como camino recomendado; el diseño completo queda como plan de respaldo. No
firmada; no autoriza cómputo.

## 1. Evidencia que lo motiva (medida, no extrapolada)

Sonda Tier 0 en CPU local (i5-1035G1, 4 hilos), `var/agent_runs/exp0008/probes/`:
continuación **L-BFGS** (receta de `recipe-v2.md §2.2`) del checkpoint v1 de
Burgers, semilla 201, sobre el pool guardado; puntuación Cole–Hopf 1025×129.

| Pasos externos | Segundos (CPU) | Pérdida | L2 relativo CH |
|---:|---:|---:|---:|
| 0 (checkpoint v1) | 0 | 9.95e-5 | 0.02261 |
| 1 | 27 | 5.42e-5 | 0.02059 |
| 5 | 126 | 2.48e-5 | 0.00970 |
| 17 | 449 | 9.46e-6 | **0.00421** |

En 7.5 minutos de un portátil, **5.4× de mejora** sobre v1 (≥ el 4× de H1) y la
pérdida seguía bajando. **Advertencias:** una semilla; puntuada en E (ya
expuesta); sonda exploratoria, no resultado preregistrado; no alcanza aún la
banda (CH ≤ 1.13e-3).

## 2. Qué se recorta y por qué es defendible

| Se recorta | Ahorro | Justificación | Costo de recortarlo |
|---|---:|---|---|
| Etapa A1 (12 corridas de screening de receta) | ~2.3 h | La receta candidata deja de tener parámetros ajustados: **Adam v1 (const.) → L-BFGS** es la receta de PINNacle/DeepXDE, ya reproducida. Sin grados de libertad no hay nada que seleccionar ni separación dev/confirmación que proteger | warmup/cosine/lr/batch full quedan **sin probar** (mejora posible, no necesaria si se cumple G2) |
| Continuar desde checkpoints v1 en vez de re-entrenar desde cero | ~10 h | Las corridas Adam de 20 000 pasos ya existen (Burgers 201–205 locales; AC 401–405 y DFG en datasets de Kaggle). El diseño es el mismo de S3 y del refinamiento DFG, y da una diferencia **pareada dentro de cada corrida** (antes/después) | los resultados son "v1 + L-BFGS", no "receta v2 desde cero" |
| Heat: sin corridas nuevas | ~0.5 h | S2 ya usa Adam constante + L-BFGS de 20 000 pasos y da 2.06e-6, que cumple G1 (≤1e-5) y G2 | B-spline incompleto sigue incompleto; se declara |
| Controles MLP/B-spline bajo la misma receta | ~4–6 h | Para "competitivo frente a la literatura" bastan las recetas **reproducidas** (PINNacle, DeepXDE, MLP de S3) | no se podrá afirmar RBF vs MLP "bajo idéntica receta"; queda para la tesis |
| Escalera de capacidad: solo si hace falta | ~3–4 h | Se ejecuta únicamente si el R0 con L-BFGS no cumple G2 | — |
| Espera al refresco de cuota | tiempo | ~2 h caben en el saldo estimado (14.4 h, **por verificar con `kaggle quota`**) | — |

Los gates G1/G2 y sus umbrales **no se mueven**: es la misma vara con menos
corridas, no una vara más baja.

## 3. Plan

### FT-0 — CPU local, sin cuota (puede empezar ya)

- WP-E1: puntuar v1 y RAD en el FEM 101×11 (mide el piso).
- WP-E3: scorer σn único para DFG 301–305 y 501–505 (decide si DFG necesita GPU).
- WP-R0: backfill de `registry/`.
- Opcional, si lo autorizas: **continuación L-BFGS de Burgers en CPU local**,
  5 semillas × 900 s ≈ 75 min de reloj, cero cuota. La sonda mide ~26 s por paso
  externo; los 900 s dan ~35 pasos, menos de lo que hará la GPU, así que sería
  una cota inferior del efecto.

### FT-1 — una sola sesión GPU (T4×2), pasos como presupuesto, tiempo como tope

| Bloque | Corridas | Presupuesto por corrida | Sesión (est.) |
|---|---:|---|---:|
| Burgers RBF R0: L-BFGS desde v1 201–205 | 5 | `min(15 000 pasos externos, 900 s)` | ~0.65 h |
| Allen–Cahn RBF R0: L-BFGS desde 401–405 (S3 usó 240 s) | 5 | `min(15 000 pasos externos, 1 200 s)` | ~0.85 h |
| DFG RBF: L-BFGS desde 301–305, **solo si WP-E3 lo pide** | 5 | 900 s | ~0.65 h |
| Reserva de infraestructura (D0–D5 obligatorio antes) | — | — | ~0.3 h |
| **Total** | 10–15 | | **≈ 1.6 h (2.2 h con DFG)** |

Cada corrida guarda checkpoints al 25/50/75/100 % del presupuesto y, si se
puede, se prueban 2 procesos por GPU (la sonda de WP-B0 lo mide): podría
reducir la sesión a la mitad. Puntuación en un job aparte: E + externos
(FEM 101×11, `.mat`, FEATFLOW).

### FT-2 — solo si falta la banda

| Condición | Acción | Costo |
|---|---|---:|
| Burgers R0 no cumple G2 (CH ≤ 1.13e-3) | rungs R3 y R4 con **Adam 5 000 full-batch → L-BFGS** (paridad PINNacle), 3 semillas dev nuevas (611–613); luego confirmar el ganador en 201–205 | ~1.4 h + ~0.8 h |
| AC R0 no cumple G2 (≤ 2e-3) | rung R3, 3 semillas dev; luego confirmar | ~0.5 h + ~0.8 h |

Aquí sí se conserva la separación dev/confirmación (esa es la parte con
grados de libertad).

### Presupuesto resumido

| Escenario | GPU-h |
|---|---:|
| Mejor caso (R0 + L-BFGS ya cumple G2 en Burgers y AC; DFG cubierto por WP-E3) | ≈ 1.6 |
| Caso medio (DFG + un rung en Burgers) | ≈ 4 |
| Peor caso razonable (ambos rungs, confirmaciones, DFG) | ≈ 6 |
| Diseño completo original (respaldo) | ≈ 17.5 |

## 4. Qué NO entrega esta ruta (declarado)

- No prueba warmup/cosine/lr/full-batch como factores.
- No compara arquitecturas bajo idéntica receta (RBF vs MLP vs B-spline).
- No produce el paquete de señales v2 densas por checkpoint salvo que se
  active (es barato: se puede añadir a la continuación sin coste relevante de
  GPU y sí se recomienda).
- No repara la referencia FEM de DFG ni acepta referencias; sigue siendo
  `screening_only` donde `reference_accepted=false`.

## 5. Riesgos abiertos

- El tiempo de GPU por paso externo **no está medido**; el CPU ofrece una cota
  superior (~26 s/paso). WP-B0 lo mide en ≤ 0.1 h antes de comprometer topes.
- L-BFGS full-batch puede sobreajustar el pool fijo (pérdida baja, error de
  campo no): por eso el error de campo se puntúa en todos los checkpoints de
  fase y el gate se aplica al final, no al mejor.
- Una sola sonda (semilla 201): la mejora de 5.4× puede no ser representativa;
  las 5 semillas de FT-1 lo dirán.
