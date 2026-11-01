---
experiment_id: EXP-0009
slug: piezo-indirect-rbf
status: decided            # draft | preregistered | running | scored | decided
suite: exploratory
problem_id: piezo_bimorph_indirect_2d
model_id: model.rbf_pikan (arm) vs MLP 2-100-250-8 (control)
mechanism_id: none (baseline only)
hypothesis_id: none
preregistered_at:
preregistered_by:
---

# EXP-0009 — Piezo (bimorfo PVDF, formulación indirecta): problema, referencia y baseline

Exploratorio. `scientific_accepted=false`, `reference_accepted=false`. Nada aquí es un claim.

## Pregunta

¿Puede una RBF-PIKAN con la formulación mixta de 8 salidas de Daniel González igualar o superar a su PINN MLP contra su FEM, bajo una receta Adam→L-BFGS idéntica? (Etapa de baseline: se fija el entorno y se mide; todavía no hay mecanismo.)

## Problema

Viga bimorfa 2D de PVDF en tensión plana: L = 0.1 m, H = 1 mm, empotrada en x = 0 (u = v = 0), 100 V en y = H, 0 V en y = 0, polarización opuesta en las dos capas (interfaz y = H/2). Extremo x = L libre (σx = τ = 0, Dx = 0); superficies libres (σz = τ = 0). Salidas (u, v, φ, σx, σz, τxz, Dx, Dy); BC duras en u, v, φ.

Coeficientes: C11 2.1836e9, C12 0.6332e9, G 0.775e9 Pa; d31 2.2e-11, d33 −3.0e-11; e31 = +0.02904 y e33 = −0.05158 C/m² en la capa inferior (signos opuestos), capa superior con signo invertido; permiti­vidad a deformación constante 1.0406e-10 F/m.

## Convención de signos (decisión del investigador: la del paper / coherente con la literatura)

`results/reference/sign_check.py` evalúa, sobre el campo del FEM (201×21), el residuo de ∇·σ = 0 y ∇·D = 0 bajo la ley constitutiva del PINN para las cuatro combinaciones de signo de (e31, e33) y dos permitividades. **Solo la convención de signos opuestos con permitividad a deformación constante cierra el equilibrio eléctrico (residuo 2.6e-8)**; las de mismo signo dan 1.8e-2 a 3.6e-2. Con el FEM `poling_sign = −1` la capa inferior es e31 > 0, e33 < 0. Consecuencias:
- La tabla `tab:pvdf` de la metodología (e21 y e22 con el mismo signo) **no** es la convención que satisface el FEM; el cargador de Daniel invierte e33 (`coefficients[:,7] = -coefficients[:,7]`), lo que explica la diferencia entre su tesis y su código. Pendiente confirmar con él.
- Referencia de trabajo = FEM de Daniel con `poling_sign = −1`; su default (+1) invierte el sentido de flexión.

## Referencia

FEM P2 triangular de Daniel (scikit-fem 12.0.2), malla 400×16, convergencia verificada (flecha en la punta 3.2875e-5 → 3.2896e-5 m entre 100×4 y 400×16, y 0.06 % entre las dos finas) y estimación de bimorfo de mano 3.3e-5 m. Campo en rejilla 201×21: `results/reference/fem-grid-indirect-polingminus.npz`. **Es un FEM del mismo grupo que la PINN**: sigue pendiente un contraste de terceros (COMSOL o solución analítica de bimorfo) antes de aceptarla como referencia.

## Métricas

Error L2 relativo de u, v y φ sobre la rejilla del FEM (primaria: v y φ; u es la más difícil). Se reportan dos etapas, fin de Adam y tras L-BFGS, sin mezclar. Guardia: valor de la flecha en la punta frente al FEM.

## Controles y comparador externo

- Comparador externo: PINN de Daniel con sus pesos del paper contra su FEM (`poling_sign = −1`): u 0.195, v 0.193, φ 1.0e-3 (`results/reference/paper-weights-vs-fem.json`). Solo referencia; no se integra su código (licencia sin confirmar).
- Control propio: MLP 2-100-250-8 (pirámide, tanh, Glorot), mismo pool y receta.
- Brazo: RBF-PIKAN ancho 16, 11 bases por arista.

## Receta (Tier 1, CPU, float64)

Entradas normalizadas ξ = x/L, η = y/H; salidas con escalas características fijas; pools Sobol por semilla compartidos entre brazos (4 096 colocación, 256 por borde); pérdida = suma de cuadrados medios de los residuos constitutivos, de equilibrio y de borde; Adam lr 1e-3 y L-BFGS strong-Wolfe (max_iter 25, max_eval 125, historia 100). **Diferencias deliberadas con Daniel:** coordenadas y salidas adimensionalizadas; condiciones de superficie libre (σz = τ = 0 en y = 0, H) incluidas, que su código calcula pero no usa; sin pesos adaptativos.

## Estado

| Etapa | Estado |
|---|---|
| Tier 0 (`piezo_pinn.py smoke`: BC duras, gradientes finitos, descenso) | pasa (MLP 27 558 y RBF16 4 992 parámetros) |
| Tier 1 CPU (semilla 1001, 1 500 Adam + 120 s L-BFGS; MLP 871 s de Adam, RBF16 1 789 s) | **no aprende la flexión todavía**: MLP u 0.89 / v 0.89 / flecha 4.1e-6 m (FEM 3.29e-5); RBF16 u 1.03 / v 1.05 / flecha de signo dudoso. φ ≈ 1 % solo porque es casi lineal por la BC dura. Presupuesto muy corto; no es una comparación. Ver `results/tier1-cpu-20261001/` |
| Tier 2 (GPU, cinco semillas, receta larga) | requiere presupuesto aprobado |

## Corridas GPU (2026-10-01, Kaggle T4×2, semilla 1001, float64, exploratorio)

| Sesión | Variante | Pasos | MLP u / v tras L-BFGS | RBF16 u / v tras L-BFGS | Lectura |
|---|---|---|---|---|---|
| `piezo-probe` (0.19 h) | base | 3 000 Adam + 120 s | 0.65 / 0.65 (f32: 0.91 y L-BFGS se estanca en 11 it.) | 1.00 / 1.00 | float64 necesario; no aprende |
| `piezo-long1` (0.74 h) | base | 30 000 Adam + 600 s | 0.71 / 0.64 (flecha −1.3e-5 vs −3.29e-5 m) | 1.00 / 1.00 | Adam no avanza; L-BFGS sí algo en MLP; RBF en solución nula |
| `piezo-vsplit` (0.46 h) | `vsplit` 1e-4 (separa flexión de deformación en espesor) | 10 000 Adam + 300 s | 1.66 / 1.96 (L-BFGS diverge; Adam 0.94) | 1.08 / 1.14 | no ayuda |

### Receta del paper de Daniel (2026-10-04; `daniel_arm.py`, upstream e7856f1, semilla 20260728, float64, 20 000 Adam, sin L-BFGS)

Su pipeline completo (formulación por capas con interfaz, SiLU, características de fase, escalas por campo, remuestreo cada 200 épocas, enrutamiento de gradiente) con su MLP 3×50 y con un tronco RBF-PIKAN intercambiado en `model.net` (ancho 16, 11 bases por arista). Mismo evaluador propio contra el FEM (`poling_sign=−1`, 201×21).

| Brazo | Parámetros | u (L2) | v (L2) | φ (L2) | Flecha en la punta (FEM −3.29e-5 m) | Tiempo (T4, float64) |
|---|---|---|---|---|---|---|
| MLP 3×50 SiLU | 5 758 | 0.144 | 0.198 | 0.047 | −2.79e-5 | 3 873 s |
| RBF-PIKAN 16 | 5 376 | **0.090** | **0.094** | 0.0485 | −3.07e-5 | 9 222 s |

El MLP reproduce el orden de lo publicado por Daniel (preprint: 0.107 / 0.150 / 0.045; aquí 0.144 / 0.198 / 0.047, con otra malla/semilla de remuestreo y evaluador propio). **Con la receta correcta el RBF aprende la flexión y supera al MLP en u y v** (0.09 vs 0.14–0.20) con parámetros comparables, a ~2.4× el tiempo. φ es igual (~0.047, limitado por la referencia/formulación). Una sola semilla (la del paper); trayectoria del RBF no monótona (0.065 en la época 18 000, 0.090 al final). Se corren semillas 20260729–20260730 para dar n=3 (`piezo-daniel-seeds`).

**Semillas adicionales (2026-10-04, `piezo-daniel-seeds`, n=3 con la del paper):**

| Brazo | u | v | φ | Tiempo/corrida |
|---|---|---|---|---|
| MLP 3×50 SiLU | 0.137 ± 0.009 | 0.190 ± 0.011 | 0.046 ± 0.000 | ~4 030 s |
| RBF-PIKAN 16 | 0.094 ± 0.029 | 0.103 ± 0.065 | 0.047 ± 0.001 | ~9 510 s |

Por semilla (u, v) — MLP: 0.144/0.198, 0.127/0.178, 0.140/0.194; RBF: 0.090/0.094, 0.124/0.172, 0.066/0.043. El RBF queda por debajo en las 3 semillas (diferencia pareada en u: −0.054, −0.003, −0.073), con una semilla casi empatada, mayor dispersión y 2.4× el costo; en las tres su error aún bajaba al final de las 20 000 épocas. Con n=3 no hay significancia. Datos: `results/daniel-recipe-per-seed.csv`.

**n = 5 (2026-10-04, `piezo-daniel-seeds2`, semillas 20260728–32):**

| Brazo | u | v | φ | Tiempo/corrida |
|---|---|---|---|---|
| MLP 3×50 SiLU | 0.136 ± 0.007 | 0.186 ± 0.011 | 0.046 ± 0.000 | ~4 000 s |
| RBF-PIKAN 16 | **0.092 ± 0.021** | **0.104 ± 0.048** | 0.048 ± 0.003 | ~9 400 s |

Diferencia pareada RBF − MLP: u −0.054, −0.003, −0.073, −0.036, −0.055 (media −0.044 ± 0.026, **el RBF gana 5/5**); v −0.104, −0.006, −0.152, −0.083, −0.066 (media −0.082 ± 0.053, 5/5). φ es igual (~0.047). El RBF cuesta ~2.3× y tiene más dispersión; en 4 de 5 semillas su error aún bajaba o oscilaba en las últimas épocas (no convergido). Por semilla: `results/daniel-recipe-per-seed.csv`; hashes de checkpoints: `results/daniel-recipe-checkpoints-sha256.json`; figura: `figures/piezo-daniel-recipe-n5.png`.

Lectura del fallo anterior: no era una limitación del RBF; faltaban la formulación por capas con interfaz, las escalas por campo y el enrutamiento de gradiente. Nuestra reimplementación propia (`piezo_pinn.py`) queda como diagnóstico negativo.

### Barrido del 2026-10-03 (MLP, float64, 10 000 Adam + 300 s L-BFGS, semilla 1001; cuota ya renovada)

| Variante | u / v fin de Adam | u / v tras L-BFGS | Lectura |
|---|---|---|---|
| pesos adaptativos (balance por gradiente cada 100 pasos) | 0.94 / 0.92 | 0.60 / 0.60 | mejora con L-BFGS |
| balance + `vsplit` 1e-3 | **0.190 / 0.190** (flecha −2.67e-5 vs −3.29e-5 m) | 1.82 / 2.06 (diverge) | **aprende la flexión en Adam**, al nivel del PINN de Daniel (0.19); L-BFGS con los pesos congelados lo destruye |
| peso 100 en τ y σz | 0.91 / 0.90 | 1.02 / 0.98 | no ayuda |
| balance + peso 100 | 0.94 / 0.92 | 1.01 / 1.01 | no ayuda |

Lectura: la combinación que funciona es **balance adaptativo + separación de escalas de v (`vsplit` 1e-3)**. El fallo del L-BFGS es de pesos: al final los pesos adaptativos llegan a razones de 1e4–1e5 entre grupos, y L-BFGS optimiza ese objetivo desbalanceado. Siguiente sesión (`piezo-sweep2`): misma variante con MLP y RBF16, 10 000 y 30 000 pasos de Adam, y L-BFGS con pesos unitarios.

### Barrido 2 (2026-10-03, balance + `vsplit` 1e-3, semilla 1001, 30 000 Adam + 400 s L-BFGS con pesos unitarios)

| Brazo | u / v fin de Adam | u / v tras L-BFGS | Lectura |
|---|---|---|---|
| MLP | 0.256 / 0.278 (flecha −2.42e-5 m) | 0.96 / 1.02 | Adam aprende la flexión (peor que con 10 000 pasos: 0.19); L-BFGS con pesos unitarios también diverge |
| RBF16 | 0.994 / 1.002 | 1.00 / 1.00 | no aprende: sigue en la solución nula |

La segunda tanda (10 000 pasos) no corrió: la sesión superó el tope de arranque (3 200 s) porque dos kernels compartían GPU. Estado de piezo: **el MLP alcanza ~0.19–0.26 al final de Adam (nivel del PINN de Daniel), pero ni el L-BFGS ni el RBF16 funcionan todavía con esta formulación.** El L-BFGS parece moverse a mínimos de la pérdida que no son la solución (el residuo es indulgente con la discontinuidad de σx en la interfaz de las capas). Siguientes pasos: (a) interfaz explícita (continuidad de σz, τ y desplazamientos entre capas con σx/Dx discontinuos), (b) entrenar el RBF con inicialización o curriculum desde el MLP, (c) pedir la receta a Daniel.

### Formulación por capas (2026-10-03; una red por capa + continuidad en la interfaz)

| Variante (semilla 1001, 10 000 Adam + 300 s L-BFGS) | u / v fin de Adam | u / v tras L-BFGS |
|---|---|---|
| MLP, balance + `vsplit` 1e-3 | 0.62 / 0.63 (flecha −1.2e-5 m) | 1.09 / 1.09 |
| MLP, balance | 1.03 / 0.98 | 0.77 / 0.93 |
| RBF16, balance + `vsplit` 1e-3 | 1.00 / 1.00 | 1.33 / 1.03 |
| RBF16, balance | 1.01 / 1.01 | 0.74 / 1.00 |

La interfaz explícita **no mejora** (el MLP de una red con balance + `vsplit` llegaba a 0.19): la discontinuidad de σx no era la causa del fallo. El RBF16 sigue sin salir de la solución nula. Siguiente prueba: tasa de aprendizaje y ancho del RBF (`piezo-rbf-lr`).

`vsplit` 1e-3 (solo, sin balance) no corrió: el tope de arranque de la sesión (900 s) dejó fuera la segunda tanda (error mío de configuración). Diagnóstico de los puntos de control (CPU): ambos brazos cumplen las BC y la ley constitutiva casi (residuos 0.02–0.08 de la escala) en un estado "deformación casi nula con esfuerzos bloqueados" (σx ≈ −e31·Ey por capa); la salida de ese estado exige resolver una deformación en el espesor ~1e-5 veces menor que la flecha (aspecto 100:1, cortante τ≈0 por cancelación de ∂u/∂y y ∂v/∂x). **Ninguna variante probada converge; no hay comparación MLP vs RBF todavía.**

Siguientes opciones (sin gastar cuota a ciegas): (a) formulación estándar de 3 salidas (u, v, φ) con tensiones por autograd y un ansatz de viga (v = w(x) + corrección), (b) pesos adaptativos entre PDE y borde como Daniel, (c) pérdida de energía (formulación débil), (d) preguntar a Daniel qué receta le funcionó (épocas, pesos).

## Lectura del Tier 1

Con 1 500 pasos el problema sigue en la solución casi nula en u y v: la flexión exige que ∂u/∂y y ∂v/∂x se cancelen en el cortante (τ ≈ 0) con precisión de ~1e-3, lo que es lento en CPU y mal condicionado. El PINN de Daniel necesitó entrenamiento largo. Se confirma que el entorno corre y es diferenciable, no que la receta converja; la comparación MLP vs RBF requiere Tier 2 (GPU, receta larga).

## Pendientes humanos

1. Permiso y licencia de Daniel; aclarar el signo de e31/e33 entre su tesis y su código.
2. Contraste de terceros del FEM (COMSOL o analítico).
3. Presupuesto GPU para Tier 2 (cuota restante 3.55 h hasta el 2026-10-03).
4. Pre-registro antes de abrir un experimento de mecanismo.
