# EXP-0016 v2 — resultados de las etapas A y B (2026-10-19)

Exploratorio. `scientific_accepted=false`. Spec firmado el 2026-10-19 (etapas A y B); los umbrales no se movieron. Las etapas C y D **no se corrieron** (requieren segunda firma, y la regla de decisión del spec cierra el camino al brazo T si H-AWARE-1 falla). Datos y registros: `results/stage-a/` (8 JSON de barrido, 8 trazas de Adam, puntos de control, `stage_a_analysis.json`, `stage_b_analysis.json`), kernel `rodrimandub/af-pikans-exp-0016-stage-a`.

## 1. Qué se corrió

| | |
|---|---|
| Etapa A | Burgers RBF48, semillas 301–305 (20 000 Adam f32, L-BFGS tope 600 s) y Allen–Cahn RBF16, 601–603 (40 000 Adam, tope 300 s). Malla de traspaso 0.25/0.40/0.60/0.80/1.00, un Adam por semilla, calentamiento por proceso. |
| Sesión | 1.85 h en T4×2, 8/8 trabajos (esperado 3.1 h). Los 40 L-BFGS terminaron por **meseta** (130–245 s en Burgers, 95–185 s en Allen–Cahn): el tope de reloj nunca actuó. |
| Etapa B | CPU, sobre los registros de A: mapa de asentamiento de la capa 0 (cada 500 pasos), residuo de la red en cada checkpoint y, solo como etiqueta, el error contra la referencia. |

## 2. Etapa A — criterios firmados

| Hipótesis | Burgers (n = 5) | Allen–Cahn (réplica, n = 3) |
|---|---|---|
| H-HANDOFF-1 (techo de error: GM de E(k*)/E_fijo ≤ 0.80 y ≥ 4/5 semillas) | **se sostiene**: GM 0.430, 4/5 semillas ≤ 0.80 (razones 0.253, 0.613, 0.298, 0.319, 1.000) | no: GM 0.846, 1/3 |
| H-HANDOFF-T (techo de costo: E(k) ≤ 1.05·E_fijo con reloj ≤ 0.60·T_control) | **se sostiene**: 4/5 semillas | 2/3 |

### Cómo leerlo (análisis posterior, **no preregistrado**)

El criterio H-HANDOFF-1 toma el mínimo de 5 valores de E(k), uno de los cuales es el propio control. Si E(k) fuera ruido intercambiable entre traspasos, la razón saldría < 1 igualmente. Dentro de una misma semilla de Burgers, E(k) varía 3.6× (rango de log E: 1.28) y **no hay tendencia sistemática en k** (E(k)/E(1.00): GM 0.69, 0.70, 0.88, 0.98 para k = 0.25, 0.40, 0.60, 0.80; pruebas de signos p ≥ 0.375). Prueba de permutación (`null_stage_a.py`, 100 000 reasignaciones del "control" dentro de cada semilla):

| | observado | p |
|---|---|---|
| Burgers, media geométrica del mínimo/control | 0.430 | **0.234** |
| Burgers, semillas con razón ≤ 0.80 (4/5) | 4 | **0.481** |
| Allen–Cahn, media geométrica | 0.846 | 0.896 |

El techo de Burgers que cumple el criterio firmado **no se distingue de elegir el mejor de cinco valores ruidosos**. Lo que sí se observa, sin significancia, es que un traspaso temprano (k = 0.25 y 0.40) no empeora el error mediano en Burgers (GM 0.69–0.70) y ahorra 60 % a 75 % del tiempo de Adam; en Allen–Cahn ocurre lo contrario (GM 1.23–1.37, 0/3 y 1/3 semillas mejor). Con n = 5 no hay evidencia de un efecto de k.

## 3. Etapa B — criterios firmados

**B1, H-AWARE-1 (¿la representación sabe dónde falla?). No se sostiene en ninguno de los dos problemas.**

| Problema | Spearman mapa–error (umbral 0.30) | Solape 5 % mapa–error (umbral 0.25) | Residuo: Spearman / solape | Semillas donde el mapa supera al residuo |
|---|---|---|---|---|
| Burgers | **+0.145** | **0.004** | +0.455 / 0.393 | 0/5 |
| Allen–Cahn | **+0.190** | **0.000** | +0.371 / 0.011 | 1/3 |

(Mediana sobre los 25 y 15 checkpoints de Adam; azar del solape = 0.05.) En Burgers el mapa de asentamiento tiene correlación **negativa** con el error en 10 de 25 checkpoints, sobre todo temprano (5 000–16 000 pasos), y su solape con el 5 % de mayor error es casi nulo: el error se concentra en el frente y el mapa de franjas no lo ve. El residuo, en cambio, localiza la zona del frente (concentración que ya había visto EXP-0013).

**B2, H-HANDOFF-2 (¿el disparo cae donde conviene?). No se sostiene.** El disparo interno (80 % de las bases asentadas) **no dispara en ninguna de las 5 semillas de Burgers** (k_t = k_max, EI = 0) y en 1 de 3 de Allen–Cahn (EI −0.33). El disparo por estancamiento de la pérdida (línea base) tampoco: EI en Burgers 0.32, −0.09, 0.12, 0.0 (mediana ≈ 0.06). La señal interna es demasiado estricta para las bases de la capa 0 en 20 000 pasos de Adam (no se recalibró: umbrales firmados).

### Comprobación posterior (no preregistrada): ¿alguna otra señal propia le gana al residuo?

`posthoc_structure.py`, mismos checkpoints: la estructura |∇u| de la red tampoco supera al residuo (Burgers: Spearman −0.315, solape 0.238 frente a +0.455 y 0.393; Allen–Cahn: +0.290 / 0.000 frente a +0.371 / 0.011; 0 de 8 semillas). Entre las señales sin referencia probadas, **el residuo es la mejor localizadora del error durante Adam en Burgers**.

## 4. Qué decide la regla del spec

- **H-AWARE-1 falla → el "dónde" interno se cierra en esta construcción** y se reporta como resultado negativo; **no se pasa a la etapa C con el brazo T** (mapa interno + traspaso interno).
- H-HANDOFF-1 y H-HANDOFF-T se sostienen en Burgers por la letra del criterio, pero H-HANDOFF-1 no es distinguible de ruido (permutación) y Allen–Cahn no lo replica.

## 5. Límites

- La capa 0 es separable por dimensión: el mapa es suma de franjas en x y en t, no una región arbitraria. El resultado negativo es de **esta construcción** (asentamiento de las bases de la capa 0), no de toda señal interna posible.
- n = 5 y 3; el oráculo es un techo de esta malla de cinco traspasos con una sola trayectoria de Adam por semilla, y E(k) de una semilla no son muestras independientes.
- Heat (control negativo) no se corrió.
- Etiquetas de error y residuo evaluados en una submalla de la referencia (Burgers: cada 4.º x y 2.º t); Allen–Cahn en la malla completa.
- "Asentarse no es converger": la señal puede no distinguir bases estancadas de bases bien ubicadas, y el resultado lo confirma en la práctica.
