# EXP-0022 — resultados (2026-10-20)

Exploratorio. `scientific_accepted=false`. Spec v2 firmado el 2026-10-20; los criterios no se movieron. Solo CPU: 40 pares (Burgers 5 semillas × 5 traspasos = 25; Allen–Cahn 3 × 5 = 15), N = 2 048 puntos del pool de entrenamiento por estado, unos 25 min. Datos por estado en `results/states/`, criterios en `analysis.json` (`python3 analyze_exp0022.py`). Pruebas: `tests/test_exp0022_gn.py` (7).

## Veredicto según los criterios firmados

| | Burgers (n = 25) | Allen–Cahn (n = 15) |
|---|---|---|
| **H-COND-1**: mediana de ρ_d / ρ_sr (umbral ≤ 0.10) | **0.042** → se sostiene | **0.144** → no se sostiene (33 % de los estados bajo 0.10) |
| **H-COND-2** edge / crudo en κ_eff(100) (umbral ≤ 0.10) | 0.0084 → cumple | 0.0168 → cumple |
| **H-COND-2** edge / Jacobi en κ_eff(100) (umbral ≤ 0.33) | **0.346** → no cumple (rango p10–p90: 0.31–0.38; 36 % de los estados bajo 0.33) | 0.171 → cumple (todos los estados) |
| κ_eff(100) crudo, mediana | 2 062 | 2 710 |

Veredicto formal: **H-COND-1 no se sostiene** (falla Allen–Cahn) y **H-COND-2 no se sostiene** (falla Burgers contra Jacobi, por 0.016). En Burgers cae en la fila intermedia de la regla (cumple contra el crudo, razón a Jacobi entre 0.33 y 0.6) y en Allen–Cahn en la fila «H-COND-1 no se sostiene». La regla manda: el EXP-0023 **no pasa a firma sin decisión explícita del investigador**.

Lectura de los márgenes. En Burgers la razón a Jacobi está 5 % sobre el umbral y el ruido de muestreo del cálculo es del orden de 6 % (calibración con mitades del pool): es indistinguible del umbral y no se interpreta como evidencia en contra ni a favor. En Allen–Cahn ρ_d / ρ_sr baja con el traspaso tardío (0.23 con f = 0.25; 0.07 con f = 1.0).

## Lo que sí está medido

- **Hay mal condicionamiento real.** κ_eff(100) ≈ 2 000–2 700 en el estado de Adam. Jacobi lo baja a ~35 (Burgers) y ~280 (Allen–Cahn); el bloque por arista a ~12 y ~46: 2.9× y 5.9× mejor que Jacobi, y 60–120× mejor que el crudo.
- **El desplazamiento de L-BFGS es más plano que su magnitud implica** en Burgers (ρ_d es 4 % de ρ_sr, mediana), no tanto en Allen–Cahn (14 %).
- Descriptivo (sin hipótesis): el bloque por **entrada** (todos los coeficientes que leen la misma entrada) reduce κ_eff(100) a 0.0039 × el crudo en Burgers (el bloque por arista, 0.0084). No se eligió post hoc: es descriptivo.

## Hallazgo descriptivo sobre la representación (no era una hipótesis)

Número de condición de la Gram de la base por arista (mediana sobre aristas), frente al de entradas uniformes en [−1, 1]:

| Capa | Burgers: cond real / cond uniforme · rango efectivo | Allen–Cahn |
|---|---|---|
| 0 (entrada fija x) | 1.0 · 3.5 de 11 | 0.9 · 3.6 |
| 1 | **2.4e4** · **2.0** de 11 | **1.4e4** · 2.0 |
| 2 | 21 · 2.0 | 2.8 · 3.1 |

En las capas ocultas las activaciones se concentran (tanh) y cada arista usa en la práctica unas **2 bases de 11**; la Gram de la base queda 10⁴ veces peor que con entradas uniformes. Eso es una propiedad de la **representación entrenada**, no del solape de las gaussianas: una Gram fija con entradas uniformes (brazo F del EXP-0023) no la vería; una Gram calculada de las activaciones sí.

## Límites

- 40 estados salen de 8 trayectorias (5 y 3 semillas): no son independientes; el p por semilla solo informa.
- N = 2 048 subestima la cola del espectro; las conclusiones usan razones en m ≤ 100.
- Tres de los 40 estados se calcularon antes de firmar (declarado en el spec).
- Un desplazamiento plano y un buen precondicionador no prueban que el entrenamiento mejore; lo mide un experimento con entrenamiento.
- El hallazgo de capas ocultas es la mediana de números de condición, sensible a bases casi inactivas; es una descripción, no una prueba.
