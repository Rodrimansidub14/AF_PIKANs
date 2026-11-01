---
experiment_id: EXP-0026
slug: e-burgers-diagnostic
version: v1.1   # enmienda del 2026-10-22 antes de lanzar y antes de ver datos (ver «Paso 0 y enmienda v1.1»)
status: preregistered   # firmado por el investigador en el chat 2026-10-22: «redacta y preregistra el diagnóstico de E en Burgers (cadencia 500/1000 × recálculo al arrancar L-BFGS sí/no, semillas nuevas), y lánzalo si cabe en la cuota»
suite: tier2_candidate
problem_id: problem.burgers_pinnacle_1d
model_id: model.rbf_pikan (RBF48), arnés de 1-B sin cambios
mechanism_id: mechanism.gram_preconditioning (variante E; solo se parametrizan cadencia y recálculo al arrancar L-BFGS en el ejecutor del experimento, no en el mecanismo)
hypothesis_id: H-DX-1, H-DX-2
preregistered_at: 2026-10-22
preregistered_by: Rodrigo Mansilla (chat 2026-10-22); redactado por el agente
---

# EXP-0026 — ¿Por qué E ayuda en Burgers en un bloque y daña en otro? Cadencia × recálculo al arrancar L-BFGS

## Pregunta
La corrección de Gram E (EXP-0023) dio sobre Burgers, pareada con B: T1 de 1-B mediana 0.43 (mejora 4/5); V de 1-B mediana **5.70** (1/5); y dentro del observador de EXP-0025 (`fixed:aG`, cadencia 500, **sin** recálculo al arrancar L-BFGS) mediana **0.26** (4/5). Las configuraciones difieren en la cadencia de recálculo (1 000 frente a 500) y en el recálculo al arrancar L-BFGS (sí frente a no), y las semillas y el camino de código también. ¿Alguno de esos dos factores explica el daño, o es variabilidad por semilla?

## Diseño
Burgers, RBF48, Adam 2 000, L-BFGS 500 iteraciones, P normalizado (Enmienda 1), N = 2 048 puntos del pool, float32, arnés de 1-B sin cambios (`pre_trainer`, `t1_trainer`, `sweep`). Factorial 2 × 2 más control, **10 semillas nuevas 3001–3010** (no usadas; verificado por búsqueda textual en `experiments/` y `registry/`), pareadas entre brazos:

| Brazo | Cadencia | Recálculo al arrancar L-BFGS |
|---|---|---|
| B | — | — |
| E1000L | 1 000 | sí (= configuración de 1-B) |
| E1000 | 1 000 | no |
| E500L | 500 | sí |
| E500 | 500 | no (= el `aG` del observador, salvo el camino de código) |

50 corridas. La métrica es el error L2 relativo tras L-BFGS (`primary_final`).

## Hipótesis y reglas (fijadas antes de ver datos)
Todos los contrastes son pareados por semilla (n = 10). Sin corrección por multiplicidad: cada contraste es una pregunta separada y no se combinan.
- **H-DX-1 (el recálculo al arrancar L-BFGS daña).** q1 = mediana de e(E1000)/e(E1000L). **Respaldada** si q1 ≤ 0.90 y E1000 mejora a E1000L en ≥ 7 de 10 semillas.
- **H-DX-2 (la cadencia 500 ayuda).** q2 = mediana de e(E500)/e(E1000). **Respaldada** si q2 ≤ 0.90 y E500 mejora a E1000 en ≥ 7 de 10 semillas.
- **Descriptivos (sin criterio de paso):** e(E500L)/e(E1000L) (cadencia con recálculo) y e(E500L)/e(E500) (recálculo con cadencia 500).
- **Réplica de 1-B:** razón mediana e(E1000L)/e(B) y número de semillas con razón > 2 (cola). No es una prueba: dice si la configuración de 1-B reproduce su comportamiento en estas semillas.
- **Criterio de estabilidad por brazo E\*:** estable si mediana e(E\*)/e(B) ≤ 0.90, mejora en ≥ 8 de 10 y ninguna semilla con razón > 2.
- **Lectura:** si H-DX-1 → el recálculo al arrancar L-BFGS es el factor y se propone quitarlo (mecanismo candidato). Si solo H-DX-2 → la cadencia. Si ambas → ambos. Si **ninguna**: la discrepancia no se atribuye a estos factores con esta potencia y se reporta como variabilidad por semilla con la distribución agrupada de e(E)/e(B) (T1 y V de 1-B, EXP-0025 y este bloque), descriptiva.

## Condición de falsación y límites
Se **rechaza** H-DX-1 (H-DX-2) si q1 > 0.90 (q2 > 0.90) o gana en menos de 7 de 10 semillas. Un rechazo no se rescata bajando umbrales. **Potencia:** con 10 semillas solo se detectan efectos grandes; Burgers tiene variabilidad por semilla alta y una cola pesada (razones de 0.04 a 11 en 1-B), así que «no respaldada» no equivale a «sin efecto».
Amenazas: las semillas del bloque de EXP-0025 (2001–2005) y las de 1-B (1–5, 101–105) son distintas; la comparación con ellas es descriptiva. Este experimento no incluye Allen–Cahn ni KPP: no dice nada sobre otros problemas.

## Paso 0 y enmienda v1.1 (2026-10-22, antes de lanzar; sin datos de este experimento vistos)
Por una revisión recibida del investigador se leyó el código antes de correr. **Diferencias entre el camino de E en 1-B (`pre_trainer`) y el `aG` fijo del observador (`observer_v0`, EXP-0025 T1):**
1. **Cadencia:** 1-B recalcula en los pasos 0 y 1 000 de Adam (2 veces); el observador, en 0, 500, 1 000 y 1 500 (4 veces). Cada recálculo con optimizador **limpia el estado de Adam** de los coeficientes (`_set`).
2. **Recálculo al arrancar L-BFGS:** 1-B sí (`due_lbfgs`); el observador no.
3. **Semillas:** 1–5 y 101–105 frente a 2001–2005.
4. **Lo que NO difiere:** el pool de colocación **nunca se remuestrea** en E (ni en 1-B ni en `aG`); el recálculo no cambia el conjunto de puntos. `_set` **conserva los coeficientes efectivos** (la función representada no cambia): solo reparametriza (cambia la métrica). Además L-BFGS se construye nuevo en `lbfgs_stage`, sin historial de curvatura previo. Por tanto la hipótesis «el recálculo cambia el objetivo bajo el historial de L-BFGS» **no aplica a E tal como está implementado**; lo que sí cambia al refrescar es el condicionamiento del problema que ve L-BFGS. El mismo Adam (lr 1e-3, mismo bucle), el mismo N = 2 048 y el mismo `refresh` en ambos caminos.
**Cambios de diseño (decididos antes de ver datos):**
- **A. Réplica cruzada (descarta ruido de semillas), solo Burgers:** brazos B, E1000L (código de 1-B) y E500 (≈ `aG` del observador) sobre las semillas de 1-B T1 (1–5) y V (101–105): 30 corridas. Lectura preregistrada: «**reproduce V**» si, en 101–105, la mediana de e(E1000L)/e(B) ≥ 2.0 y E1000L empeora a B en ≥ 3 de 5 semillas; «**reproduce T1**» si en 1–5 la mediana ≤ 0.90. Si E1000L **no** reproduce V en esas semillas, el 5.70 de 1-B no es atribuible al código y la discrepancia se declara **ruido** (no se sigue con H-DX en Burgers como mecanismo). Si lo reproduce y E500 en esas mismas semillas mejora (mediana ≤ 0.90 y ≥ 3/5 mejores que B), el efecto sigue al código y no a las semillas.
- **B. Allen–Cahn también:** el factorial 2×2 + B se corre además en Allen–Cahn d = 1e-3 (RBF16, Adam 4 000) con las mismas semillas 3001–3010 (50 corridas más). **Cada problema se analiza por separado y no se agrupan.**
- **C. Regla de decisión sobre mediana y cola:** H-DX-1/2 usan la mediana y el conteo (≥ 7/10) como arriba; además se reporta por brazo y problema el número de semillas con razón > 2 respecto de B y la razón máxima. Un brazo con cualquier semilla > 2× no puede llamarse «estable».
- **D. Lectura de R (solo lectura, no modifica `aR` ni `rad.py`):** una corrida diagnóstica de `RadLever` sobre Burgers (modelo tras Adam 2 000 con B, semillas 3001–3003) registra puntos únicos del pool remuestreado, `effective_candidates` y la fracción de puntos en la banda del choque (|x| < 0.1) frente a la del pool uniforme. Descriptivo: separa «reemplazo» de «agudeza del peso».
## Presupuesto
Total 130 corridas en T4×2: 50 Burgers + 50 Allen–Cahn (semillas 3001–3010), 30 Burgers de réplica cruzada y 3 de lectura de R. Estimado ~90 min de pared (Burgers ~1 min por corrida, Allen–Cahn ~1.5 min; E con recálculo algo más). Topes: no se despacha trabajo nuevo después de 1.75 h; se matan los trabajos en curso a las 2.25 h.

## Superficie de edición
Solo el ejecutor `diag_e_burgers.py` y `analyze_diag.py` (nuevos, en este directorio). No se modifica el mecanismo, `evals/` ni el arnés de 1-B. No se importa `evals`.
