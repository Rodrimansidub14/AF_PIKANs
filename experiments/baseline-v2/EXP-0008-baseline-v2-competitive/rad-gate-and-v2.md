# Estudio de señal SS-1 y rediseño de RAD (RAR-D) — condicional

Estado: borrador de EXP-0008. Punto 4 del análisis. **Fuera del alcance de
aprobación de EXP-0008**: aquí se define el *gate* (SS-1, que sí es análisis
de CPU sobre checkpoints v2) y el diseño de la intervención, que exige su
propio preregistro si el gate se cumple.

## 1. Por qué RAD-v1 no es evidencia contra la idea, pero sí contra ese diseño

Resultado EXP-0006 v4 (5 parejas Sobol/RAD, semillas 201–205, RBF v1, Adam
20 000 pasos): L2 CH 0.0395 → 0.2127, razón geométrica 5.9× peor (IC95
2.8–12.1); el error en el frente empeoró 4× (0.216 → 0.883) y el del
complemento 11× (0.0074 → 0.0849). No es un compromiso entre regiones: degradó
todo.

Rasgos del diseño v1 compatibles con esa degradación (**hipótesis, no
aisladas**):

| Rasgo de RAD-v1 | Evidencia en los artefactos | Riesgo |
|---|---|---|
| Reemplazo total del pool activo cada 1 000 pasos con `lr=1e-3` constante y estado de Adam conservado | 18 refrescos por corrida | cambios bruscos de objetivo |
| Muestreo con reemplazo, 8 192 sorteos de 32 768 candidatos | puntos únicos 4 484–6 478 de 8 192 | duplicados; menos cobertura |
| Probabilidad `∝ (|r|/max|r|)²` normalizada por el máximo | ESS de candidatos entre 321 y 11 543 (rango entre refrescos) | pocos outliers dominan |
| Sin corrección de importancia | declarado en el diseño | la pérdida deja de estimar la integral uniforme |
| Modelo base no convergido | error Sobol entre 0.021 y 0.083 según semilla | el residual no es un buen proxy |
| Señal débil previa | Spearman 0.12–0.30 residual↔error; decil superior 2.6–4.8× | la propia auditoría no cumplía la precondición |

## 2. SS-1 — estudio de señal (gate; solo CPU)

**Cuándo:** después de la etapa C de Burgers y AC (o de A2, si se adelanta
para tomar decisiones de presupuesto), sobre los `dense_signals_*.npz` y las
etiquetas de `V`. **No requiere GPU.**

**Datos.** Checkpoints `c_adam` y `c_lbfgs_100` de todas las semillas de la
etapa C, más los checkpoints v1 (201–205) como conjunto **fuera de
distribución**. Unidad de análisis: **la semilla**; los puntos de una grilla no
son réplicas independientes.

**Modelos** (predicen `log|error|` y la pertenencia al decil superior de
error, en puntos de la grilla densa que no fueron de entrenamiento):

| Modelo | Rasgos |
|---|---|
| M0 | residual |
| M1 | M0 + contexto geométrico (`t`, `|x|`, distancia estimada al frente/interfaz desde el propio `u`) |
| M2 | M1 + `|u_x|`, `|u_xx|`, `|∇residual|` |
| M3 | M2 + señales RBF (entropía de ocupación local, sensibilidad `‖du/dθ‖`, saturación) |

Ajuste: regresión ridge o árboles pequeños con validación **dejando fuera una
semilla completa**; se reportan AUC del decil superior, Spearman y `R²` fuera
de muestra; incertidumbre por bootstrap de **semillas**.

**Criterios de paso (pre-registrados):**

| ID | Condición | Qué habilita |
|---|---|---|
| P0 | en modelos v2 convergidos: Spearman(residual, error) ≥ 0.5 **y** AUC(M0, decil superior) ≥ 0.75 en ≥ 2 de 3 problemas con referencia (Burgers, AC; DFG si hay FEM) | RAD residual-only es **elegible** (el residual es un proxy útil del error) |
| P1 | `ΔAUC(M2 − M1) ≥ 0.05` con IC95 (bootstrap de semillas) `> 0` en ≥ 2 problemas | señales de derivadas justificarían una política guiada por derivadas |
| P2 | `ΔAUC(M3 − M2) ≥ 0.05`, IC95 `> 0` en ≥ 2 problemas | señales RBF aportan información **más allá del residual y la geometría** |
| P3 | costo de calcular las señales de la política ≤ 10 % del costo de paso | viable |
| P4 | direcciones consistentes en checkpoints `c_adam` y `c_lbfgs_100` | robusta al punto de entrenamiento |

Si P0 falla (esperable si el residual sigue correlacionando 0.1–0.3 con el
error), **no se ejecuta RAD residual-only**; el hallazgo se registra como
resultado, y el siguiente candidato es una política guiada por P1/P2 (si
pasan) o bases aprendibles. Asociaciones negativas se registran igual.

## 3. Diseño RAR-D-v2 (solo si P0 y P3 pasan; preregistro propio)

Objetivo: cambiar **un** factor (la distribución de puntos) frente a Sobol
fijo, con el mismo presupuesto final de puntos y la misma receta AF-R2.

| Elemento | Diseño v2 | Motivo |
|---|---|---|
| Presupuesto de puntos | 8 192 finales, igual al control | mismo costo de paso y de L-BFGS |
| Composición | 4 096 Sobol **fijos** (nunca se quitan) + 4 096 ranuras adaptativas | evita perder cobertura uniforme |
| Rondas | 8 rondas de +512 puntos, en pasos Adam 4 000, 6 000, …, 18 000 | adición gradual (RAR-D), no reemplazo total |
| Candidatos | 32 768 Sobol **frescos** por ronda (flujo único por ronda) | soporte renovado, reproducible |
| Puntaje | `r̃ = min(|r|, q99(|r|))` (winsorización al percentil 99) | acota el peso de outliers |
| Distribución | `p_i ∝ r̃_i^k / E[r̃^k] + c` con `k=2, c=1` (parámetros de RAD de Wu et al. como punto de partida; **no** se afirma óptimo) | mantiene masa uniforme |
| Selección | sin reemplazo (Gumbel top-k con clave `log p + G`) | sin duplicados |
| Corrección de importancia | **ninguna** en el brazo primario; brazo de ablación `IW` (pesos autonormalizados `w_i ∝ 1/p_i`, recortados a 4× la media) con 3 semillas | decisión explícita: RAD busca el error de solución, no la integral uniforme |
| Fase L-BFGS | sobre el pool final fijo de ambos brazos | comparación limpia |
| ESS floor | si `ESS(p) < 0.05·N_cand`, se mezcla con uniforme hasta alcanzarlo y se registra | evita degeneración |
| Lr | igual que AF-R2 (warmup+cos); las rondas ocurren con `lr` ya decayendo | evita saltos de objetivo con `lr` alto |

**Diseño experimental:** 5 parejas (mismas semillas y pools iniciales que la
etapa C), 10 corridas + 3 de ablación. Evaluación: E + externos; métrica
primaria `log(L2_RAD/L2_Sobol)` por pareja.

**Gate propuesto (a firmar en su propio preregistro):** razón geométrica
`≤ 0.8` con IC95 superior `< 1`; guardas: RMSE del complemento ≤ 1.25× Sobol,
RMSE del frente ≤ Sobol, costo ≤ 1.15×. **Falsación:** IC95 que incluye 1 o
razón > 1 ⇒ RAD queda falsado para Burgers bajo receta convergida.

**Señales adicionales:** ESS y puntos únicos por ronda, distancia entre
distribuciones sucesivas (JS), mapa de densidad antes/después, trayectoria del
error V a lo largo de las rondas, costo de cada ronda.

## 4. Lo que no se hace

No se lanza RAD hasta que SS-1 esté leído por el humano; no se usa el error de
referencia como entrada de la política; no se compara contra RAD-v1 (diseño
distinto); no se mezcla con bases aprendibles ni con balanceo de pérdidas.
