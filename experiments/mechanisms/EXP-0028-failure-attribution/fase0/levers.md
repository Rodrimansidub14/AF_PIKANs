# EXP-0028, Fase 0: mapa de palancas (reanálisis uniforme, CPU, datos guardados)

Generado por `fase0.py` (solo lee JSON ya guardados; no se ejecutó nada de entrenamiento). Criterio uniforme del spec v1.1, **por palanca × problema, sin agrupar problemas**, pareado por semilla con la base del mismo bloque:
sirve = mediana ≤ 0.90, mejora (razón < 1) en ≥ 70 % y ninguna semilla > 2×; daña = mediana ≥ 1.10 y empeora en ≥ 70 %; dependiente de semilla = ni una ni otra y alguna semilla fuera de [0.5, 2]; indiferente = el resto. Umbrales sin tocar.

Archivos: `table.csv` (una fila por brazo × problema × bloque × etapa; «pooled» = todos los bloques del mismo EXP con el mismo presupuesto; 0024 no se agrupa porque T0 y T1 tienen presupuestos distintos), `pairs.csv` (razones por semilla), `heatmap_palanca_problema_{lbfgs,adam}.{png,pdf}`, `spearman_efectos.{csv,png,pdf}`, `check_B_repro.csv`, `fase0.py`.
Etapas: `lbfgs_final` = error final (L-BFGS); `adam_end` = error al final de Adam. Se informan ambas en `table.csv`.

## 1. Palancas, señal, código y datos

| # | Palanca | Señal que la gobierna | EXP | Código | Problemas con datos pareados |
|---|---|---|---|---|---|
| 1 | Generación/remuestreo de puntos | residuo \|r\| (RAD, R); concordancia residuo–estructura (S, EXP-0014) | 0025 (R); 0014 (S) | `src/mechanisms/rad.py`, `rad_callback.py`; S: `self_gated_sampler.py` | R: AC d=1e-3, Burgers. **S (0014): sin datos** (`results/` vacío; spec nunca firmado, diferido el 2026-10-18) |
| 2 | Centros y anchos | cuantiles de activación (D1–D4, 0024-A); equidistribución del monitor de la referencia (O11, 0015) | 0015, 0024 | 0024: `regrid.py` en la carpeta del EXP (**no** está en `src/mechanisms/`); 0015: `step0/fit_step0.py` (ajuste supervisado, **no** es entrenamiento físico) | 0024: AC, Burgers (T1 GPU), AC y Heat (T0 CPU). 0015: Heat, Burgers, DFG (n=3), AC, NS2d-CG, piezo (n=1) |
| 3 | Parametrización de coeficientes | Gram de la base (F), diagonal del bloque (J), bloque por arista con residuo (E), bloque por entrada (I) | 0023, 0025 (E=aG), 0026 (E500/E1000, L=sin recálculo al arrancar L-BFGS) | `src/mechanisms/gram_preconditioning.py` | E, F: AC 1e-3, AC 2e-4, Burgers, KPP. J, I: AC 1e-3, Burgers (n=5). E500/E1000: AC 1e-3 (n=6), Burgers |
| 4 | Términos/pesos de la pérdida; foco por responsabilidad | A\* (error real, oráculo), ocupación | 0016v2/0017 (diagnóstico), 0018 (intervención) | `src/mechanisms/responsibility_focus.py`, `settling_focus.py` | 0018: AC 1e-3, Burgers (n=5). P, H, N usan la referencia (oráculo). 0017 no interviene: sin razón pareada |
| 5 | Fase del optimizador | k = fracción de Adam antes de pasar a L-BFGS | 0016 | `src/mechanisms/stall_handoff.py` | AC 1e-3 (n=3), Burgers (n=5); base = k=1.00 |
| 6 | Forma del residuo / autodiferenciación | sin mecanismo propio | — | — | Inventario, sin datos |
| 7 | Base/activación | sin mecanismo propio (solo 0015 toca el número de bases, U41) | — | — | Inventario; U41 de 0015 es el único dato (ajuste supervisado) |
| — | Observador (meta) | sondeo corto entre brazos aG/aR/a0 | 0025 | `src/mechanisms/observer_v0.py` | O, Om, Orand en AC, Burgers |

## 2. Resultado por problema (error final tras L-BFGS; mediana, k/n con mejora, razón máxima, etiqueta)

**Palanca 3, parametrización (EXP-0023, bloques T1+V(+P) agrupados dentro del mismo problema y presupuesto)**

| Brazo | AC d=1e-3 | AC d=2e-4 | Burgers | Fisher–KPP |
|---|---|---|---|---|
| E | 0.138, 10/10, 0.22, **sirve** | 0.306, 10/10, 0.47, **sirve** | 0.507, 12/20, 11.1, **dependiente de semilla** | 3.46, 2/10, 18.1, **daña** |
| F | 0.414, 10/10, 0.59, **sirve** | 0.514, 10/10, 0.90, **sirve** | 0.490, 7/10, 2.20, **dependiente de semilla** | 2.84, 2/10, 34.0, **daña** |
| J | 0.884, 3/5, 1.12, indiferente | sin datos | 5.54, 1/5, 12.1, **daña** | sin datos |
| I | 1.45, 2/5, 9.88, dependiente de semilla | sin datos | 181.9, 0/5, 296.7, **daña** | sin datos |

Burgers por bloque (E): T1 (1–5) 0.432, 4/5, máx 4.17; V (101–105) **5.701**, 1/5, máx 11.08 (daña); P (11–20) 0.383, 7/10, máx 5.13. F: T1 0.489, 4/5 (sirve); V 0.490, 3/5, máx 2.20.
Etapa Adam (fin de Adam): E sirve en AC 1e-3 (0.277) y 2e-4 (0.474), dependiente en Burgers (0.540, 15/20), daña en KPP (2.05). F: dependiente en AC (0.338, 8/10, máx 6.26), sirve en Burgers (0.241, 7/10).

**Variantes de E en 0026 (cadencia 500/1000; L = sin recálculo al arrancar L-BFGS)**

| Brazo | AC 1e-3 (n=6, semillas 3001–3006) | Burgers (pooled) | Burgers 3001–3010 | Burgers 101–105 | Burgers 1–5 |
|---|---|---|---|---|---|
| E1000 | 0.102, 6/6, sirve | 0.204, 8/10, máx 6.91, dep. | igual (solo ese bloque) | — | — |
| E1000L | 0.104, 6/6, sirve | 0.317, 13/20, máx 10.9, dep. | 0.212, 8/10, dep. | 3.128, 1/5, **daña** | 0.230, 4/5, dep. |
| E500 | 0.133, 6/6, sirve | 0.258, 17/20, máx 21.6, dep. | 0.203, 10/10, máx 0.95, **sirve** | 0.347, 4/5, dep. | 0.665, 3/5, máx 12.8, dep. |
| E500L | 0.135, 6/6, sirve | 0.166, 10/10, máx 0.83, **sirve** (solo 3001–3010) | igual | — | — |

**Palanca 3 dentro del observador (0025, E = aG fijo, K=500; semillas 2001–2005):** AC 1e-3: 0.204, 5/5, **sirve**; Burgers: 0.260, 5/5, máx 0.92, **sirve**.

**Palanca 1, R (RAD, 0025):** AC 1e-3: 0.654, 5/5, máx 0.71, **sirve** (Adam 0.893, 4/5); Burgers: 33.3, 0/5, máx 164.7, **daña**. KPP, AC 2e-4: sin datos. Observador: O AC 0.605 sirve / Burgers 2.50 dependiente; Orand AC 0.216 sirve / Burgers 63.5 daña; Om ≈ B (razón 1.000 en 3/5 semillas; AC y Burgers «dependiente de semilla» por una semilla 2.5×).

**Palanca 2, centros (0024; D1 cuantiles una vez, D2 mezcla 0.5, D3 cuantiles ×4, D4 solo T0):**

| Brazo | AC 1e-3 T1 GPU | Burgers T1 GPU | AC 1e-3 T0 CPU | Heat T0 CPU |
|---|---|---|---|---|
| D1 | 0.695, 5/5, 0.87, **sirve** | 21.2, 0/5, 146, **daña** | 0.677, 5/5, **sirve** | 1.44, 1/5, 2.67, **daña** |
| D2 | 0.675, 5/5, 0.75, **sirve** | 4.52, 0/5, 11.2, **daña** | 0.694, 5/5, **sirve** | 0.924, 4/5, indiferente |
| D3 | 0.765, 5/5, 1.09, **sirve** | 11.5, 0/5, 15.5, **daña** | 0.806, 4/5, sirve | sin datos |
| D4 | sin datos | sin datos | 0.807, 3/5, indiferente | sin datos |

(Las cifras de AC T1/T0 salen de `table.csv`, filas por bloque; coinciden con `round1-T1.md` salvo redondeo.) KPP, AC 2e-4, NS, DFG, piezo: sin datos.
**Palanca 2, bases (0015, ajuste supervisado, no es el entrenamiento físico):** U41 sirve en DFG (0.380, 3/3), Heat (0.585, 3/3), NS2d-CG (0.599, n=1), AC (0.877, n=1); indiferente en Burgers (1.11, 1/3) y piezo (1.09, n=1). O11 sirve en DFG (0.467, 3/3), daña en Burgers (1.14, 0/3), dependiente en Heat. X11: indiferente. Etiquetas con n < 5 (todas las de 0015) son provisionales.

**Palanca 4, foco por responsabilidad (0018; Adam compartido, solo etapa final):**

| Brazo | AC 1e-3 (601–605) | Burgers (301–305) |
|---|---|---|
| P (oráculo) | 1.05, 2/5, indiferente | 1.03, 1/5, indiferente |
| H (oráculo) | 1.70, 1/5, **daña** | 1.51, 1/5, **daña** |
| N (permutado, oráculo) | 1.37, 0/5, **daña** | 0.82, 4/5, **sirve** (control de ruido) |
| O (ocupación) | 1.34, 1/5, **daña** | 0.94, 3/5, dependiente |

**Palanca 5, traspaso (0016; k = fracción de Adam; base k=1.00):** Burgers k=0.25: 0.660, 3/5, dep.; k=0.40: 0.636, 4/5, máx 2.95, dep.; k=0.60: 1.41, 2/5, dep.; k=0.80: 0.972, 3/5, indiferente. AC (n=3): k=0.25 1.185 daña (0/3); k=0.40 1.44 indiferente; k=0.60 1.10 indiferente; k=0.80 1.18 dep. Con n=5 y 3 (y un único Adam por semilla) no hay evidencia de efecto de k (0016 ya hizo la prueba de permutación). La etapa «Adam» de esta palanca no es comparable (es el estado de Adam en k).

## 3. Lectura por problema (solo lo observado)

- **Allen–Cahn (1e-3 y 2e-4):** la palanca 3 sirve en todos sus brazos E/F (y en las variantes de 0026 y aG de 0025), también en Adam; reubicar centros ocultos (0024) sirve, más débil (≈0.7); R y Orand sirven en AC 1e-3; J es indiferente; I es dependiente; el foco por responsabilidad y el traspaso temprano no sirven.
- **Burgers:** la palanca 3 es «dependiente de semilla» con mediana baja (0.2–0.5) y cola de 2–20× (E, F, E1000, E500, E1000L); solo E500L y E500 en 3001–3010 y aG en 0025 quedan en «sirve». J, I dañan (I catastrófico), R y Orand dañan fuerte, y los centros ocultos (D1–D3) dañan 4.5–21×. Traspaso y foco: dependiente o indiferente.
- **Fisher–KPP:** solo E y F (n=10): ambos dañan (≈3×; mejora 2/10), a pesar de que F no usa residuo ni activaciones. Resto de palancas: sin datos.
- **Heat, NS2d-CG, DFG, piezo:** casi sin datos de intervención física; solo 0015 (supervisado, n ≤ 3) y D1/D2 en Heat (0024 T0).

## 4. Huecos «sin datos» (y por qué)

- S (0014): no hay corridas. C/regrid, W (foco, 0018) y H (traspaso) en KPP, AC d=2e-4, NS2d-CG, DFG, piezo: sin datos. R en AC d=2e-4, KPP, NS, DFG, piezo: sin datos. E/F/J/I en NS, DFG, piezo, Heat: sin datos (residuo vectorial no implementado). J/I en AC d=2e-4 y KPP: sin datos.
- 0017 (batería de señales) y 0022 son diagnósticos sin brazo de intervención: no tienen razón e_X/e_B. 0017: S4 no computable.
- Palancas 6 y 7: solo inventario.
- 0015 réplica en Kaggle (CPU, 20 trabajos) no está incorporada en las carpetas (el propio `results.md` lo dice).
- No existe B pareado de la misma sesión para 0015 supervisado más que C (se usó C como base); en 0018, B comparte el Adam con los demás brazos (se ignora la etapa Adam: razón idéntica a 1).

## 5. Inconsistencias (reportadas, **no resueltas**)

1. **E en Burgers: 5.70 frente a 0.26.** 0023 V (semillas 101–105): mediana 5.701, 1/5, máx 11.08. 0025 (E=aG, K=500, semillas 2001–2005): 0.260, 5/5, máx 0.92. En el medio: 0023 T1 (1–5) 0.432; 0023 P (11–20) 0.383, 7/10; 0026 E1000L en 101–105 reproduce el fallo (3.128, 1/5), en 1–5 da 0.230, en 3001–3010 0.212; E500 en 101–105 da 0.347 (4/5, una semilla 21.6×). Es decir, los datos son compatibles con un efecto de bloque de semillas (101–105 malo con E1000, 1 y 105 malas con E500) más cadencia/camino de código, y no hay resolución en lo guardado. Pooled por brazo, E en Burgers queda «dependiente de semilla» (0.51, 12/20), no «daña» ni «sirve».
2. **La base B no reproduce por semilla entre sesiones** (`check_B_repro.csv`): B(0026)/B(0023) en las mismas semillas de Burgers: bloque 1–5 mediana 0.68 (rango 0.33–1.88); bloque 101–105 mediana 1.40 (0.61–2.25). Parte de la «dependencia de semilla» de Burgers puede ser ruido de B (el control de Burgers varía > 10× entre semillas, 0024). No hay réplica de B para AC (0026 usa otras semillas).
3. **Criterio «sirve» frente a cola:** por la regla «sin semilla > 2×», E500/E1000 en Burgers quedan «dependiente de semilla» con medianas 0.20–0.26 y 8–17/10–20 mejoras; la etiqueta oculta que la mediana es de las mejores del mapa. Tal como fija el spec; se muestra la razón máxima.
4. **0027 (verificación):** hasta la primera intervención E y B deberían compartir trayectoria y no lo hacen (desviación 1.67 y 1.89 en el script de 0027): significa que el «pareo» E frente a B por semilla no controla el estado inicial exacto en 0026; puede inflar la dependencia de semilla.
5. **«E» no es el mismo brazo en todos los EXP:** 0023 (cadencia 1000, recálculo al arrancar L-BFGS), 0025 (aG, K=500, sin recálculo extra), 0026 (E500/E1000 ± L). Son filas separadas en el mapa; no se agrupan.
6. **0024 T0 (CPU, A600/L250/n3000) y T1 (GPU, A4000/L500) son presupuestos distintos**; no se agrupan. El nombre de brazo en T0 aparece como `C` dentro de los JSON (campo `arm`), se usó el campo `design`.
7. **0018 P, H, N usan la referencia (oráculo, `oracle_ceiling`)**; no son candidatos. N, el control permutado, «sirve» en Burgers (0.82, 4/5) sin información: es el ruido que el propio EXP-0018 midió, una advertencia de que un cambio arbitrario puede cruzar el umbral.
8. **Tabla 0016 Adam/«E_adam»** es el error del estado de Adam en el paso k, no un fin de Adam: no se interpreta como efecto de fase.

## 6. Correlaciones entre efectos por semilla (Spearman)

`spearman_efectos.csv/.png`: 124 pares con ≥ 5 semillas comunes (IC95 bootstrap, 2 000 réplicas; n = 5–10, muy baja potencia; AC 1e-3 y Burgers solamente, KPP/AC 2e-4 solo E–F). Salvedades: (i) misma identidad de semilla en EXP distintos no implica el mismo estado inicial (ver 4); (ii) las razones de brazos del mismo bloque comparten el denominador e_B, lo que induce correlación positiva espuria entre brazos de un mismo EXP. Lo que se ve, descriptivo: en Burgers los efectos de E son muy consistentes entre sí por semilla (E1000L–E1000 0.96; E500L–E500 0.98; E1000L[0026]–E[0023] 0.94, n=10), es decir, las semillas malas son malas para todas las variantes de E; en KPP E y F correlacionan (0.81, n=10). Correlación contra las propiedades de la fase 2: **pendiente** (la fase 2 aún no se ha medido).
