# Diseño del mecanismo: arquitectura «araña» (propuesta, 2026-10-25)

Estado: **propuesta para iterar con el investigador.** Sustituye a `DISENO_MECANISMO_FINAL.md`. Nada firmado ni lanzado. Principio acordado: el mecanismo no es una elección binaria entre variantes; es un **cuerpo** que siempre trabaja y **patas** que se activan en grado (0 a 1) según señales medibles del problema. Cada pata se considera aunque hoy tenga poco apoyo: su apoyo se registra y su peso se ajusta con la evidencia, en vez de descartarla.

## 1. Cuerpo (siempre activo): E
Precondicionador por arista, P = (B/m + 1e-3·I)^(-1/2) con B el bloque Gauss-Newton de la pérdida, recalculado cada 1 000 pasos (20 recálculos en 20 000 pasos; ≈ 4-5 % del tiempo en H100). **Evidencia:** única pieza que baja media y varianza en el converso de piezo (error 0.198 → 0.053, CV 0.81 → 0.42), neutra o mejor en Allen–Cahn, KPP y NS a receta completa, modesta en el directo (0.208 → 0.193). Condicionamiento: los bloques de piezo son casi de rango uno (cond ≈ 1e4); E los lleva a cond 1 por construcción. **Riesgos conocidos:** el potencial φ del converso (×1.18), cola de semillas malas en Burgers (EXP-0026/0027, sin explicar).

## 2. Patas (activación graduada a_i ∈ [0, 1])
La activación se calcula con una función suave de una señal s_i (p. ej. logística entre un umbral bajo y uno alto), con valores por defecto conservadores y calibrados con los problemas medidos; se ajustan al iterar. Una pata a medio activar se implementa por **interpolación** (por ejemplo P_efectivo = (1 − a)·I + a·P) o por mezcla del término correspondiente, no por un interruptor.

| Pata | Qué hace | Señal que la activa (cuándo y a qué costo) | Apoyo en datos | Apoyo en literatura/matemática | Estado |
|---|---|---|---|---|---|
| **P1. Escalares globales (REFUTADA para el voltaje flotante de piezo; se conserva como pata genérica de peso ≈ 0)** | Elimina cada escalar global en forma cerrada: la pérdida es cuadrática en él (ec. 75 del paper); V* = −⟨r₀, r₁−r₀⟩/‖r₁−r₀‖², dos evaluaciones, en cada recálculo | estructural (el problema tiene escalares fuera de las capas) y su razón de gradiente frente a los coeficientes (piezo directo: 1e-4 a 1e-3 frente a ≈ 6) | V̂_f ≈ 20 % bajo el FEM en B, E y J; error de φ concentrado en la banda del electrodo (35-38 %). **Comprobada: el V̂_f aprendido ya coincide con el exacto a 3 cifras en las 9 corridas del directo; sustituirlo no cambia φ, u, v ni la pérdida** (`analisis_piezo/check_Vf.md`) | VarPro, complemento de Schur: matemática válida, pero aquí no aporta | **refutada** para piezo; útil si otro problema tuviera un escalar mal optimizado |
| **P2. Balance por grupo físico** | Calcula B con cada grupo físico (σ, D, φ, equilibrio) normalizado, o mezcla B de la pérdida con B balanceado | concentración de la traza de salida en un canal (piezo: 82-99 % en σ_yy); costo: un Jacobiano por término en el recálculo | σ_yy domina; φ ×1.18 con E; D constitutivas ×1.33 | NTK (Wang–Yu–Perdikaris), gradient pathologies, FOSLS (Cai et al.) | por comprobar |
| **P3. Gram empírico por arista** | Variante barata de E sin Jacobiano del residuo: Gram de las bases bajo la distribución real de las entradas de cada arista (lo que F suponía uniforme), recalculado con el refresco | desajuste medido entre el Gram uniforme y el empírico (en piezo F empeora el cond en 82-88 % de las aristas); costo bajo | F falla en piezo, funcionó en 4 de 5 problemas del banco | FastKAN (normalizar entradas), RBF-QR (reparametrizar la base) | por diseñar y comprobar |
| **P4. Guarda de estabilidad** | Vigila saltos de pérdida (> 5× el mínimo previo en una ventana) y responde amortiguando P hacia I, o volviendo al último punto estable | tasa de saltos en ventana; costo despreciable | B directo 25-38 saltos; F 99-122; E 26-36; J 13-29; semillas malas de Burgers | Adam y el umbral de estabilidad (Bai et al. 2025 ⚠, solo resumen) | por diseñar |
| **P5. Arranque denso** | Más recálculos en los primeros pasos (0, 250, 500, 1 000, 2 000) | detección de llano (pérdida casi constante cerca de la solución nula) | B pasa 6 000 a 8 000 pasos en el llano en el converso; E sale antes en 2 de 3 semillas; el orden de salida no explica el error final | sin referencia directa | débil; se mantiene como candidata |
| **P6. Pata ligera (J)** | Diagonal del bloque, cuando E es demasiado caro o el bloque se estima mal | presupuesto de recálculo; costo menor que E | converso 0.089, directo 0.181 (el mejor); daña en Burgers y KPP corto | Jacobi (clásico) | disponible, ya medida |

**Patas fuera de la receta del paper (vía B, declaradas como extensión):**
| Pata | Qué hace | Señal | Apoyo | Estado |
|---|---|---|---|---|
| **X1. Consistencia constitutiva** | Término o peso adicional que penaliza la discrepancia entre las tensiones de la red y las reconstruidas desde los gradientes | discrepancia medida (σ_xx 0.8-1.75; D_y directo 0.8-1.0) | datos + Rezaei et al. 2022 (restricción de energía) | por especificar; cambia la pérdida |
| **X2. Pesos FOSLS** | Pondera los residuos por la norma de su espacio dual en vez de fijarlos | desbalance entre grupos (σ 57-76 %, equilibrio hasta 41 %) | medio | cambia la pérdida |
| **X3. Optimización del directo** | Decaimiento de paso, recorte de gradiente | tasa de saltos | bajo | fuera de la receta (Adam constante) |

**Patas retiradas, con razón:** F de Gram fijo (empeora el condicionamiento en 82-88 % de las aristas de piezo) y «Fn» (F con P reescalada: el problema es de dirección y no de escala). Quedan como brazos de contraste en el banco, no como patas.

## 3. Cómo se combinan (reglas de integración)
1. **Orden de operación en cada recálculo:** (i) P1 (resolver escalares, para que el residuo que ve E no cargue su error); (ii) construir B (con el balance de P2 según su peso); (iii) P = (B/m + ε·I)^(-1/2) ; (iv) interpolar según la activación; (v) registrar las señales (dispersión de trazas, concentración por canal, cond, tasa de saltos).
2. **Conflictos:** P4 puede reducir a otras patas (si hay inestabilidad, amortigua P); P3 sustituye a E (no se suman) cuando E es demasiado caro; X1-X3 se activan solo en la pista B.
3. **Presupuesto:** E ≈ 4-5 % del tiempo en H100; P1 añade dos evaluaciones de la pérdida por recálculo (< 1 %); P2 añade un Jacobiano por grupo (≈ 2-5× el costo de un recálculo); P4 despreciable.
4. **Señales por cada recálculo (contrato):** traza por arista (ya se guarda), dispersión log10 p95/p5, cuota de la traza por canal de salida, gradnorm por parámetro (ya guardado), tasa de saltos (necesita pérdida por paso: **hoy se guarda cada 100 pasos; hay que pasar a cada paso**).
5. **Dos pistas:** **A** (alineada con la receta del paper: cuerpo + P1, P2, P3, P4 sobre P, solo cambian el precondicionamiento y el escalar) y **B** (extensiones con cambio de pérdida u optimizador, declaradas como tales).

## 4. Qué se predice y cómo se comprueba cada pata (ciclo iterativo)
Para cada pata: (1) hipótesis con predicción de signo y magnitud antes de mirar; (2) **comprobación sin entrenar** sobre estados guardados donde exista; (3) búsqueda de literatura enfocada en esa pata, solo si la comprobación lo pide; (4) ajuste del diseño; (5) solo entonces, una corrida pequeña con especificación firmada.
| Pata | Comprobación sin entrenar (disponible) | Si se sostiene | Si no |
|---|---|---|---|
| P1 | sustituir V̂_f por V* en los estados finales del directo y medir el cambio de φ, u, v y pérdida | pasa a ser pata prioritaria del directo | se reduce su peso; se revisa por qué el escalar no explica el error (¿el sustituto de la carga neta?) |
| P2 | Jacobiano por término sobre los estados guardados: cuota de la traza por grupo y por canal | se diseña el balance; se calibra el umbral de la señal | queda como pata débil |
| P3 | distribución de entradas por arista (ocupación guardada) frente al Gram uniforme | se especifica el Gram empírico | se descarta como pata de reemplazo de E |
| P4 | pérdida por paso (**requiere registro nuevo**); relación de los saltos con los remuestreos | se especifica la respuesta | se mantiene en la pista B |
| P5 | distancia entre semillas en el espacio de coeficientes efectivos, estados cada 2 000 pasos | candidata a diseño | se retira |

## 5. Lo que este diseño no promete
No promete cerrar la brecha del directo con el paper (0.19-0.21 frente a 0.117) ni mejorar φ del converso; son los objetivos que las patas P1 y P2 intentan atender, con apoyo medio. La activación graduada y sus umbrales se calibran con siete problemas de banco y piezo (n = 3 semillas en piezo): son valores de partida, no resultados.
