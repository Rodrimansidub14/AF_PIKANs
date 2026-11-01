# EXP-0023 — Enmienda 1 al spec v2 (2026-10-20, 23:54)

Firmada por el investigador en el chat («firmo»). No modifica `spec.md`; lo enmienda en los puntos de abajo. Se redacta **después de la prueba previa 7b y antes de ver datos de ningún bloque**.

## Motivo
Prueba 7b (T4 y CPU): los brazos J, E e I divergen en los primeros 100 pasos de Adam; B (referencia) reproduce CPU/GPU con <= 0.11 %. Medido: valores singulares de P hasta 3.4e3. Causa probable (no probada): P = (B+eps I)^(-1/2) tiene unidades de 1/sqrt(curvatura); Adam descarta la escala absoluta del gradiente y da pasos `lr` por coordenada en c', que en c son hasta ~3.

## Cambio 1 — normalizacion de P por bloque
P = (B/m + eps' I)^(-1/2), con m = traza(B)/11 y eps' = 1e-3 (el eps relativo firmado no cambia). Efecto esperado: valores singulares de P entre ~1 y ~32. **Verificar antes de la 7b** que el rango medido caiga ahi; si no, parar y consultar.
Aplica a los brazos J, E, I y F segun su definicion de B en el spec. El brazo B (referencia) no cambia.

## Cambio 2 — prueba 7b repetida (criterio fijado ahora)
Cinco brazos, dos problemas, CPU y GPU, 100 pasos de Adam: perdida finita y, en el paso 100, <= 2x la perdida del brazo B. Hasta que pase no hay bloque. **No hay segunda enmienda:** si con esta receta aun diverge, cuenta como fallo de la ronda 1-B y rige la regla de salida firmada (T1-plan.md del EXP-0024).

## Resolucion de las cuatro decisiones de `budget.md`
1. Presupuesto: convencion T1 (10 % de Adam, 500 it. de L-BFGS), ~2 h estimadas. Corridas completas solo para un finalista (etapa V).
2. Criterio de paso de T1: mediana <= 0.90 en ambos problemas (regla de salida). Los umbrales del 0023 (0.85, 0.95, 0.95, 4/5) rigen en V y C. Brazos: B/J/E/I/F, B de referencia.
3. Semillas: 1-5 en T1. Las semillas 801-815 quedan reservadas al bloque de confirmacion (C).
4. Interpretaciones de Claude Code aceptadas como desviaciones anotadas: recalculo tambien antes del paso 1; J con el mismo eps que E; N = 2048 puntos equiespaciados del pool; F con la Gram de entradas uniformes, igual para todas las aristas.

## Firma
- [x] Enmienda 1 — Rodrigo Mansilla — 2026-10-20 (chat: «firmo»)

## Fe de erratas (2026-10-21, a pedido de la verificacion de Claude Code)
El rango «~1 a ~32» del Cambio 1 fue un error de calculo mio en el extremo bajo. El minimo de los valores singulares de P es 1/sqrt(lambda_max(B/m)+eps'), y como traza(B/m) = n, lambda_max(B/m) esta entre 1 y n: con n = 11 el piso es ~0.3, y con n = 528 (bloque por entrada) ~0.044. Medido (results/verify_sv.json): maximo 31.3-31.6 (J desde 5.8); minimo J 0.46-0.57, E 0.30-0.38, F 0.47, I 0.054-0.15 (capas 0 y 1). El maximo, que es el que provocaba la divergencia, cumple. El minimo se desvia del lado que **amortigua** las direcciones rigidas (paso en c de 0.05-0.6 x lr), no del que provoca divergencia. **Aceptado por el investigador (chat 2026-10-21)**; no es una segunda enmienda: el criterio de la 7b no cambia. Consecuencia a vigilar: en I el progreso en direcciones rigidas puede ser hasta ~20x mas lento; eso se leera en T1 como falta de efecto, no como divergencia.
