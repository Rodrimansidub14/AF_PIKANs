# Mecanismo condicional «E si cond(P) supera un umbral» — BORRADOR, NO EJECUTADO, SIN FIRMA (2026-10-21)

**Estado: conjetura.** Solo tiene sentido si el predictor preregistrado de 1-C (cond(P) inicial → log razón, dirección negativa) **se sostiene** en el T1 de 1-C. Hoy su único soporte es un hallazgo nominal post-hoc en Burgers (ρ = −0.72, p = 0.019 sin corrección, n = 10) y es un argumento a favor del umbral, no una prueba. Si el predictor no se sostiene, este documento se archiva sin implementar.

## Idea
Activar el precondicionamiento E solo en los problemas/semillas donde el modelo inicial está mal condicionado (cond(P) alto) y dejar el baseline B donde no. Es una política de selección, no un mecanismo nuevo de representación.

## Entradas
- Modelo inicial (pesos de la semilla) y el pool de colocación del entrenamiento (sin referencia, sin residuo adicional más allá del de la propia pérdida).
- Sonda `p_cond_probe` (definición exacta en `round1C/arnes_round1c.py`): mediana sobre grupos-arista de cond(P_g) de la capa de índice 1, N = 2048 puntos equiespaciados del pool, P de la Enmienda 1.
- Parámetro `tau` (umbral). **Conjetura:** se fija con el T1 de 1-C (semillas 6–10) y se prueba **una sola vez** en V de 1-C (semillas 106–110); `tau` no se ajusta con V.

## Salida
Una decisión binaria por corrida, antes de entrenar: `E` si cond(P) ≥ `tau`, `B` en otro caso. Si `E`, el entrenamiento sigue exactamente la receta del brazo E (recálculo en el paso 0, cada 1000 pasos y al arrancar L-BFGS). La decisión y el valor de la sonda se registran en el JSON de la corrida.

## Qué se mide
Para cada problema (réplica Allen–Cahn d = 2e-4, control KPP, Burgers bloque P, más los de 1-B como contexto): razón pareada error(condicional)/error(B) por semilla, mediana, media geométrica, mejora en k/5, razón máxima; fracción de semillas que activaron E; y la comparación con «E siempre» y «B siempre». Criterio de paso: el de siempre (mediana ≤ 0.90, ≥ 4/5), sin cambios. Falsificación: si el condicional no mejora a «E siempre» en el problema donde E empeora (por ejemplo Burgers), o si decide al azar respecto de la razón por semilla, se descarta.

## Límites declarados
- Con n = 5–10 semillas y un solo contraste previo, la selección de `tau` es frágil; un umbral ajustado en T1 puede sobreajustar.
- La sonda cuesta un recálculo de bloques (≈ 2–5 s en T4) más una factorización por grupo.
- No distingue por qué una semilla está mal condicionada; no explica el mecanismo.
- Depende de la definición exacta de P (Enmienda 1) y del conjunto de N puntos; cambiar cualquiera de los dos invalida el umbral.
