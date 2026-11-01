# Ronda 1-C — problemas y referencias (BORRADOR SIN FIRMA)

No se entrenó ni se lanzó nada. Contenido: `problems_round1c.py` (ecuación, IC, BC, residual, transformación dura, receta nominal),
`make_references.py` (genera y valida referencias), `fisher_kpp_ref.npz`, `allen_cahn_d2e-4_ref.npz`, `provenance_round1c.json`.
Rejilla de referencia: t:101 × x:201, igual que `Allen_Cahn.mat`. Cargar con `load_reference(key)`.

## Fisher-KPP 1D (control suave fuera de clase)
`u_t = D u_xx + r u(1-u)`, D=1e-3, r=10, x∈[-1,1], t∈[0,1]. Referencia **analítica** (onda viajera de Ablowitz–Zeppetella,
c = 5√(rD/6) = 0.204, k = √(r/6D) = 40.8, centro x0=-0.2 → 0.004 en t=1). IC = la onda en t=0. BC: u(-1)=1, u(1)=0; la onda exacta
difiere de esos valores en ≤1.3e-14 (IC) y <3e-43 (derecha), así que la transformación dura `u0 + t(1-x²)NN` los cumple a precisión de máquina.
- Solver Chebyshev+Radau vs exacta: relL2 1.8e-12 (N=600), 2.1e-12 (N=900). Residual autograd de la exacta: 8.9e-16.
- `u_numeric` también va en el .npz por si se prefiere una referencia numérica.

## Allen-Cahn, coeficiente de dosis
`u_t = d u_xx + 5(u-u³)`, IC x²cos(πx), BC -1. **d = 2e-4** (elegido 2026-10-21; base 1e-3). Referencia numérica Chebyshev N=600, convergida (N=600 vs 900: 1.7e-7 relL2).

| d | max|u_x| | ancho ≈ √(d/5) |
|---|---|---|
| 2e-3 | 30.9 | 0.020 |
| 1e-3 (base) | 39.2 | 0.014 |
| 5e-4 | 49.3 | 0.010 |
| **2e-4 (elegido)** | 61.7 | 0.0063 |
| 1e-4 | 68.0 | 0.0045 |
Fisher-KPP: max|u_x| = 11.9 (frente mucho más suave que AC).

## Para tu decisión / firma
1. **Dosis.** RESUELTO: d=2e-4 (+57 % de gradiente máximo sobre la base). Riesgo abierto: el frente (~0.006) puede quedar mal resuelto con 8000 puntos y ancho 16; se evalúa con un piloto de una semilla, sin ajustar la receta por brazo.
2. **Descriptor retirado como definición de clase (2026-10-21).** Burgers (ν=0.01/π) tiene max|u_x| = 147.5 y quedaba dentro de "tipo AC", contradiciendo la hipótesis. La hipótesis se formula para la **familia Allen-Cahn**; max|u_x| solo describe dificultad. KPP (11.9) es control suave fuera de la familia; **predicción:** su ganancia de E/F frente a B es menor o nula respecto de `ac_d2e-4` (descriptivo). El contraste de Spearman de cond(P) pasa a Burgers (semillas nuevas 11-20, arnés de 1-B); ver `round1C-proposal.md` (e).
3. **Receta** = la de 1-B T1/V: 8000+400+800 puntos, Adam 4000, L-BFGS tope 500 iteraciones, RBF-PIKAN ancho 16. Adam 4000 y L-BFGS 500 se verificaron en los JSON registrados (25 corridas AC de T1 y 15 de V, todas `adam_steps=4000`, `lbfgs_iters=500`, `lbfgs_iter_cap`); los 40 000 pasos anteriores venían de la línea base de DeepXDE y no de 1-B. Los puntos y el ancho no quedan en los JSON: vienen del arnés compartido. En `SPECS`; no está firmada.
4. **Piso de la referencia base.** Mi solver difiere de `Allen_Cahn.mat` en 1.57e-4 relL2, independiente de N y de la tolerancia: es el piso de
   la referencia base, ~50× menor que el error mediano de E (7e-3). No afecta las razones B/E/F, pero conviene citarlo.
5. No hay código de EXP-0023 en las carpetas conectadas: el módulo es independiente. Falta cablearlo al arnés (semillas, MLflow, brazos).
