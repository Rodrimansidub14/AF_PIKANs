# Piezo, etapa 1 (semilla 20260728, n = 1 por brazo): B y E, converso y directo — 2026-10-25
Kernels `piezo-s1a` (converso) y `piezo-s1b` (directo), Kaggle 2×T4, float64, Adam 20 000 sin L-BFGS, receta alineada con el paper. 4 corridas completas, B == ID verificado en GPU, artefactos (.npz de 2.5-5.9 MB) y W&B offline guardados. Fuente: results/piezo-s1a, results/piezo-s1b. **n = 1: lectura provisional, no pasa por la mediana de 5 de `predictions.md`.**

| Objetivo | métrica (paso 20 000) | B (RBF16) | E | razón E/B | log10 previsto (E) | dentro del intervalo [-0.65, -0.05] |
|---|---|---|---|---|---|---|
| Converso | l2re_u | 0.367 | 0.0495 | **0.135** (log10 −0.87) | −0.30 | no: mejora mayor a la prevista |
| Converso | l2re_v | 0.434 | 0.0292 | 0.067 | | |
| Converso | l2re_phi | 0.041 | 0.051 | 1.25 (peor) | | |
| Directo | l2re_u | 0.181 | 0.129 | **0.71** (log10 −0.15) | −0.30 | sí (diferencia 0.15 < 0.3) |
| Directo | l2re_v | 0.199 | 0.129 | 0.65 | | |
| Directo | l2re_phi | 0.184 | 0.151 | 0.82 | | |

Referencia del paper (MLP 3×50, 1 semilla): converso u 0.107, v 0.150, φ 0.045; directo u 0.117, v 0.153, φ 0.067.
Razón E/B de l2re_u por paso (converso): 0.99 (1 000), 0.90 (2 000), 0.28 (5 000), 0.12 (10 000), 0.07 (15 000), 0.13 (20 000). Directo: 2.33, 1.08, 0.94, 0.90, 0.61, 0.71.

## Lectura (provisional, n = 1)
- Regla 8 de `predictions.md` («mejora mayor a la prevista: E < 0.30»): se cumple en el converso (0.135); no en el directo. Contradice «indiferente», que no se había firmado; apoya la dirección «mejora» firmada, con magnitud mayor a la prevista en el converso.
- El B del converso (RBF16) queda mal convergido: l2re_u 0.367 y no monótono (0.319 en el paso 10 000, 0.367 en el 20 000), pérdida 0.158; satisface mejor φ (0.041) que el desplazamiento. Es el régimen de «B mal convergido» de KPP, donde el mecanismo ayuda mucho. Con un solo B, parte de la razón puede venir de la variabilidad de B y no del mecanismo; se necesitan las otras semillas.
- E en el converso empeora φ (1.25): la mejora en u y v no es uniforme en los campos.
- No se corrieron F ni J: H-PRED-1 (mejor y peor brazo, magnitud de F y E) no se puede evaluar todavía.
