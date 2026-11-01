# I. Invariantes integrales de equilibrio (post hoc, exploratorio)

Seccion a seccion en x, campos finales (s020000), malla 201 x 21. Violacion relativa a la escala del FEM: `N` = max|∫σxx dy| / (max|σxx|·H); `V` igual con τxy; `M` = rms de M(x) / (max|σxx|·H²); `tracciones` = max(|σyy|, |τxy|) en y = 0 y y = H sobre max|σxx|. Converso: N = V = M = 0 exactos; directo: N = 0, V constante, M lineal. Medianas entre semillas; FEM = discretización de la referencia (cota de lo alcanzable).

| problema | brazo | n | N | V | M (rms) | tracciones | V medio | error V vs FEM | error M vs FEM |
|---|---|---|---|---|---|---|---|---|---|
| converse | FEM | 1 | 0.0561 | 0.00823 | 0.00214 | 2.37 | -0.000106 | nan | nan |
| converse | B | 5 | 0.0989 | 0.00451 | 0.0297 | 0.00614 | -0.0031 | 0.317 | 13.8 |
| converse | E | 5 | 0.024 | 0.00274 | 0.004 | 0.0152 | 0.000348 | 0.0869 | 2.61 |
| converse | M | 3 | 0.0614 | 0.00179 | 0.0101 | 0.00278 | -0.000745 | 0.162 | 4.66 |
| direct | FEM | 1 | 0.00283 | 0.00458 | 0.0977 | 0.414 | -0.101 | nan | nan |
| direct | B | 3 | 0.0147 | 0.00168 | 0.0976 | 0 | -0.0993 | 0.0449 | 0.118 |
| direct | E | 3 | 0.0302 | 0.00168 | 0.092 | 0 | -0.0993 | 0.0449 | 0.101 |
| direct | M | 3 | 0.0327 | 0.00168 | 0.0902 | 0 | -0.0993 | 0.045 | 0.113 |