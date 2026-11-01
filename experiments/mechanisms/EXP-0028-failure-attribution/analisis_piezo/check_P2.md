# Comprobación de P2 (balance por grupo físico), estados finales guardados, CPU, 128 puntos por clave

Grupos: mec = constitutivas mecánicas (σxx, σyy, τ); elec = constitutivas D + Gauss; equil = equilibrio; borde = condiciones de borde; carga = carga neta (directo).
Cada celda: mediana sobre semillas y brazos de la mediana por aristas. `cos` = coseno del autovector dominante de B frente al de B balanceado; `dP` = diferencia relativa entre P y P balanceado.

| problema | capa | cuota mec | cuota elec | cuota equil | cuota borde | aristas con elec<5 % | aristas con mec>90 % | cos med | aristas cos<0.9 | dP med | cond efectiva E → balanceado |
|---|---|---|---|---|---|---|---|---|---|---|---|
| converse | 0 | 1.00 | 0.000 | 0.001 | 0.000 | 1.00 | 1.00 | 0.93 | 0.44 | 0.16 | 1 → 14.9 |
| converse | 1 | 0.99 | 0.002 | 0.010 | 0.000 | 0.84 | 0.77 | 0.98 | 0.25 | 0.05 | 1 → 5.82 |
| converse | 2 | 0.00 | 0.000 | 0.000 | 0.000 | 0.62 | 0.25 | 0.99 | 0.40 | 0.13 | 1 → 25.6 |
| direct | 0 | 0.17 | 0.022 | 0.792 | 0.000 | 0.77 | 0.00 | 0.94 | 0.41 | 0.07 | 1 → 2.98 |
| direct | 1 | 0.12 | 0.008 | 0.863 | 0.000 | 0.93 | 0.00 | 1.00 | 0.06 | 0.06 | 1 → 3.82 |
| direct | 2 | 0.00 | 0.074 | 0.000 | 0.000 | 0.41 | 0.16 | 1.00 | 0.10 | 0.02 | 1 → 1.67 |