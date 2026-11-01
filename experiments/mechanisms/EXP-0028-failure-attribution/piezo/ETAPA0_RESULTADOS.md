# Piezo, etapa 0: mediciones en T4 (2026-10-25) — fuente: results/piezo-stage0/stage0/stage0_results.json
Kernel `rodrimandub/af-pikans-exp-0028-piezo-stage0`, 2×T4, float64, 150 pasos por medición. Sesión de 1 128 s.

| Problema | Brazo | s/paso | recálculo (s) | pico de memoria | utilización GPU0 |
|---|---|---|---|---|---|
| converso | B | 0.436 | — | 2.4 GB | 70 % |
| converso | E | 0.445 | 14.4 | 7.9 GB | 81 % |
| converso | J | 0.448 | 14.1 | 7.8 GB | 81 % |
| directo | B | 0.323 | — | 1.7 GB | 77 % |
| directo | E | 0.342 | 15.5 | 9.4 GB | 77 % |
| directo | J | 0.338 | 15.1 | 9.4 GB | 77 % |

**Concurrencia (converso B):** 1 proceso 2.29 pasos/s; 2 procesos en la misma GPU 2.46; 3 procesos 2.47 (sin ganancia: la GPU ya está saturada, util 79 → 97 %); 2 procesos en cada una de las dos GPU: 4.92 pasos/s (el doble que una GPU).
**CUDA graphs:** 1.08× (converso) y 1.12× (directo); la pérdida final coincide a 1e-16. Ganancia irrelevante.
**Lectura:** el límite es el cálculo (float64 en T4), no el lanzamiento de kernels. Más procesos por GPU, CUDA graphs y vmap sobre semillas no ayudan. Solo ayudan: usar las dos GPU (2×) y una GPU con mejor float64.
**Cuota:** la cuota subió ≈ 0.33 h para una sesión de ≈ 0.35 h usando las dos GPU en parte del tiempo: se cuenta por sesión, no por GPU (medición burda: cuota 21.77 → 22.10 h).

## Costo medido por corrida (20 000 pasos; recálculo cada 1 000 = 20 recálculos)
converso B 2.42 h, F ≈ 2.55 h, E 2.55 h, J 2.55 h; directo B 1.80 h, F ≈ 1.9 h, E 1.97 h, J 1.97 h (T4 float64).
Con dos GPU en paralelo: 1 semilla de B, F, E, J en ambos problemas = 17.8 GPU-h = 8.9 h de sesión; 3 semillas = 26.7 h; 5 semillas = 44.5 h de sesión (89 GPU-h).
