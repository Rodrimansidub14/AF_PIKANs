# Allen–Cahn, re-corrida con contrato DeepXDE — resultados (2026-09-30)

Kernel `rodrimandub/af-pikans-ac-rerun-20260930`: COMPLETE, sesión 3 869 s (1.07 h), 10/10 corridas con `return_code=0`, todas con los 40 000 pasos de Adam completos. Exploratorio; `scientific_accepted=false`. Datos por semilla: `ac-rerun-per-seed.csv`; artefactos: `outputs/ac/`.

Contrato: lote completo (8 000 dominio + 400 borde + 800 iniciales), float32, Adam 1e-3 × 40 000, luego L-BFGS (tope 300 s / 15 000 it.), puntuado solo contra `Allen_Cahn.mat` (terceros) con la métrica de DeepXDE. Semillas 601–605, mismos pools por semilla en ambos brazos.

| Brazo | Parámetros | Fin de Adam (media ± sd, n=5) | Tras L-BFGS (media ± sd, n=5) | L-BFGS |
|---|---|---|---|---|
| MLP tanh 3×20 (DeepXDE) | 921 | `1.947e-1 ± 9.9e-2` | `1.773e-2 ± 8.7e-3` (0.73–3.0e-2) | 4 paradas por meseta, 1 por tope de 300 s; 9.6k–13.4k it. |
| RBF-PIKAN ancho 16 | 3 648 | `2.838e-2 ± 7.4e-3` | `2.686e-3 ± 9.5e-4` (1.7–3.7e-3) | 5 paradas por meseta; 2.4k–3.7k it. (86–127 s) |

Diferencia pareada RBF − MLP por semilla (601–605):
- fin de Adam: −0.130, −0.104, −0.149, −0.328, −0.122 (media −0.166 ± 0.092);
- tras L-BFGS: −0.0055, −0.0085, −0.0161, −0.0275, −0.0176 (media −0.0150 ± 0.0086).
El RBF tiene menor error en 5/5 semillas en ambos endpoints; tras L-BFGS el cociente medio MLP/RBF es ≈ 6.6.

## Cómo leerlo

- **La receta DeepXDE queda reproducida**: el MLP 3×20 tras L-BFGS da `1.77e-2 ± 8.7e-3`, coherente con el `2.00e-2 ± 6.1e-3` que dio la librería DeepXDE 1.12.2 oficial en Kaggle (misma referencia `.mat`). Esto valida la implementación propia como control.
- **No es una comparación a parámetros iguales**: el RBF tiene 4× los parámetros del MLP de la demo. Sí es la comparación "receta estándar vs baseline RBF actual".
- **Frente a la literatura** (tabla de AEH-PINN, otra red 4×50, 12 800 puntos, hasta 100 000 épocas, referencia distinta): el RBF (2.7e-3) queda en el orden de RAD (1.3e-3) y SelectNet (6.6e-4) y unas 30 veces por encima de AEH-PINN (8.1e-5). Es un contexto, no una comparación pareada.
- Mejora frente al baseline anterior de tiempo fijo (RBF `9.18e-3` tras L-BFGS 240 s): ahora `2.69e-3`. Cambió a la vez lote completo, float32, pasos fijos, y L-BFGS hasta meseta, así que no se puede atribuir a un solo factor.
- Limitaciones: n=5; referencia `.mat` de terceros pero sin convergencia verificada por nosotros (concordancia con las referencias de agentes a 1.6e-4); no hay pre-registro firmado.
