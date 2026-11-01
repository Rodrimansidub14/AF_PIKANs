# Claims candidatos para `registry/claims.jsonl` (2026-10-26)

**Aceptados por el investigador el 2026-10-26 («firmo todo») y transcritos a `registry/claims.jsonl`** (campo `recorded_by` lo indica). Cada claim trae su evidencia y el límite con que debe citarse. Corrección al aceptar: C3 se apoya en 9 estados finales del directo (B, E, J; semillas 28 a 30).

| # | Candidato | Evidencia (corrida → artefacto) | Límite obligatorio | Sección de la tesis |
|---|---|---|---|---|
| C1 | En el converso de piezo, E sale del llano inicial antes que B en las 5 semillas (mediana 5 500 frente a 7 000 pasos). | `EXP28/results/piezo-modal-run/`, `piezo-s1a/`, `piezo-exp0029-final/` → `LECCIONES_EXP0029.md` | Hallazgo **post hoc**, no predicción; hardware mixto; n = 5 | Resultados, piezo |
| C2 | E recorta la cola mala del converso: error robusto (mediana de `l2re_u`, pasos 18 000 a 20 000) E 0.033 a 0.102 frente a B 0.039 a 0.391. | mismos JSON; cálculo en `EXP-0030/results/A_B_E_resultados.md` | Métrica robusta **definida después de ver los datos**; con el error del último paso B y E quedan iguales (0.152 y 0.158) | Resultados, piezo |
| C3 | El voltaje flotante mal ajustado no explica el error del problema directo (V̂_f ya es el minimizador exacto). | `EXP28/analisis_piezo/check_Vf.md` | 9 estados finales del directo (B, E, J; semillas 28 a 30) | Discusión |
| C4 | En piezo, F es el peor brazo y E el mejor: contradice la predicción firmada («mejora modesta» para ambos). | `EXP28/results/piezo_vs_predictions.md` | n = 3 por objetivo en esa tabla; sin p-valores | Resultados, predicciones |
| C5 | En el converso la parte eléctrica casi no pesa en el ajuste (≈ 0.015 % de la traza en un estado). | `EXP28/analisis_piezo/por_termino.py`; `EXP-0030/results/D_fisica.md` | Un estado, una semilla; las cuotas dependen de cómo se agregue (`RESUMEN.md`, matices) | Discusión |
| C6 | La versión final M no cumple lo firmado en el converso (no conserva la ventaja de E). | `EXP28/results/EXP0029_vs_predicciones.md` | **No citar como «las patas dañan»:** M solo en A100, sin par M/E en el mismo hardware (`EXP-0030/results/A2_mismo_hardware.md`) | Resultados; Discusión (como límite) |
| C7 | El efecto de E en el banco es dependiente de la semilla en Burgers y Fisher–KPP, y se diluye en NS a receta completa. | `EXP28/results/phase1*_scored.md`, `phase1e_all_scored.md` | Receta corta en parte del banco; Burgers completo con n = 2 | Resultados, banco |

Dejados fuera a propósito: cualquier enunciado de que el mecanismo «mejora el error promedio» (no se sostiene con n = 5) y cualquier atribución de efecto a las patas de M.

## Formato de línea (una por claim, en `registry/claims.jsonl`)

```json
{"claim_id": "C1", "text": "…", "run_ids": ["exp0028-piezo-modal-run", "exp0029-piezo-final-20261026"], "artifacts": ["experiments/mechanisms/EXP-0028-failure-attribution/LECCIONES_EXP0029.md"], "thesis_section": "j2-resultados.tex#piezo", "limits": "post hoc; hardware mixto; n = 5", "accepted_by": "Rodrigo Mansilla", "accepted_on": "AAAA-MM-DD"}
```

Los `run_id` salen de `registry/runs.jsonl`. El campo `thesis_section` se llena cuando exista el capítulo en `ProyectoGrad`.
