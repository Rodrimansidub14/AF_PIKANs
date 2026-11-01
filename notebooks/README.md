# Notebooks de investigación

Los notebooks son interfaces reproducibles para explorar y ejecutar experimentos. La definición de la hipótesis, los gates y la procedencia viven en el manifiesto y en el harness; el notebook no debe depender de estado oculto.

## Roles

| Rol | Uso |
|---|---|
| `exploration` | Prueba rápida o exploración que todavía no produce evidencia final. |
| `baseline` | Referencia reproducible antes de cambiar el mecanismo. |
| `candidate` | Variante acotada comparada contra un baseline. |
| `analysis` | Comparación de corridas y preparación de evidencia. |
| `reporting` | Figuras, tablas o borradores basados en evidencia aceptada. |

## Metadatos

Los notebooks nuevos deben incluir este bloque en `metadata.research`:

```json
{
  "notebook_id": "nb.heat.baseline.v1",
  "role": "baseline",
  "problem_id": "problem.heat_1d",
  "model_id": "model.pikan.baseline",
  "hypothesis_id": "hyp.canon.baseline",
  "source_of_truth": "manifest"
}
```

Los notebooks de exploración solo requieren `notebook_id` y `role`. La auditoría inicial puede marcar notebooks históricos como `unclassified`; no los modifica.

## Salidas experimentales

Un notebook `baseline` o `candidate` debe poder producir mediante el harness:

- `metrics.json`;
- `run_summary.json`;
- `decision_note.md`;
- mapas o curvas necesarios para la hipótesis.

El primer flujo que se normalizará es el baseline de Heat 1D. Las variantes históricas de Heat y Navier–Stokes se conservan hasta que cada una tenga un rol y una procedencia declarados.
