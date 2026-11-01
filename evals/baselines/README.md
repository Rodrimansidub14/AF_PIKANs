# baselines/ — de dónde salen los umbrales

Un `threshold` no se inventa. Se congela desde una corrida de referencia
medida con estos mismos graders, sobre esta misma malla held-out.

## Estado actual: baseline_v1_exploratory congelado (ver al final); sin umbrales

Por eso hoy `src/research_pipeline/pipeline.py gate` devuelve
`threshold_not_defined` para todas las métricas y `technical_pass = False`
en cualquier corrida: las cinco métricas de `acceptance_gates` en
`src/harness_loop/experiment_manifest.yaml` declaran `direction` y `required`
pero ninguna declara `threshold`.

No es un bug del código: es la consecuencia correcta de no tener baseline.
El verificador se niega a aprobar en vez de aprobar por omisión.

## Cómo se congela uno

1. Correr Tier 1 sobre `heat_1d` con el modelo de referencia y ambas semillas.
2. Verificar que Tier 0 pasó (los graders son sanos).
3. Escribir `heat_1d@model.pikan.baseline.json` con el scorecard por semilla,
   el commit, el entorno y la fecha.
4. A partir de ahí Tier 1 tiene umbrales y puede fallar de verdad.

Un baseline congelado **no se edita**. Si cambia el modelo de referencia, se
congela uno nuevo con otro id y se anota la supersesión.

## Formato

```json
{
  "baseline_id": "heat_1d@model.pikan.baseline",
  "frozen_at": "YYYY-MM-DD",
  "commit": "<sha>",
  "problem": "problem.heat_1d",
  "model": "model.pikan.baseline",
  "suite": "tier1_regression",
  "seeds": {"11": {"relative_l2_error": 0.0, "...": 0.0},
            "17": {"relative_l2_error": 0.0, "...": 0.0}},
  "aggregate": {"relative_l2_error": {"mean": 0.0, "std": 0.0}},
  "environment": {"python": "3.12", "backend": "local_cpu"}
}
```

## Congelado

- `baseline_v1_exploratory.{json,md}` — 2026-10-03, firmado por el investigador. Exploratorio, sin umbrales derivados. No se edita.

- `baseline_v2_exploratory.{json,md}` — 2026-10-04, firmado por el investigador. Supersede a v1 solo en Burgers y piezo; v1 no se edita. Exploratorio, sin umbrales derivados.
