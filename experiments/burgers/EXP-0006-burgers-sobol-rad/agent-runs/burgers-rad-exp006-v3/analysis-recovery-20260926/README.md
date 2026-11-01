# EXP-0006 · recuperación de artefactos

La transferencia de salida de Kaggle agotó el límite de 1200 s. Los archivos locales contienen cuatro jobs RAD completos (1201–1204) y artefactos parciales del job RAD 1205. La salida no contiene archivos de las cinco tareas Sobol, aunque `campaign.json` dentro del paquete incompleto informa que terminaron. Por ello no se calcularon pares ni una métrica primaria.

- Jobs completos recuperados: **4 RAD**.
- Jobs Sobol recuperados: **0 de 5**.
- Parejas comparables: **0 de 5**.
- Estado científico: `scientific_accepted=false`.
- La cuenta Kaggle configurada localmente aún muestra el identificador anterior; los artefactos parciales se preservan, pero no bastan para cerrar el contraste.

## Archivos

- `completed-rad-runs.csv`
- `quantitative-signals-recovered.csv`
- `quantitative-signals-with-units-recovered.csv`
- `rad-adaptation-signals.csv`
- `figures/rad-effective-candidates.html`
- `figures/rad-refresh-cost.html`
- `figures/rad-unique-points.html`
- `figures/training-internal-signals.html`
- `figures/training-loss-curves.html`
- `figures/validation-fields-burgers-seed-1201.html`
- `figures/validation-fields-burgers-seed-1202.html`
- `figures/validation-fields-burgers-seed-1203.html`
- `figures/validation-fields-burgers-seed-1204.html`
