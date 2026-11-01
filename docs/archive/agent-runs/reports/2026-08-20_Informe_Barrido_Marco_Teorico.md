# Informe general — Barrido bibliográfico del marco teórico (2026-08-20)

**Alcance**: investigación pura. No se modificó `i-marco_teorico.tex`,
`m-bibliografia.bib`, `PIKAN-Vault/PIKAN-Vault`, ni la propuesta externa de
reestructuración. Ubicación de este informe: `var/agent_runs/reports/`
(único `write_path` permitido para el rol `literature_scout` activo en este
hilo, según `context/agent-policy.toml`). El borrador de nota para el vault
vive en `var/agent_runs/vault_drafts/` y queda pendiente de promoción manual.

---

## 1. Objetivo y disparador

Barrido de research automation motivado por
`propuesta_reestructuracion_marco_teorico.md` (externa), que reordena el
capítulo como cadena causal:

```
PDE → aproximación numérica → SciML → limitación actual →
matemática del problema → mecanismo → stress tests
```

en vez de catálogo de métodos, evitando llegar a PIKAN como "la solución"
antes de tiempo.

## 2. Método

- Lectura del material preliminar (`i-marco_teorico.tex`, 257 líneas,
  `m-bibliografia.bib`, 366 líneas / ~37 claves) y de la propuesta externa.
- Matriz de preguntas y términos de búsqueda por sección 6.1–6.7.
- Búsqueda web (arXiv, Crossref, DuckDuckGO) con verificación cruzada de
  metadatos (autor, año, DOI, journal) para cada fuente candidata.
- Mapeo de fuentes verificadas a secciones y detección de vacíos.
- Segundo barrido dirigido a partir de preguntas del usuario: cobertura 2026
  y cobertura de canónicos faltantes (12 referencias adicionales), más
  barrido de adyacencia a partir de esas 12.

## 3. Estado de cobertura por sección

| Sección | Cobertura | Vacíos identificados |
|---|---|---|
| 6.1 PDE y aproximación | Alta tras este barrido | Resuelto: Dörfler 1996, Verfürth 2013 (cierran los dos `\pendiente` de numérica clásica) |
| 6.2 Numérica → SciML | Alta | Cuomo 2022 como complemento de Karniadakis 2021 |
| 6.3 Estado del arte | Alta | Bien cubierta: familias puntos/pesos/dominio/representación con >15 referencias primarias |
| 6.4 Estructura matemática de la dificultad | Alta | Reforzada con He et al. 2026 (spectral, no arbitrado) |
| 6.5 Asignación de capacidad | Media — sigue siendo la más débil | Falta literatura que defina "esfuerzo de aprendizaje ∝ dificultad local" como objeto formal; probablemente es aporte propio de la tesis, no vacío bibliográfico |
| 6.6 Mecanismo | Media-alta, con riesgo de solapamiento | Ver sección 5 (decisión de diseño) |
| 6.7 Pruebas de estrés | Alta | Referencias analíticas y de benchmark cubiertas para las 5 PDE del banco |

## 4. Catálogo consolidado de referencias nuevas (14)

### Canónicos faltantes (6) — arbitrados, alta confianza
1. Dörfler (1996) — DOI `10.1137/0733054`
2. Jagtap, Kawaguchi & Karniadakis (2020) — DOI `10.1016/j.jcp.2019.109136`
3. McClenny & Braga-Neto (2023) — DOI `10.1016/j.jcp.2022.111722`
4. Moseley, Markham & Nissen-Meyer (2023) — DOI `10.1007/s10444-023-10065-9`
5. Cuomo et al. (2022) — DOI `10.1007/s10915-022-01939-z`
6. Verfürth (2013) — libro OUP, ISBN 9780199679423

### Candidatos 2026 (6) — mixtos, mayoría sin arbitraje
7. Dzimah et al. (2026) — arXiv 2602.15068 (no arbitrado)
8. He, Liang, Zhao & Zhong (2026) — arXiv 2603.00904 (no arbitrado)
9. Chen, Howard & Stinis (2025) — arXiv 2511.05452 (no arbitrado)
10. Duanmu et al. (2026) — DOI `10.1016/j.engappai.2026.115798` (arbitrado)
11. Georgiou et al. (2026) — arXiv 2608.07255 (no arbitrado)
12. WCCM-ECCOMAS 2026 — resumen de congreso, sin DOI (solo nota de tendencia)

### Hallazgos de la ronda de adyacencia (2)
13. Botvinick-Greenhouse et al. (2025), AB-PINNs — ahora arbitrado, DOI
    `10.1088/2632-2153/ae8638` (antes solo preprint arXiv 2510.08924)
14. Jnini, Vella & Zeinhofer (2024) — arXiv 2402.00531, precursor arbitrado
    del enfoque de He et al. 2026

## 5. Decisión de diseño: punto de inyección del mecanismo

El `.tex` deja pendiente (línea 112) si `T` se inyecta sobre el residual, el
muestreo, o la densidad de grilla interna de la spline. Hallazgo central del
barrido:

- **Reponderación del residual** y **modulación del muestreo** están
  saturados en la literatura reciente (SA-PINN, ReLoBRaLo, RAR/RAD/RAR-D,
  FI-PINN, PINNACLE, y ahora el método híbrido de Chen et al. 2025). Si el
  mecanismo final usa cualquiera de estos dos puntos de inyección, se
  superpone con trabajo ya publicado y debilita el claim de originalidad.
- **Modulación de la densidad de grilla interna (PIKAN)** es el único punto
  de inyección que explota la propiedad que justifica elegir PIKAN sobre un
  MLP (soporte local de la base spline). El precedente más cercano —
  Rigas et al. (2024), grid-dependent PIKAN training — adapta la grilla por
  cronograma de entrenamiento, no por una señal derivada del estado del
  modelo/residual. Ese contraste (señal-driven vs. cronograma) es el hueco
  de contribución real identificado por este barrido.
- Riesgo a mitigar explícitamente en el texto: diferenciar frente a
  AB-PINNs (ya arbitrado), que redistribuye capacidad a nivel de
  subdominios (macro) y no a nivel de nodos de control dentro de una sola
  red (micro).

**Recomendación**: tomar la densidad de grilla interna como punto de
inyección de `T`, documentando el análisis de costo por época
(`\pendiente` de línea 128-130) como parte crítica del diseño, no como nota
opcional.

## 6. Estado del repositorio

Verificado con `git status --short` al cierre del barrido original: sin
cambios sobre el estado preexistente en los archivos canónicos del capítulo.
Este informe y el borrador de nota de vault son las únicas escrituras
nuevas de esta sesión, ambas dentro de `var/agent_runs/`, que es el
`write_path` autorizado para el rol activo.

## 7. Pendientes para el humano

- Revisar y, si corresponde, promover
  `var/agent_runs/vault_drafts/2026-08-20_Barrido_Marco_Teorico_Foco_Adaptativo.md`
  a `PIKAN-Vault/PIKAN-Vault/` (requiere aprobación humana según
  `context/agent-policy.toml`, `canonical_contract_write =
  "human_approval_required"`).
- Decidir el punto de inyección del mecanismo (sección 5) antes de tocar
  `i-marco_teorico.tex`.
- Decidir si las 14 referencias nuevas se incorporan a `m-bibliografia.bib`
  (acción de escritura fuera del alcance de este barrido, pendiente de
  instrucción explícita).
