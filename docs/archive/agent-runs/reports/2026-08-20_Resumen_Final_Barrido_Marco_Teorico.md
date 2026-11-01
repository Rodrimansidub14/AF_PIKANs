# Resumen final — Barrido bibliográfico del marco teórico (2026-08-20)

**Estado**: investigación completa. No se modificó `i-marco_teorico.tex` ni
`m-bibliografia.bib`. Detalle completo del proceso en
`2026-08-20_Informe_Barrido_Marco_Teorico.md` (misma carpeta). Nota de vault
en `../vault_drafts/2026-08-20_Barrido_Marco_Teorico_Foco_Adaptativo.md`,
pendiente de promoción manual a `PIKAN-Vault/PIKAN-Vault` — **el agente no
puede escribir en `PIKAN-Vault/` ni en `artifacts/` en este hilo**: ambas
rutas están bloqueadas a nivel de herramienta para el rol `literature_scout`
(`write_paths = ["var/agent_runs"]` en `context/agent-policy.toml`,
enforced por el host, no solo por política de prompt). La promoción a
`PIKAN-Vault` y cualquier copia a `artifacts/` requieren acción manual del
estudiante, o retomar el hilo bajo un rol/workflow con esa escritura
habilitada.

---

## 1. Qué se investigó

Barrido de SOTA y referencias primarias para las 7 secciones (6.1–6.7) de la
propuesta de reestructuración del marco teórico
(`propuesta_reestructuracion_marco_teorico.md`, externa), que reordena el
capítulo como cadena causal:

```
PDE → aproximación numérica → SciML → limitación actual →
matemática del problema → mecanismo → stress tests
```

## 2. Qué se encontró

- **23 referencias base** verificadas por DOI/arXiv en el primer barrido,
  clasificadas por familia (puntos/residual, pesos/pérdida, dominio/test
  space, representación, validación).
- **14 referencias adicionales** en el segundo barrido dirigido: 6 canónicos
  que faltaban en `m-bibliografia.bib` (Dörfler 1996, Verfürth 2013, Jagtap
  2020, McClenny 2023, Moseley 2023, Cuomo 2022) y 6 candidatos 2026
  (mayoría preprints sin arbitrar), más 2 hallazgos de una ronda de
  adyacencia (AB-PINNs ya arbitrado; Jnini 2024 como precursor de He 2026).
- Cobertura por sección: alta en 6.1, 6.2, 6.3, 6.4 y 6.7; media en 6.5
  (asignación de capacidad) y 6.6 (mecanismo), por el riesgo de
  solapamiento descrito abajo.

## 3. El hallazgo que más importa: riesgo de solapamiento del mecanismo

`i-marco_teorico.tex` deja pendiente (línea 112) el punto de inyección del
mecanismo de focalización (`T` en `w = T(s(·;θ))`). El barrido muestra que
dos de las tres opciones ya están saturadas en la literatura reciente:

- **Reponderación del residual** — SA-PINN (McClenny 2023, arbitrado),
  ReLoBRaLo, y el método híbrido de Chen et al. 2025 (arXiv 2511.05452).
- **Modulación del muestreo** — RAR, RAD/RAR-D, FI-PINN, PINNACLE.

La opción que sigue defendible es **modular la densidad de la grilla
interna de la spline dentro de PIKAN**, porque es la única que explota la
propiedad (soporte local) que justifica elegir PIKAN sobre un MLP. El
precedente más cercano (Rigas et al. 2024, PIKAN de grilla adaptativa) usa
un cronograma de entrenamiento, no una señal derivada del residual —
esa diferencia es el hueco de contribución real de la tesis.

## 4. Conclusión de posicionamiento

El barrido no invalida la pregunta de investigación: el hueco de
"redistribuir capacidad representacional local, guiada por señal de estado
del modelo, sin fragmentar el dominio" sigue abierto en toda la literatura
revisada. Sí invalida la formulación actual del mecanismo tal como está
escrita hoy en el `.tex` (reponderación directa del residual), que se
superpone con trabajo ya publicado. Se recomienda decidir el punto de
inyección antes de continuar redactando la sección 6.6.

## 5. Pendientes explícitos para el estudiante

1. Decidir el punto de inyección del mecanismo (residual vs. muestreo vs.
   grilla interna) — bloquea la redacción de 6.6.
2. Promover manualmente la nota de vault desde `var/agent_runs/vault_drafts/`
   a `PIKAN-Vault/PIKAN-Vault/` (el agente no puede escribir ahí bajo este
   rol; el host lo bloquea aunque haya aprobación explícita en el chat).
3. Decidir si las 14 referencias nuevas se incorporan a
   `m-bibliografia.bib` (no se ha tocado ese archivo en ningún momento de
   este barrido).
4. Si se necesita un resumen físicamente dentro de `artifacts/`, copiarlo
   manualmente desde este archivo, o retomar la tarea bajo un rol con
   `write_paths` que incluya `artifacts` (p. ej. `experiment_runner`).
