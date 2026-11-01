# Decisión pendiente — EXP-0022, tras el diagnóstico de condicionamiento

Preparado por un agente el 2026-10-20. **No decide el agente.** Detalle en `results.md`.

## Evidencia

- Hay mal condicionamiento (κ_eff(100) ≈ 2 000–2 700) y el bloque por arista lo reduce 2.9× (Burgers) y 5.9× (Allen–Cahn) más que Jacobi.
- Formalmente fallan H-COND-1 (Allen–Cahn: 0.144 contra 0.10) y H-COND-2 (Burgers contra Jacobi: 0.346 contra 0.33, dentro del ruido).
- Hallazgo descriptivo: en las capas ocultas cada arista usa ~2 bases de 11 y la Gram de la base está 10⁴× peor que con entradas uniformes. Es de la representación entrenada.

## Opciones

1. **Firmar el EXP-0023 rediseñado.** Brazos: Jacobi (comparador principal), bloque por arista, bloque por entrada, Gram fija uniforme (control de «¿importa que sea de las activaciones?»). Burgers como problema principal. Umbrales nuevos, calibrados con el ruido de este diagnóstico. Riesgo: L-BFGS compensa buena parte del condicionamiento, y la ganancia sobre Jacobi es de ~3× en κ.
2. **Explorar primero en el túnel de viento (EXP-0024), sin firma por corrida:** el precondicionamiento por bloque y una idea nueva que sale del hallazgo, reubicar los centros de las capas ocultas con los cuantiles de las activaciones (mecanismo nativo de KAN, sin pérdida ni referencia). Se confirma solo el finalista. Requiere firmar el EXP-0024.
3. **Cerrar la ruta de condicionamiento** con este resultado (mal condicionamiento real pero sin ventaja clara sobre Jacobi).

Recomendación del agente: 2, y que el finalista (o el rediseño de la opción 1) pase a un preregistro de confirmación.

- [x] Decisión del investigador: **opciones 1 y 2** — firmó el EXP-0023 (rediseñado, v2) y el EXP-0024 (túnel de viento) — fecha: 2026-10-20 — chat: «firmo 0023 y 0024»; registrada por el agente.
