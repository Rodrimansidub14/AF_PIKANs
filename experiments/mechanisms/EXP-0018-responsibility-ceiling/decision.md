# Decisión pendiente — EXP-0018, tras el techo oráculo

Preparado por un agente el 2026-10-20. **No decide el agente.** Detalle y límites en `results.md`.

## Evidencia

- **H-CEIL-1 no se sostiene en P ni en H.** Con el error real (A\*): P deja el error igual (razón mediana 1.03 en Burgers, 1.05 en Allen–Cahn; mejora en 1/5 y 2/5 semillas) y H lo empeora (1.51 y 1.70) y daña el frente (+30 % y +42 %).
- El control con escalas permutadas (N) cumpliría el umbral numérico en Burgers por azar (mediana 0.82, 4/5) y lo incumple en Allen–Cahn (1.37, 0/5): el escalado heterogéneo no es un efecto robusto.
- Regla del spec: se cierra la familia «foco por responsabilidad por base» (estática, β = 1, q = 0.5); la ocupación (brazo O) queda como única señal sin explorar (O: Burgers mediana 0.94, 3/5; Allen–Cahn 1.34, 1/5).

## Cadena de resultados hasta aquí (para decidir con todo a la vista)

| Línea | Resultado |
|---|---|
| Bases (EXP-0015) | ubicar o refinar bases de la capa 0: sin efecto robusto (U41 mejora también Heat: capacidad) |
| Traspaso / asentamiento (EXP-0016) | el mapa de asentamiento localiza el error peor que el residual; el techo del traspaso no se distingue de ruido |
| Batería y atribución (EXP-0017) | ninguna señal interna supera al residual; S1 era artefacto de la restricción dura |
| Techo oráculo del foco por responsabilidad (EXP-0018) | con el error real tampoco hay techo |

Lo que sí es consistente en todo el recorrido: el residual es la mejor localizadora entre las señales sin referencia probadas (Burgers: solape 0.40 frente a 0.12 de la geometría), y el error de la PIKAN-PDE es de optimización más que de capacidad o de localización.

## Opciones

1. **Cerrar la línea «señales internas para focalizar» con el resultado negativo encadenado (EXP-0015 a 0018)** y escribirlo como hallazgo de la tesis. Es el cierre que la evidencia sostiene. Deja abierta la restricción de fijar un mecanismo.
2. **Explorar la ocupación (brazo O) con más semillas.** Evidencia débil (mediana 0.94 y 1.34, opuesta entre problemas): probablemente ruido; solo tendría sentido con un diseño de no inferioridad y potencia calculada.
3. **Redefinir el mecanismo fuera de lo probado** (p. ej. adaptación del propio soporte de las bases, centros o anchos entrenables, o una familia de optimización distinta del refinamiento ponderado). Exige que el investigador decida cómo se compagina con el título fijo del protocolo, porque las señales internas probadas hasta ahora no superan al residual.

- [ ] Decisión del investigador: ______ — fecha: ____
