# Decisión pendiente — EXP-0017, tras la batería en lazo abierto

Preparado por un agente el 2026-10-20. **No decide el agente.** Detalle y límites en `results.md`.

## Evidencia

- **H-LOC-1 no se sostiene para ninguna señal.** La única que supera al residuo en Spearman (S1, ‖∂u/∂θ‖: +0.71 y +0.64 frente a +0.45 y +0.37) es un **artefacto de la restricción dura**: el factor geométrico (1 − x²)·t, sin red, ya da +0.70 y +0.64; la parte que aporta la red tiene correlación negativa. Entre las demás señales de la red, ninguna localiza el error mejor que el residuo; el solape del 5 % superior en Burgers solo lo logra el residuo (0.40 frente a 0.12 de la geometría).
- **H-ATTR-1 y H-ATTR-2 no se sostienen.** Las bases que más se mueven en el refinamiento son las más usadas (ocupación, Spearman +0.46 y +0.51), no las de mayor responsabilidad (+0.33 y +0.10); incluso con el error real (A*) la correlación es +0.26 y +0.28. La responsabilidad es difusa (20 % superior concentra 0.37 y 0.28; umbral 0.50). Excepción: en Allen–Cahn el error real sí se concentra (0.62) pero el residuo no.
- Línea base geométrica: el Spearman sobre todo el dominio queda dominado por «dónde puede haber error» (restricción dura); el residuo (+0.45) está por debajo del factor geométrico solo (+0.70).

## Qué dice el árbol firmado

Fila «Nada pasa»: se diseña igual la focalización en el espacio de la representación (paso por base escalado por su responsabilidad), pero **primero con su techo oráculo (A* con el error real) en GPU** antes de cualquier versión con residuo.

## Opciones

1. **Seguir la fila firmada: EXP-0018 con A\* oráculo en GPU.** Mide el techo causal (¿mejora el error enfocar el refinamiento en las bases de mayor responsabilidad real?), que la correlación descriptiva no responde. Riesgo: el A* descriptivo ya es peor que la ocupación, y en Burgers el error real tampoco se concentra por base (0.36).
2. **Mecanismo con la ocupación como señal** (lo que mejor predice el movimiento): no localiza error, así que no sería focalización guiada por el error; solo por uso de la representación.
3. **Cerrar la línea de señales internas** con el resultado negativo (EXP-0016 y EXP-0017) y reportar que, entre las señales sin referencia probadas, el residuo es la mejor localizadora cuando se controla la geometría de la restricción dura. Es un resultado reportable, pero deja sin respuesta la restricción de fijar un mecanismo.

- [x] Decisión del investigador: **opción 1** (EXP-0018 con A* oráculo en GPU) — fecha: 2026-10-20 — firmada por Rodrigo Mansilla en el chat ("Firmo opción 1"); registrada por el agente.

**Condiciones acordadas al firmar:** (1) un solo β y un solo q, fijados antes de correr; (2) regla de parada escrita en el spec: si el techo con el error real no supera al baseline, se cierra la familia; (3) si el techo supera al baseline, el siguiente paso es sustituir A* por un estimador sin referencia. Desviación a registrar: S4 no se calculó (la etapa A no guardó la capa 1); el veredicto «ninguna señal pasa» vale para S1, S2, S3 y S5. El análisis de S1 contra el factor de la restricción dura es exploratorio y no corrige el veredicto firmado.
