---
kind: chapter6_attack_plan
status: ready_for_execution
owner: researcher
adviser_gate: required
---

# Capítulo 6 — Attack plan

Este archivo define el orden de trabajo y los gates. La redacción científica la hace el investigador; este plan no decide por el asesor ni rellena los párrafos de la propuesta.

## Fase 0 — Congelar el terreno

1. Trabajar en la nota 60_Marco_Teorico y conservar el módulo LaTeX original.
2. Mantener el Capítulo 1 separado para localizar duplicaciones, sin editarlo
   hasta tener el nuevo esqueleto.
3. Mantener la bibliografía como autoridad de claves y registrar toda cita que
   falte como source_needed.
4. No marcar ninguna sección como lista para promoción mientras haya una
   decisión científica abierta.

## Fase 1 — Cerrar primero las decisiones río arriba

Llevar al asesor cuatro fichas breves, con la pregunta y las consecuencias
textuales y experimentales:

1. DEC-001: escoger el punto de inyección A, B o C.
2. Confirmar en la fuente primaria de KAN si existe grid extension y si es
   uniforme o programada antes de considerar C.
3. DEC-002: confirmar tesis de mecanismo frente a tesis de aplicación
   piezoeléctrica.
4. DEC-003: fijar qué información adicional aporta la señal respecto del
   residual.
5. DEC-004: fijar la convención del residual y sostenerla en todo el capítulo.

Hasta cerrar DEC-001, usar lenguaje de esfuerzo de aprendizaje y no afirmar
capacidad local como si ya estuviera demostrada.

## Fase 2 — Rehacer el esqueleto, sin pulir prosa

Construir los encabezados 6.1–6.7 con una entrada y una salida por sección:

- 6.1: formulación PDE, dominio, operador, forzamiento, condiciones y
  aproximación.
- 6.2: FDM/FVM/FEM y precedente adaptativo AMR, hp y a posteriori.
- 6.3: PINN, MLP, KAN y PIKAN sobre el eje qué redistribuye cada estrategia.
- 6.4: oscilación, gradiente localizado, curvatura, no linealidad, interfaz
  móvil y acoplamiento; sin nombrar benchmarks.
- 6.5: señalar la fila vacía y formular la pregunta de investigación.
- 6.6: definir s → T → w, tipos, inyección, reducciones, costo y PIKAN como
  sustrato de medición.
- 6.7: ordenar los casos por ausente, estática, móvil, anclada y entre campos;
  cerrar con predicciones falsables.

## Fase 3 — Redacción por bloques

Redactar cada sección como unidad independiente y comprobar antes de pasar a
la siguiente:

1. ¿Definí los objetos de la sección en el primer uso?
2. ¿Incluí el contexto y las referencias que pide la anotación correspondiente?
3. ¿La sección tiene una función argumental única?
4. ¿El cierre entrega explícitamente el siguiente eslabón?
5. ¿Estoy repitiendo en 6.5 algo que ya está en 6.3?

## Fase 4 — Evidencia y diagramas

Completar el mapa de citas y después los cinco briefs de figura:

1. malla FEM adaptativa;
2. soporte global de MLP frente a soporte local de B-spline;
3. descomposición de la pérdida PINN;
4. transformación s → T → w;
5. escalera de pruebas y relación con las firmas de 6.4.

Las figuras sirven para resolver las solicitudes de contexto de GAB sin
convertir 6.1–6.3 en un curso de historia de IA.

## Fase 5 — Revisión y gate de promoción

Antes de pedir revisión del asesor:

- comprobar que las 29 anotaciones tienen respuesta o estado explícito;
- revisar citas, siglas, etiquetas y referencias;
- verificar el presupuesto de páginas;
- comprobar que 6.4 no nombra benchmarks y 6.7 sí los instancia;
- escribir el diff LaTeX en staging, nunca sobre el archivo activo;
- compilar y revisar visualmente el PDF solo después de la aprobación humana.

## Orden de ataque resumido

DEC-001 y DEC-002 → esqueleto 6.1–6.7 → DEC-003 y DEC-004 →
redacción del investigador → citas y figuras → revisión de GAB/asesor →
staged LaTeX → promoción explícita.
