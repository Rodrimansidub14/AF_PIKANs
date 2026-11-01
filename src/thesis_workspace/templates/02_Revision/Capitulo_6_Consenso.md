---
kind: chapter6_consensus
status: working_consensus
sources:
  - attachment:ddc84932-50b9-4781-8c94-8127068f25a9
  - attachment:62e9822a-474b-46a4-a1a1-ee908ece61bf
  - attachment:70be24e7-a321-43b3-a866-1ef81eeb39aa
---

# Capítulo 6 — Consenso de revisión

## Lectura compartida

La revisión de GAB y la revisión estructural no compiten: trabajan en capas
distintas. GAB revisa la pedagogía, las definiciones, el contexto y las
referencias. La revisión estructural revisa la argumentación, el hueco lógico
y la objeción que cierra cada sección. El plan debe satisfacer ambas capas.

El consenso de trabajo es que el capítulo presenta una tesis de mecanismo, no
una tesis de aplicación. El objeto no son las PDEs por sí mismas, sino cómo se
construye y distribuye la capacidad o el esfuerzo de aproximación de sus
soluciones.

## Cómo se resuelve la tensión entre las revisiones

Cada subsección será autocontenida en sentido expositivo: definirá sus propios
objetos y no exigirá volver atrás para saber de qué trata. A la vez, conservará
la dependencia argumental del capítulo. El dispositivo operativo de entrada y cierre es:

- un párrafo de entrada que ubica el objeto y recuerda el contexto mínimo;
- un desarrollo con las definiciones y referencias necesarias;
- una frase o párrafo de cierre que entrega el eslabón a la siguiente sección.

Esto responde a la exigencia de GAB de que el lector pueda saltar a una
subsección y también conserva la cadena de eliminación del argumento.

## Correcciones que forman parte del consenso

1. Restaurar en 6.2 el precedente clásico: FEM, refinamiento adaptativo de
   malla, refinamiento hp y estimadores de error a posteriori.
2. Presentar 6.3 sobre un eje único: qué redistribuye cada estrategia. 6.5
   debe señalar la fila vacía y formular la pregunta, no repetir el estado del
   arte.
3. Escribir 6.4 como espacio de propiedades matemáticas, sin nombrar los
   problemas concretos del banco. 6.7 hará la instanciación.
4. Mantener PIKAN como objeto del estado del arte en 6.3 y justificarla como
   sustrato de medición al cierre de 6.6.
5. Introducir cada sigla y objeto en su primer uso, con definiciones mínimas,
   tablas y diagramas cuando reduzcan prosa introductoria.
6. Mantener un límite de páginas: 6.1–6.3 hasta 35 %, 6.4–6.5 cerca de 20 %,
   6.6 cerca de 20 % y 6.7 cerca de 25 %.

## Contrato de la cadena 6.1–6.7

| Sección | Función | Objeción que cierra |
| --- | --- | --- |
| 6.1 | PDE como problema de aproximación y objeto de estudio | ¿De qué trata exactamente? |
| 6.2 | Discretización clásica y adaptividad | ¿Es una ocurrencia nueva? |
| 6.3 | Estado del arte sobre el eje qué redistribuye | ¿Esto ya existe? |
| 6.4 | Firma matemática de la dificultad | ¿Qué significa dificultad de forma medible? |
| 6.5 | Fila vacía y pregunta de asignación local | ¿Cuál es el hueco concreto? |
| 6.6 | Mecanismo s → T → w y casos de reducción | ¿Es RAR con otro nombre? |
| 6.7 | Banco de estrés y predicciones falsables | ¿La mejora es solo tasa de aprendizaje? |

## Puntos que no se deben perder

- El precedente clásico adaptativo es una defensa conceptual, no un adorno.
- El Cuadro 1 se ordena por qué redistribuye cada estrategia y debe tener
  referencias suficientes.
- 6.4 separa propiedad del problema, señal observable y efecto representacional.
- Las reducciones de 6.6.4 deben preceder al banco de 6.7.
- Poisson oscilatorio queda como control negativo.
- La piezoelectricidad puede cerrar el banco como caso de acoplamiento, pero
  no convierte automáticamente la tesis en una tesis de aplicación.
- La ausencia de anotaciones en las páginas PDF 22–24 no se interpreta como
  aprobación.

## Estado

Este documento fija el consenso de trabajo, no sustituye la aprobación del
asesor ni redacta el marco teórico.
