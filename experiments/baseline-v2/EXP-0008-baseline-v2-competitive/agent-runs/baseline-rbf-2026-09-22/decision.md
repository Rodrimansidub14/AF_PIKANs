# Primer baseline exploratorio RBF — propuesta, no corrida ejecutada

P1–P4 confirmados por el investigador el 2026-09-22. P5 excluido.
El investigador delegó la elección de variante, solicitando respaldo reciente.

## Decisión de arquitectura

RBF gaussianas con geometría fija como control inicial. Esto permite observar
señales sin mezclar desde el comienzo aprendizaje de pesos y adaptación de
geometría. Es una inferencia de diseño para esta pregunta, no superioridad
empírica demostrada. La siguiente comparación relevante es geometría aprendible.

- FastKAN (2024), repositorio original: https://github.com/ZiyaoLi/fast-kan
- Free-RBF-KAN (enero de 2026): https://arxiv.org/abs/2601.07760
  Separa RBF fija de centros/anchos aprendibles y reporta mejoras en sus casos.
  Sus experimentos no prueban superioridad en nuestra campaña P1–P4.
- Hierarchical RBF-KAN/RBF-SKAN (junio de 2026): https://arxiv.org/abs/2606.02936
  Es una ruta reciente sobre aproximación y campos aleatorios; no basta para
  reemplazar nuestro control de PDE sin evaluación específica.

La adaptación propuesta conserva tanh y escalado físico de la metodología del
proyecto; no se denomina reproducción exacta de FastKAN o Free-RBF-KAN.
No se aplicará LayerNorm a las dos coordenadas físicas de entrada.
Once gaussianas más un coeficiente SiLU por arista dan 3648 parámetros por
rama [2,16,16,1], comparable en número al sustrato spline actual. Igual número
de parámetros no implica igual costo ni igual capacidad expresiva.

PINA admite modelos torch.nn.Module personalizados:
https://pina-org.github.io/PINA/tutorial1/tutorial.html
Su RBFBlock es un interpolador no entrenable, no una capa RBF-KAN:
https://pina-org.github.io/PINA/_rst/model/block/rbf_block.html

## Qué queremos obtener

Checkpoints, curvas de entrenamiento, residuales normalizados, cobertura de
bases, saturación, sensibilidad física y costo. Las señales solo reciben
coordenadas, modelo y física. Las referencias y máscaras de interfaces quedan
fuera del diagnóstico. Sus relaciones con error se estudian después en V;
E permanece cerrado. Una correlación exploratoria no establece causalidad.

La arquitectura fija puede seguir cambiando su sensibilidad espacial al entrenar
pesos. Las gaussianas tampoco tienen soporte compacto: se medirá localización
efectiva en vez de asumirla por el nombre de la base.

## Orden y presupuesto propuestos

Pruebas de operadores/condiciones y derivadas; integración PINA; prueba breve
de artefactos; autenticación; envío privado; primera pasada P1, P2, P3, P4.
Semilla 101, máximo 2000 pasos o 15 minutos de entrenamiento/diagnóstico por
problema, lo que ocurra primero. Timeout exterior 20 minutos por corrida para
evaluación/guardado. Máximo agregado 80 minutos de ejecución de las cuatro
corridas; instalación, cola y referencias independientes se contabilizan aparte.
No ejecutar referencias FEM costosas automáticamente. Este recorte es un
prepiloto exploratorio y no sustituye los 20000 pasos y tres semillas del piloto
canónico. Semillas 102/103 se deciden después de medir costo y estabilidad.

P3/P4 pueden producir diagnósticos de entrenamiento con estado de referencia
pendiente; no se publicarán errores de campo ni claims de precisión sin sus
referencias convergidas. Los valores DFG publicados no se registrarán como
resultados propios. Ningún gate científico se considera aprobado por defecto.

## Configuración y autenticación pendientes

Kaggle ya está autorizado para notebooks privados y campañas aprobadas.
La autorización no equivale a credenciales disponibles ni a campaña enviada.
W&B online requiere extender la política vigente que solo enumera Kaggle,
limitado a un proyecto privado de esta campaña y sus métricas/artefactos.
Preparación offline no necesita API key.

Solicitar usuario Kaggle, entidad/proyecto W&B (datos no secretos).
Ingresar KAGGLE_API_TOKEN y WANDB_API_KEY mediante un mecanismo local protegido
o secretos de Kaggle, nunca en mensajes, código, manifiestos ni outputs.
W&B permite WANDB_MODE=offline y WANDB_API_KEY por entorno:
https://docs.wandb.ai/models/track/environment-variables

El entorno remoto de transporte está instalado; aún faltan el entrenador,
dependencias científicas verificadas, callbacks diagnósticos, pruebas de física,
notebooks definitivos y autenticación. No se ha iniciado entrenamiento.
