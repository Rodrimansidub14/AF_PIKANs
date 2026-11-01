# Ampliación autorizada del verificador exploratorio

El investigador autorizó preparar la ampliación del verificador para revisión,
con Poisson pendiente. Alcance activo: Burgers, Allen–Cahn, DFG Re20.
Esta autorización permite implementar estos contratos y meta-evals; no firma
un experimento, no aprueba gasto GPU ni congela resultados científicos.

## Diseño e implementación

Conservar Heat y las suites previas. Añadir tres módulos NumPy de operadores,
trazas y mallas deterministas, sus problem.yaml y una suite exploratoria aparte.
Burgers usa la onda viajera exacta del protocolo, no el caso sinusoidal de
otros benchmarks. Allen–Cahn usa Dirichlet según DeepXDE 1.12.2. DFG usa salida
natural para el operador laplaciano, no cavidad ni tracción simetrizada.
Las reglas y constantes físicas se conservan de la metodología acordada.

Los nuevos graders rechazan arrays parcialmente no finitos y formas distintas.
Las métricas de campo de Allen–Cahn/DFG quedan bloqueadas incluso si alguien
entrega un array de referencia. Para desbloquearlas se necesitará una extensión
revisada que cargue referencias verificadas con procedencia y convergencia.
No se cambian los graders previos ni se inventan thresholds.

Orden: meta-evals fallando por módulos ausentes; operadores y graders; contratos
y suite; validación de consistencia; pruebas de regresión; revisión independiente.

## Límites

La malla fija no garantiza por sí sola independencia del entrenamiento: el runner
deberá asegurar separación de pools. Los operadores reciben derivadas NumPy;
no son todavía un evaluador que calcula autodiferenciación desde un checkpoint.
No se aceptarán derivadas declaradas por el candidato como prueba de precisión.
La malla tensorial proporciona diagnósticos discretos; claims de error integral
requieren pesos y estudio de refinamiento. DFG necesitará cuadratura apropiada a
la frontera curva y gradientes para fuerzas. Nada de esto se presenta como ya hecho.

Poisson permanece en deferred_problems. Heat sigue como prueba de regresión de
infraestructura y no reemplaza los tres problemas de la campaña.

## Visualización y siguientes etapas

Plotly: curvas, histogramas y comparación interactiva de señales.
PyVista: campos y geometría, exportación sin ventana donde el backend lo admita.
PINA/PyTorch: entrenador y derivadas. W&B: trazabilidad, junto a archivos locales.
Panel es posterior al primer run. No se añaden motores de partículas/bioimaging
sin una necesidad del problema. Los métodos posteriores se justificarán con
literatura y controles; instalar una librería no establece un resultado SOTA.

Fuentes: metodología del proyecto; documentación oficial DeepXDE 1.12.2
https://deepxde.readthedocs.io/en/v1.12.2/demos/pinn_forward/allen.cahn.html
