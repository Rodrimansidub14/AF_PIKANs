# Protocolo de referencias numéricas: Allen–Cahn BDF y DFG Taylor–Hood

**Estado:** especificación previa a ejecución. No se han ejecutado solves, mallas, benchmarks, tests ni verificaciones. Las reglas y presupuestos quedan fijados antes de ver resultados; cualquier cambio posterior debe quedar versionado y justificarse, no usarse para relajar el criterio después de inspeccionar datos.

El protocolo genera dos referencias independientes para evaluar PINNs: (1) Allen–Cahn mediante líneas de método de líneas y un integrador BDF implícito, y (2) el flujo estacionario DFG 2D-1 con elementos Taylor–Hood P2/P1 en una familia de mallas sucesivamente refinadas. Las referencias deben mantenerse separadas de los puntos de entrenamiento.

## Fuentes primarias y contratos

| Fuente | Uso en este protocolo |
|---|---|
| [DeepXDE, demo oficial de Allen–Cahn](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html) | Fija el target comparable: `u_t = d u_xx + 5(u-u^3)`, `d=.001`, `x∈[-1,1]`, `t∈[0,1]`, `u(x,0)=x²cos(πx)`, `u(±1,t)=-1`; usa una tabla `.mat` a `dx=dt=.01`, cuya generación/convergencia no documenta la página. La tabla es comprobación auxiliar, no la única verdad. |
| [SciPy, clase BDF oficial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html) | Define el integrador BDF implícito de orden variable 1–5, opción de Jacobiano disperso y tolerancias adaptativas. Se usará una versión de SciPy fijada en el lockfile. |
| [FEATFLOW/TU Dortmund, DFG 2D-1 Re20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html) | Fuente de geometría, condiciones, esfuerzo/fuerzas, presión diferencial y QoIs espectrales de referencia. |
| [FEniCS Hands-on, Navier–Stokes y cilindro](https://fenics-handson.readthedocs.io/en/latest/navierstokes/doc.html) | Ejemplo público del caso, la forma débil Taylor–Hood P2/P1 y exportación de campos. Sus parámetros de malla tutorial no bastan por sí solos para certificar convergencia. |
| [jNO, tutorial FEM DFG 2D-1](https://fhg-iisb.github.io/jNO/tutorials/08-fem-and-varpinns/navier-stokes-cylinder-dfg/) | Segundo punto de reproducción P2/P1 y advertencias sobre fuerza de reacción, outlet natural, sensibilidad del lift y refinamiento. Reporta 10,038 DOFs y una ejecución de ~5 s en su propio entorno; ese tiempo no se extrapola a otro hardware. |

Contrato DFG derivado de FEATFLOW: canal `[0,2.2]×[0,.41]` menos círculo de radio `.05` y centro `(.2,.2)`; `ρ=1`, `ν=.001`; pared superior, inferior y cilindro sin deslizamiento; perfil de entrada `u_x(y)=4 Umax y(H-y)/H²`, `u_y=0`, con `Umax=.3`, `H=.41`; salida do-nothing/tracción natural. `Umean=.2`, diámetro `.1`, `Re=20`. Valores FEATFLOW: `Cd=5.57953523384`, `Cl=.010618948146`, `Δp=p(.15,.2)-p(.25,.2)=.11752016697`.

## Allen–Cahn: método de líneas con BDF

### Discretización fijada

1. Usar doble precisión y malla uniforme inclusiva en `[-1,1]` con `N` intervalos y `N+1` nodos, extremos prescritos exactamente a `-1`. Resolver solo por los `N-1` grados interiores. La malla inicial es `u_i(0)=x_i²cos(πx_i)`; es compatible con los valores de frontera en ambos extremos.
2. Aplicar segunda diferencia central de orden 2: `u_xx,i ≈ (u_(i-1)-2u_i+u_(i+1))/h²`; incluir los valores Dirichlet fijos al ensamblar las filas de los nodos interiores adyacentes a los extremos. El RHS semidiscreto es `d L_h u + 5(u-u³)`.
3. Integrar con `scipy.integrate.solve_ivp(method="BDF")`, `rtol=1e-10`, `atol=1e-12`, `max_step=.005`; proporcionar Jacobiano analítico disperso tridiagonal `J=dL_h+diag(5-15u²)`. Guardar `nfev`, `njev`, `nlu`, órdenes/pasos si están disponibles, estado de terminación y versión exacta de SciPy/Python. SciPy describe BDF como método implícito de orden variable 1–5 y recomienda dar Jacobiano o sparsity para sistemas dispersos.
4. Evaluar la salida en tiempos fijados `t_k=.005k`, `k=0,…,200` (201 snapshots); incluir también la malla temporal `.01` al exportar una tabla compatible con la demo DeepXDE. Guardar además la malla nativa para evitar que interpolación del export oculte el error.

### Mallas y parada predefinida

| Nivel | Intervalos espaciales `N` | `h=2/N` | Estado |
|---|---:|---:|---|
| AC-L0 | 256 | 0.0078125 | Obligatorio |
| AC-L1 | 512 | 0.00390625 | Obligatorio |
| AC-L2 | 1,024 | 0.001953125 | Obligatorio |
| AC-L3 | 2,048 | 0.0009765625 | Obligatorio |
| AC-L4 | 4,096 | 0.00048828125 | Obligatorio; candidato inicial a referencia final |
| AC-L5 | 8,192 | 0.000244140625 | Contingencia si no se cumple la regla en L4 |

En cada refinamiento se comparan `u_h` y `u_(h/2)` en nodos anidados comunes, en los 201 tiempos. Definir de antemano: (a) error L2 relativo espacio-tiempo con regla trapezoidal fija; (b) máximo del error L2 relativo espacial entre snapshots; ambos se calculan sobre el mismo dominio y tienen denominador de solución de referencia, no el del error. **Parar en el primer nivel que produzca dos pares refinados consecutivos bajo ambas condiciones:** error relativo espacio-tiempo `≤5e-5` y máximo error relativo por snapshot `≤5e-5`. Se reporta el nivel fino del segundo par como referencia; publicar además el nivel anterior para auditar la diferencia. Si AC-L5 no cumple, no declarar convergencia: ampliar a `N=16,384` o cambiar a esquema espacial de orden mayor en una revisión del protocolo.

Comprobación temporal independiente en la malla espacial final: repetir con `rtol=1e-12`, `atol=1e-14`, `max_step=.0025`; la diferencia frente a la corrida nominal en los mismos tiempos debe ser `≤1e-7` en ambas métricas L2 relativas anteriores. Si no cumple, reducir `max_step` a la mitad y repetir manteniendo la tolerancia; registrar cada intento. La aceptación espacial queda bloqueada hasta que la comprobación temporal pase.

Como chequeos diagnósticos, no sustitutos de la regla: conservar perfil y valores de frontera/IC; evaluar el residuo semidiscreto y un residuo PDE de postproceso en puntos no usados para entrenar; comparar con `Allen_Cahn.mat` oficial luego de interpolar a sus coordenadas. Una discrepancia con el `.mat` no invalida automáticamente el BDF: primero se determina si viene de interpolación o de la referencia del archivo. No se reajusta la regla de parada usando ese `.mat`.

## DFG: Taylor–Hood P2/P1 y refinamiento

### Geometría, espacios y solve fijados

1. Generar malla triangular no estructurada desde CAD reproducible para el rectángulo y el arco circular exactos. Etiquetar por separado inlet, walls, cylinder y outlet. Usar geometría de borde curva de segundo orden/isoparamétrica para no dejar que el error de aproximar el cilindro con un polígono domine la sucesión.
2. Usar Taylor–Hood con velocidad P2 y presión P1 sobre cada malla. Forma débil estacionaria: convección `((u·∇)u,v)`, difusión `ν(∇u,∇v)`, acoplamiento `-(p,∇·v)+(q,∇·u)`. No añadir términos esenciales al outlet: el do-nothing natural de la forma débil representa la tracción libre requerida. El inflow y paredes/cilindro son condiciones esenciales como arriba.
3. Obtener inicialización con Stokes y subir el término convectivo con continuación `α=0,.25,.5,.75,1`; Newton amortiguado con line-search por etapa. Fijar solver, tolerancias y precondicionador en el manifest. Criterio algebraico por etapa: residuo no lineal relativo `≤1e-10` y actualización relativa `≤1e-10` durante dos iteraciones consecutivas; registrar residuo lineal final `≤1e-10`. Si una etapa falla, reducir a la mitad el incremento `α`, dejando log completo. No añadir un pin de presión cuando la condición natural do-nothing ya fija el nivel, salvo que la biblioteca exija un nullspace/gauge; cualquier gauge debe quedar declarado y aplicarse igual en todas las mallas.
4. QoI primarios: integrar la tracción/esfuerzo sobre la superficie del cilindro con la normal y normalización oficiales de FEATFLOW. Los probes `(.15,.2)` y `(.25,.2)` están **sobre el contorno del cilindro**: evaluar la traza de presión P1 desde una celda adyacente del fluido en esas coordenadas exactas, registrar la regla de selección de celda y su sensibilidad al refinamiento, sin sustituirlos por el nodo más cercano. Refinar la cuadratura de borde por separado hasta que el cambio en Cd/Cl sea menor que una décima parte del umbral de malla. Calcular también fuerza por reacción de DOFs restringidos como diagnóstico FEM; no sustituir silenciosamente con la reacción el QoI de tracción que se contrasta con FEATFLOW. Para errores de campo, evaluar u y p en el mismo conjunto de puntos físicos/cuadratura, fuera del cilindro y fijado antes de ejecutar.

### Familia de mallas

Mantener idéntico mesher, regiones, orden geométrico, criterios de calidad y estrategia local de refinamiento en todos los niveles; reducir a la mitad ambos tamaños objetivo cada nivel. Los tamaños son cotas objetivo del generador, no número predicho de celdas. Registrar la malla `.msh`, versión del mesher, cantidad real de elementos/nodos/DOFs, tamaño mínimo/máximo, calidad mínima y tags.

| Nivel | Tamaño bulk objetivo | Tamaño cerca del cilindro | Estado |
|---|---:|---:|---|
| DFG-L0 | 0.040 | 0.0060 | Obligatorio |
| DFG-L1 | 0.020 | 0.0030 | Obligatorio |
| DFG-L2 | 0.010 | 0.0015 | Obligatorio |
| DFG-L3 | 0.005 | 0.00075 | Obligatorio; candidato inicial a referencia final |
| DFG-L4 | 0.0025 | 0.000375 | Contingencia si la regla no pasa en L3 |
| DFG-L5 | 0.00125 | 0.0001875 | Contingencia documentada solo si L4 falla y hay memoria/tiempo |

### Regla de parada numérica DFG

Para dos pares refinados consecutivos, requerir a la vez:

- Arrastre: `|Cd_fino-Cd_grueso|/|Cd_fino| ≤ 2e-4`.
- Caída de presión: `|Δp_fino-Δp_grueso|/|Δp_fino| ≤ 5e-4`.
- Sustentación: `|Cl_fino-Cl_grueso| ≤ 2e-5` (tolerancia absoluta por el valor pequeño/cancelación).
- Campo en puntos de comparación comunes: `||u_fino-u_grueso||L2/||u_fino||L2 ≤ 5e-3`; para presión restar la media de cada campo antes de comparar y exigir `≤1e-2` con denominador `||p_fino-mean(p_fino)||L2`.

Además, al nivel aceptado, contrastar contra los QoIs publicados por FEATFLOW: `|Cd-Cd_ref|/|Cd_ref|≤1e-3`, `|Δp-Δp_ref|/|Δp_ref|≤2e-3`, `|Cl-Cl_ref|≤5e-5`. Estas cotas son criterios predefinidos de aceptación de la referencia, no afirmaciones sobre el error que se alcanzará. La condición de malla debe pasar en dos refinamientos consecutivos; si los QoIs convergen pero el campo no, se puede publicar la tabla de QoIs como convergida, pero etiquetar el campo solo como visual/preliminar. Si L5 no satisface la regla, reportar “sin convergencia suficiente” y no escoger un nivel porque su valor parezca mejor.

El pequeño lift puede ser sensible a asimetría de mallado: centrar el cilindro en `y=.2` es obligatorio aunque el canal tenga altura `.41`; no moverlo a la línea media `.205`. Publicar Cl con signo, valor absoluto y variación entre mallas. Hacer un chequeo de sensibilidad de cuadratura/triangulación del contorno separado del refinamiento del interior.

## Archivos, versiones e integridad

Los siguientes son nombres de entrega propuestos; no se crean resultados ni hashes durante la redacción del protocolo.

```text
reference-protocol-ac-dfg.md
reference/ac_bdf/
  solve_ac_bdf.py
  environment.lock
  manifest.json
  convergence.csv
  solver_stats.json
  ac_reference.npz
  ac_reference_deepxde_grid.npz
  SHA256SUMS.txt
reference/dfg_taylor_hood/
  make_mesh.geo
  solve_dfg.py
  environment.lock
  manifest.json
  meshes/dfg_re20_L0.msh ... dfg_re20_L5.msh
  convergence.csv
  solver_stats.json
  dfg_re20_reference.xdmf
  dfg_re20_reference.h5
  SHA256SUMS.txt
```

`manifest.json` debe incluir contrato completo, unidades, solver/commit, versions de Python/SciPy/FEniCS o DOLFINx/PETSc/Gmsh, sistema operativo, CPU/RAM, comandos exactos, semilla de generación de malla si aplica, tolerancias, malla final, DOFs y estado de las reglas de parada. El lockfile debe congelar dependencias y resolverlas desde fuente oficial. Cada CSV incluye encabezado/unidades, pareja de mallas, DOFs/elementos, valores QoI/métricas y ratios. El campo XDMF/HDF5 debe indicar espacios y orden de los campos. No eliminar niveles previos tras aceptar el final.

Después de congelar cada subdirectorio, crear su `SHA256SUMS.txt` con SHA-256 para los scripts, lockfiles, manifests, todas las mallas y salidas (incluidos los archivos HDF5 referidos por XDMF). No rellenar hashes antes de que existan los bytes finales; guardar además el hash del archivo de sumas en el registro del experimento/commit que lo distribuya. Si se vuelve a correr o edita cualquier entrada/salida, crear un nuevo ID de corrida y manifiesto, sin sobrescribir una referencia aceptada.

## Presupuesto provisional con incertidumbre

Son reservas para planificar recursos, no tiempos medidos ni promesas. La incertidumbre es alta porque no se ha ejecutado código en esta máquina y los papers usan hardware/implementaciones distintos.

| Trabajo | Reserva inicial | Incertidumbre y riesgo |
|---|---|---|
| Allen–Cahn BDF AC-L0…L4, chequeo temporal en la malla final y posible AC-L5 | 1–8 CPU core-hours; memoria límite de planificación 4 GB | Aproximadamente ×5 en tiempo hasta medir AC-L0/L1. La matriz dispersa tridiagonal debe mantenerse dispersa; un Jacobiano denso invalida el presupuesto. 1D permite correr en CPU. |
| DFG FEM DFG-L0…L3 y postprocesado; posible L4/L5 si la regla no pasa | 8–48 CPU core-hours para L0–L3; reservar 32–64 GB RAM. L4/L5 son extensión separada, planear otras 16–100 core-hours y hasta 64–128 GB si se usa LU directo. | Al menos ×3–5 por solver/mesh quality/curved geometry/line-search y coste de factorización. La cifra publicada de jNO (~5 s para 10,038 DOFs) solo es un punto de contexto en su entorno, no una extrapolación válida. Calibrar el coste con L0 y L1; si se cambia LU por método iterativo/precondicionador, repetir niveles previos con la misma configuración de solver para comparar. |
| Preparación y revisión humana de scripts, mallas, manifests y auditoría de reglas | 1–3 jornadas de ingeniería, independiente del solve | Variación aproximada ×2 según experiencia con el stack FEM instalado y revisión de force/QoI. |

Si el primer solve FEM supera la memoria asignada, detener antes de generar niveles finos, medir el patrón de escala y aprobar una variante de solver en el protocolo; no bajar la calidad de la referencia ni eliminar la regla de parada para ajustar el presupuesto.

## Puerta obligatoria antes de solicitar GPU

No comenzar entrenamiento GPU que dependa de las referencias hasta completar y revisar estos puntos:

1. Contrato AC y DFG congelado en manifests, con las BC/valores anteriores y el do-nothing del outlet explícito.
2. Referencia AC pasa las dos comparaciones espaciales consecutivas y el control temporal; se guardan `convergence.csv`, solver stats, grillas nativas y export compatible con DeepXDE. La comparación con `.mat` queda documentada como auxiliar.
3. Referencia FEM converge bajo la regla predefinida en dos refinamientos y satisface la comprobación FEATFLOW. Campo, interpolador, integrales de tracción y presión en probes quedan congelados; si solo pasan los QoIs, cualquier entrenamiento de campo debe esperar al campo convergido.
4. Revisión independiente del postprocesador de QoI: signos de normal/fuerza, normalización oficial, nombres de inlet/wall/cylinder/outlet, presión por interpolación exacta, gauge documentado y prueba de sensibilidad de cuadratura del cilindro.
5. Scripts, locks, mallas, datos, logs y manifests están completos, sus sumas SHA-256 se recalculan y se verifica integridad al cargarlos en el pipeline de evaluación.
6. Registrar el costo observado de los solves de referencia como trabajo científico. Se omiten los smoke tests y validadores locales de software por la instrucción vigente del investigador; esa omisión se consigna en el scorecard.

Solo después de la puerta se asigna presupuesto GPU a PINN. La GPU no se necesita para estos solves de referencia. Si se usa GPU para entrenamiento, guardar modelo, commit/código, configuración, seeds, hardware, coste y versión de los hashes de referencia. El scorecard debe indicar que se omitió la validación local de software solicitada por el investigador.

## Fuentes

- [DeepXDE — Allen–Cahn forward problem](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html)
- [SciPy — `scipy.integrate.BDF`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html)
- [FEATFLOW/TU Dortmund — DFG benchmark 2D-1 at Re=20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)
- [FEniCS Hands-on — Navier–Stokes cylinder tutorial](https://fenics-handson.readthedocs.io/en/latest/navierstokes/doc.html)
- [jNO — FEM Taylor–Hood DFG 2D-1 implementation](https://fhg-iisb.github.io/jNO/tutorials/08-fem-and-varpinns/navier-stokes-cylinder-dfg/)
