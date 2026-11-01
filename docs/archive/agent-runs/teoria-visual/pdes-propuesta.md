# Propuesta de expansión del marco teórico: EDP y problemas canónicos

## Alcance y control de cambios

La revisión siguió el flujo 'literature'/'synthesis' de workflow/index.toml:
la solicitud activa marco teórico, fuentes, LaTeX y redacción. Según
context/agent-policy.toml, este producto es un borrador de
'research_synthesizer'; cualquier modificación del texto teórico canónico
requiere aprobación humana. La entrega es, por tanto, una propuesta
integrable y no modifica i-marco_teorico.tex, los capítulos canónicos de
capitulos/marco o capitulos/metodologia, el vault ni las políticas.

El fragmento listo para integrar está en
var/agent_runs/teoria-visual/pdes-expansion.tex. Contiene una sección
pedagógica con las ecuaciones completas de P1--P4 y cuatro entornos figure
que solicitan exactamente p1-poisson.pdf, p2-burgers.pdf,
p3-allen-cahn.pdf y p4-dfg.pdf.

## Hallazgos del repositorio

- i-marco_teorico.tex:1-6 es un agregador: presenta el recorrido general y
  carga capitulos/marco/01-aproximacion.tex a 05-sintesis.tex. No contiene
  una lectura física separada de P1--P4.
- capitulos/marco/01-aproximacion.tex:1-117 ya define EDP, residual frente a
  error, escalamiento, Poisson, transporte y Burgers, además de la forma débil
  y FEM. La propuesta no repite la derivación de diferencias finitas,
  volúmenes o ensamblaje; añade interpretación física de los contratos y
  conecta cada balance con su figura.
- capitulos/marco/02-pinn.tex:1-84 y 03-kan.tex:1-85 cubren PINNs, KAN,
  B-splines, residuales y adaptación. La expansión no vuelve a explicar
  pérdidas, muestreo, optimización, sensibilidad, nudos o transferencia.
- capitulos/marco/04-piezo.tex:1-109 y 05-sintesis.tex:1-27 cubren el
  acoplamiento piezoeléctrico y los antecedentes. La expansión solicitada
  llega hasta P4; P5 se conserva allí y no se duplica.
- capitulos/metodologia/01-problemas.tex:8-126 es la fuente de verdad del
  banco. Fija P1 (Poisson manufacturado), P2 (onda exacta de Burgers), P3
  (Allen--Cahn de DeepXDE), P4 (DFG 2D-1) y P5 (bimorfo PVDF).
  02-mecanismo.tex y 03-evaluacion.tex definen acciones, controles,
  referencias, métricas y falsación; no se repiten en el fragmento.
- Las citas utilizadas ya existen en m-bibliografia.bib: evans1998pde,
  fenics2016, burgerswaves2025, deepxdeAC, dfgRe20, raissi2019 y las
  referencias de PINNs/KAN ya empleadas. No se añadieron claves.

## Organización propuesta del texto

La sección nueva se titula 'Lectura física de los problemas canónicos' y está
pensada para insertarse después de capitulos/marco/01-aproximacion.tex y
antes de 02-pinn.tex. Así la secuencia queda: qué es una EDP y cómo se
aproxima; qué diferencias físicas separan los contratos; y qué
representaciones neuronales intentan resolverlos. Las etiquetas del fragmento
usan el prefijo expansion- y no chocan con eq:p1--eq:p5 ni con
fig:canonicos.

El texto sigue cuatro contrastes físicos:

1. P1 es un equilibrio elíptico con frecuencia distribuida. Se explica por
   qué la curvatura y la fuente escalan como \(k^2\), y por qué una ganancia
   allí no prueba focalización localizada.
2. P2 es una onda viscosa exacta. Se derivan \(u_x\), \(u_t\) y \(u_{xx}\), y
   se cuantifica que pasar de \(\nu=0.02\) a \(0.005\) hace el frente cuatro
   veces más estrecho y lleva el gradiente máximo de \(6.25\) a \(25\).
3. P3 se interpreta como flujo de gradiente con potencial de doble pozo. Se
   distinguen \(u=\pm1\), reacción e interfaz, con el perfil estacionario de
   escala \(\ell_{\rm AC}=\sqrt{2d/5}=0.02\) para \(d=0.001\).
4. P4 liga Navier--Stokes con el balance integral, la entrada parabólica, la
   salida del benchmark y las fuerzas integrales. Se deja claro que un campo
   visualmente plausible no garantiza incompresibilidad, tracción o arrastre.

El fragmento declara que los dibujos son ilustraciones analíticas o esquemas
de contrato. No presenta una corrida PINN/PIKAN, una mejora, una convergencia
FEM ni una validación de referencia como resultado.

## Especificación matemática de las figuras

Las siguientes especificaciones son el contrato de contenido de cada PDF.
Los títulos y pies correspondientes ya están en pdes-expansion.tex.

### p1-poisson.pdf

- **Propósito:** visualizar frecuencia distribuida y conectar la solución con
  la fuente de Poisson.
- **Paneles:** mapas de \(u_k^*\) para \(k=1\) y \(k=4\), más cortes
  unidimensionales en \(y=1/(2k)\). La disposición recomendada es una
  cuadrícula 2x2: mapas arriba y cortes abajo.
- **Definición:** \(\Omega=(0,1)^2\),
  \(u_k^*(x,y)=\sin(k\pi x)\sin(k\pi y)\),
  \(f_k=2(k\pi)^2u_k^*\), \(u|_{\partial\Omega}=0\). En cada corte debe
  verse que \(u_k^*(x,1/(2k))=\sin(k\pi x)\).
- **Escalas:** barra de color común y centrada en \([-1,1]\); ejes
  \(x,y\in[0,1]\). La fuente queda documentada en el texto, sin añadir
  etiquetas de fuente al mapa ni inventar datos de una solución numérica.
- **Estatus:** evaluación exacta de una fórmula manufacturada, usada como
  ilustración matemática. No es una solución producida por el mecanismo ni
  un resultado FEM.

### p2-burgers.pdf

- **Propósito:** mostrar estrechamiento, movimiento y cancelación de términos
  en una onda viscosa.
- **Paneles:** perfiles iniciales para \((c,\nu)=(0.5,0.02)\) y
  \((0.5,0.005)\); concentración de \(|u_x|\); trayectoria
  \(x_f(t)=ct+x_0\) para \(\nu=0.02\); y comparación de
  \(u_t\), \(u\,u_x\) y \(-\nu u_{xx}\) en \(t=0.5\). Una cuadrícula 2x3
  conserva los cinco paneles sin mezclar escalas.
- **Definición:** \(x\in[-1,1]\), \(t\in[0,1]\), \(A=0.5\), \(x_0=-0.25\),
  \(u^*=c-A\tanh[A(x-ct-x_0)/(2\nu)]\). Las derivadas deben salir de la
  fórmula exacta y no de diferencias de la imagen.
- **Valores de control:** \(\ell=2\nu/A\) vale \(0.08\) y \(0.02\);
  \(\max|u_x|=A^2/(2\nu)\) vale \(6.25\) y \(25\).
- **Estatus:** diagnóstico analítico exacto. Las curvas de términos
  ilustran \(u_t+u\,u_x-\nu u_{xx}=0\); no son historia de entrenamiento ni
  evidencia de error.

### p3-allen-cahn.pdf

- **Propósito:** explicar reacción e interfaz antes de discutir una referencia
  evolutiva.
- **Paneles:** (i) potencial
  \(W(u)=\frac54(u^2-1)^2\); (ii) reacción
  \(R(u)=5(u-u^3)=-W'(u)\); (iii) perfil *kink*
  \(u_{\rm kink}(x)=\tanh[(x-x_c)/\sqrt{2d/5}]\), con \(d=0.001\) y
  \(\ell_{\rm AC}=0.02\). Una fila 1x3 facilita leer los tres objetos.
- **Contrato al que remite el pie:** \(u_t-d u_{xx}-5(u-u^3)=0\),
  \(x\in[-1,1]\), \(t\in[0,1]\), \(u(x,0)=x^2\cos(\pi x)\) y
  \(u(-1,t)=u(1,t)=-1\).
- **Restricción:** no incluir evolución numérica, mapa espacio-tiempo, máscara
  de interfaz ni curva de error. El *kink* ilustra una escala estacionaria;
  no es la solución completa con el dato inicial del contrato.
- **Estatus:** ejemplo físico ilustrativo. No es la referencia independiente
  de diferencias centradas/BDF ni un resultado de DeepXDE.

### p4-dfg.pdf

- **Propósito:** fijar geometría y signos del balance DFG 2D-1 sin sugerir
  una solución de flujo que todavía no se ha evaluado.
- **Paneles:** geometría exacta del canal y cilindro con entrada, salida,
  paredes y dimensiones; perfil parabólico de entrada; y diagrama de los
  términos convectivo, de presión y viscoso. No incluir líneas de corriente.
- **Definición geométrica:** \(\Omega=([0,2.2]\times[0,0.41])
  \setminus\overline{B_{0.05}(0.2,0.2)}\). La entrada es
  \(\mathbf u_{\rm in}(y)=(4(0.3)y(0.41-y)/0.41^2,0)\): su máximo es
  \(0.3\) y su media transversal es \(0.2\). Paredes y cilindro tienen
  \(\mathbf u=\mathbf0\); la salida es
  \(\nu\partial_n\mathbf u-p\mathbf n=\mathbf0\), con \(\nu=0.001\),
  \(U=0.2\), \(L=0.1\), \(Re=20\).
- **Balance:** rotular \(\mathbf u\cdot\nabla\mathbf u\), \(\nabla p\) y
  \(-\nu\Delta\mathbf u\), dejando claro que se suman a cero en momento. Si
  se dibuja la tracción, usar la convención local
  \(\boldsymbol\sigma_b=\nu\nabla\mathbf u-pI\).
- **Estatus:** esquema de geometría/contrato, no solución de Navier--Stokes,
  no *streamlines* y no valores calculados de \(C_D\), \(C_L\) o
  \(\Delta p\). Los valores públicos permanecen como controles externos en
  metodología.

## Integración y verificación

El documento principal define '\graphicspath{ {figuras/} }' en z-main.tex.
Por ello, los cuatro PDF deben estar bajo la ruta que resuelva esa búsqueda
al integrar el fragmento. Durante esta corrida hay copias de trabajo en
var/agent_runs/teoria-visual/figures/; el fragmento conserva los nombres
desnudos solicitados para que el documento principal pueda resolverlos desde
figuras/ después de la promoción manual.

El fragmento usa únicamente amsmath, amssymb, graphicx y macros que ya están
en el preámbulo. No introduce paquetes ni claves bibliográficas nuevas.
Antes de promoverlo conviene comprobar que los PDF estén disponibles bajo
figuras/, que Biber resuelva m-bibliografia.bib y que los pies sigan marcando
dibujos y evaluaciones analíticas como ilustraciones. La aprobación de las
ecuaciones y la decisión de modificar el texto canónico siguen siendo un
gate humano.

## Archivos entregados

- var/agent_runs/teoria-visual/pdes-propuesta.md: esta propuesta, hallazgos,
  especificaciones y cautelas de integración.
- var/agent_runs/teoria-visual/pdes-expansion.tex: texto LaTeX listo para
  insertar, de aproximadamente cuatro a seis páginas con figuras, centrado en
  la interpretación física de P1--P4.

No se modificaron i-marco_teorico.tex, capitulos/marco/*,
capitulos/metodologia/*, m-bibliografia.bib, PIKAN-Vault/PIKAN-Vault,
workflow/ ni context/agent-policy.toml.
