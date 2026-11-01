# Propuesta de expansión del problema piezoeléctrico

Este borrador responde a la necesidad de que el capítulo explique el fenómeno antes de mostrar antecedentes y de que el lector pueda reconstruir el contrato físico de P5 sin inferirlo de una figura de resultados. La propuesta queda separada del texto teórico existente y no modifica `capitulos/marco/04-piezo.tex`, la metodología ni los contratos de `context/`.

## Alcance y evidencia revisada

La ruta activa (`workflow/index.toml`) clasifica la solicitud como síntesis/redacción con gate `scientific_claim`. La política (`context/agent-policy.toml`) permite preparar el borrador en `var/agent_runs` y deja la aceptación del claim y la modificación del texto teórico a revisión humana. La ontología (`context/research-ontology.toml`) define un problema como la combinación de PDE, dominio, condiciones y solución de referencia; por eso la expansión separa esas cuatro piezas.

La revisión se basó en estos archivos del repositorio:

- `capitulos/marco/04-piezo.tex`: ya contiene la convención de cortante ingenieril, `\mathbf E=-\nabla\varphi`, la entalpía eléctrica, las leyes constitutivas generales, balances, condiciones de interfaz y la forma débil/FEM. Su debilidad narrativa es que pasa rápido de esas fórmulas al antecedente y no hace visible cómo los signos de P5 producen una tendencia de flexión.
- `capitulos/metodologia/01-problemas.tex`, subsección P5: fija el dominio 2D efectivo, (L=0.1\,\mathrm m), (H=0.001\,\mathrm m), interfaz en (y=H/2), empotramiento izquierdo, superficies mecánicas libres, (arphi=100\,\mathrm V) arriba, (arphi=0) abajo, aislamiento eléctrico en los extremos y la tabla de (C), (e) y (kappa). También deja claro que es una reformulación efectiva y no una reconstrucción 3D certificada.
- `capitulos/metodologia/02-mecanismo.tex`: declara ramas por campo y por capa para permitir saltos de derivada y usa continuidad de campos y flujo normal en la interfaz. La expansión física debe explicar por qué esa decisión representa una propiedad del problema, sin repetir el diseño del controlador.
- `m-bibliografia.bib`: las claves disponibles para sostener el texto son `lahmer2008`, `nye2012crystals`, `khan2018bimorph`, `gonzalez2024tesis` y `gonzalez2026pinnpiezo`. No se añadieron fuentes externas ni claves nuevas.

## Qué debe cambiar la narración

La subsección nueva debe entrar después de la forma débil/FEM y antes de `Antecedente local y lectura de sus figuras`. En ese lugar el lector ya conoce las ecuaciones generales y recibe ahora una instanciación completa del caso que luego aparece en metodología. El orden recomendado es:

1. Explicar la cadena causal: voltaje (ightarrow) campo eléctrico (ightarrow) términos piezoeléctricos (ightarrow) deformaciones opuestas (ightarrow) curvatura por compatibilidad.
2. Fijar geometría y orientación: (x) hacia la punta, (y) hacia arriba, (Omega^-) abajo, (Omega^+) arriba, interfaz en (H/2), normal común de abajo hacia arriba.
3. Recordar solo la notación necesaria y especializar (e) a los coeficientes no nulos de P5. La derivación general desde la entalpía permanece en la sección existente.
4. Escribir las tres ecuaciones fuertes por capa y las condiciones mecánicas, eléctricas y de transmisión con sus nombres físicos.
5. Explicar los signos y cerrar con la diferencia entre un esquema que comunica el contrato y una solución FEM que todavía no existe.

El archivo `var/agent_runs/teoria-visual/piezo-expansion.tex` contiene esta integración en una subsección de unas pocas páginas. La metodología no se repite: solo se usan sus valores para interpretar el constitutivo y se marca cada deducción que no es resultado numérico.

## Convención física que debe quedar visible

El dominio es (Omega=(0,L)\times(0,H)), con (L=0.1\,\mathrm m), (H=0.001\,\mathrm m), y capas de espesor (H/2). Se usan (\mathbf d=(d_x,d_y)) y potencial (arphi). El signo eléctrico es

\[
\mathbf E=-\nabla\varphi.
\]

Con (e_{21}^{0}=0.02904\,\mathrm{C}/\mathrm{m}^{2}) y (e_{22}^{0}=0.05157\,\mathrm{C}/\mathrm{m}^{2}), la reducción que ya adoptó la metodología puede escribirse

\[
e_\ell=\eta_\ell
\begin{bmatrix}0&0&0\\ e_{21}^{0}&e_{22}^{0}&0\end{bmatrix},
\qquad \eta_-=+1,\quad \eta_+=-1.
\]

Con los coeficientes constantes dentro de cada capa, las componentes útiles son

\[
\begin{aligned}
\sigma_{xx}&=C_{11}d_{x,x}+C_{12}d_{y,y}+\eta_\ell e_{21}^{0}\varphi_{,y},\\
\sigma_{yy}&=C_{12}d_{x,x}+C_{22}d_{y,y}+\eta_\ell e_{22}^{0}\varphi_{,y},\\
\sigma_{xy}&=C_{66}(d_{x,y}+d_{y,x}),\\
D_x&=-\kappa_x\varphi_{,x},\\
D_y&=\eta_\ell(e_{21}^{0}d_{x,x}+e_{22}^{0}d_{y,y})-\kappa_y\varphi_{,y}.
\end{aligned}
\]

El equilibrio cuasiestático sin fuerza volumétrica ni carga libre es

\[
\partial_x\sigma_{xx}+\partial_y\sigma_{xy}=0,
\qquad
\partial_x\sigma_{xy}+\partial_y\sigma_{yy}=0,
\qquad
\partial_xD_x+\partial_yD_y=0,
\]

interpretado por separado en (Omega^-) y (Omega^+). Al sustituir el constitutivo, las PDE fuertes que conviene mostrar son

\[
\begin{aligned}
C_{11}u_{,xx}+C_{66}u_{,yy}+(C_{12}+C_{66})v_{,xy}
  +\eta_\ell e_{21}^{0}\varphi_{,xy}&=0,\\
C_{66}v_{,xx}+C_{22}v_{,yy}+(C_{12}+C_{66})u_{,xy}
  +\eta_\ell e_{22}^{0}\varphi_{,yy}&=0,\\
\eta_\ell e_{21}^{0}u_{,xy}+\eta_\ell e_{22}^{0}v_{,yy}
  -\kappa_x\varphi_{,xx}-\kappa_y\varphi_{,yy}&=0.
\end{aligned}
\]

En las fronteras, el contrato es

\[
\begin{array}{lll}
\mathbf d=\mathbf 0 & \text{en }\Gamma_L, &
\sigma\mathbf n=\mathbf 0\text{ en }\Gamma_R\cup\Gamma_T\cup\Gamma_B,\\
\varphi=100\,\mathrm V & \text{en }\Gamma_T, &
\varphi=0\text{ en }\Gamma_B,\\
\mathbf D\cdot\mathbf n=0 & \text{en }\Gamma_L\cup\Gamma_R. &
\end{array}
\]

En la interfaz, con ([q]=q^+-q^-) y (mathbf n=(0,1)), se exige

\[
[\mathbf d]=0,qquad [\varphi]=0,qquad
[\sigma\mathbf n]=0,qquad [\mathbf D\cdot\mathbf n]=0,
\]

o, equivalentemente, continuidad de (d_x,d_y,arphi,sigma_{xy},sigma_{yy},D_y). La continuidad de los campos no obliga a que sus derivadas normales sean continuas. Para este P5 particular no conviene afirmar sin matiz que (D_x) salta: la primera fila de (e) es cero y (kappa_x) es igual en ambas capas, de modo que una traza de (arphi) diferenciable en (x) da (D_x=-kappa_xarphi_{,x}) continuo. Lo que sí debe destacarse es que el flujo tangencial no es una condición de transmisión independiente y que los saltos normales requieren ramas por material.

## Lectura de signos sin inventar una solución

Con (arphi=0) abajo y (arphi=100\,\mathrm V) arriba, una tendencia de potencial creciente en (y) da (arphi_{,y}>0) y (E_y<0): el campo apunta hacia abajo. La deformación libre local que anula la tensión se obtiene de

\[
s_\ell^{\mathrm{libre}}=-(C^E)^{-1}e_\ell^T\nabla\varphi.
\]

Como (e) cambia de signo entre capas, la lectura axial da extensión arriba y contracción abajo en la convención adoptada. Con el gradiente nominal que ya usa metodología, (E_0=10^5\,\mathrm{V}/\mathrm m), la sustitución algebraica en una capa libre da para la capa superior aproximadamente

\[
\varepsilon_{xx}=7.04\times10^{-7},\qquad
\varepsilon_{yy}=2.16\times10^{-6},
\]

y la capa inferior cambia ambos signos. Esto es una escala constitutiva local, no una flecha ni una solución FEM. Al unir las capas, la diferencia de deformación induce una tendencia de flexión hacia abajo en el dibujo con (x) a la derecha y (y) hacia arriba. La magnitud de la flecha, la curvatura espacial, el campo no uniforme y la distribución de carga deben quedar reservados para la solución de referencia.

El término dieléctrico de (D_y) tiene signo opuesto a (arphi_{,y}), pero el término de polarización inducida puede compensarlo; por eso no debe afirmarse el signo total de (D_y) antes de resolver. Invertir los electrodos debe invertir las respuestas del modelo lineal. La corrida a 50 V descrita en metodología puede usar esa simetría como prueba de consistencia, sin presentar el comportamiento como resultado previo.

## Figuras propuestas

Las dos figuras solicitadas son suficientes para esta expansión. Deben aparecer antes de la lectura del antecedente numérico y sus captions deben mantener la diferencia entre esquema y solución.

### `p5-bimorfo.pdf`: geometría y mecanismo de flexión

La figura debe tener una geometría plana de dos capas, preferiblemente en dos paneles:

- Panel izquierdo: rectángulo sin deformar, interfaz horizontal en (y=H/2), (x) hacia la derecha, (y) hacia arriba, cotas (L) y (H), (x=0) sombreado como empotramiento y leyenda de capas inferior/superior.
- Marcar (arphi=0\,\mathrm V) en la cara inferior y (arphi=100\,\mathrm V) en la cara superior. Dibujar (\mathbf E) hacia abajo con la leyenda (\mathbf E=-\nabla\varphi). Marcar las superficies mecánicas libres y los extremos (D_x=0).
- Hacer visible el signo del acoplamiento: (e_{21},e_{22}>0) abajo y (e_{21},e_{22}<0) arriba, sin inventar flechas de polarización cristalográfica que no estén fijadas por la fuente.
- Panel derecho: contorno curvado hacia abajo, con deformación exagerada y una nota “esquema, no solución FEM”. Puede incluir flechas cualitativas “extensión arriba” y “contracción abajo”. No debe tener barra de color, valor de flecha ni campo numérico.

Caption recomendado: “Geometría y mecanismo cualitativo del bimorfo efectivo P5. La diferencia de potencial induce signos opuestos de deformación libre en las capas; la unión produce una tendencia de flexión hacia abajo según la convención (x) derecha, (y) arriba. La deformada está exagerada y no es una solución FEM.”

### `p5-interfaz.pdf`: continuidad frente a salto de derivada

Esta figura no representa el PVDF. Es un contraejemplo dieléctrico escalar, didáctico y normalizado:

\[
\hat y=y/H,\qquad \hat\varphi(0)=0,\qquad \hat\varphi(1)=1,
\qquad \hat\kappa_1=1,\quad \hat\kappa_2=2.
\]

Debe mostrar la interfaz en (hat y=0.5), potencial continuo con pendientes (4/3) en la capa 1 y (2/3) en la capa 2, y flujo

\[
\hat D=-\hat\kappa\,\frac{\mathrm d\hat\varphi}{\mathrm d\hat y}=-\frac43
\]

continuo a ambos lados. La figura debe rotular claramente “ejemplo escalar sin (e), no PVDF” para que el lector no interprete (hat\kappa_1,hat\kappa_2) como los valores de P5. Su función es mostrar que el campo y el flujo normal pueden ser continuos mientras la derivada cambia por el salto de propiedad.

No hace falta una tercera figura en el texto actual. Una figura adicional de solución solo tendría sentido después de obtener la referencia FEM: debería mostrar una variable identificada, unidades, malla o puntos, barra de color y una comparación de referencia/predicción/error. Antes de esa corrida, agregarla convertiría un esquema en una afirmación no sustentada.

## Lagunas que deben quedar explícitas

- P5 se define como reducción 2D efectiva; no se debe llamar reconstrucción tridimensional certificada.
- El mapeo de índices de la fuente a (e_{21},e_{22}) es una decisión de reformulación. El texto debe mantenerla separada de una reproducción literal de la ecuación de la fuente.
- La metodología no etiqueta de forma visible si la reducción es de tensión plana o de deformación plana. Esa elección debe fijarse antes de llamar “solución de referencia” a un FEM; no se debe inventar aquí.
- Las permitividades negativas observadas en la transcripción del antecedente no se deben copiar: P5 adopta permitividades positivas y coloca el signo en (\mathbf E=-\nabla\varphi). Esto es una convención del modelo adoptado, no evidencia de lo que ejecutó el código de la fuente.
- No existe en los archivos revisados una flecha FEM, un mapa de potencial, una carga de electrodo o una curva de error de la presente corrida. Las figuras nuevas no deben insinuar esos resultados.
- La continuidad de (D_y) en la interfaz supone ausencia de carga libre superficial; si el contrato cambiara, la condición tendría que cambiar explícitamente.
- Las condiciones en las esquinas se interpretan por trazas de cada segmento, como ya declara metodología; no se debe dibujar una condición normal adicional en el punto.

## Criterios de revisión humana

Antes de promover esta expansión al capítulo, conviene revisar que:

1. La orientación de (x), (y), la normal de interfaz, los potenciales y el sentido de (\mathbf E) coincidan con `p5-bimorfo.pdf`.
2. Los signos de (e_{21},e_{22}) y la tendencia “extensión arriba/contracción abajo” coincidan con la convención efectiva adoptada.
3. `p5-interfaz.pdf` conserve sus valores didácticos (1,2,4/3,2/3,-4/3) y esté rotulada como ejemplo escalar sin acoplamiento.
4. Se mantenga la distinción entre deducción constitutiva local, esquema y solución FEM.
5. Se decida explícitamente tensión plana o deformación plana antes de cerrar la referencia numérica.

