from __future__ import annotations
import json, sys
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(r'C:\Users\rodri\Documents\AF_PIKANs')
OUT=ROOT/'var/agent_runs/research-progress-brief-20260923'
DATA=json.loads((OUT/'summary_for_brief.json').read_text(encoding='utf-8'))
LABELS={'rbf_fixed':'RBF-PIKAN fija','mlp_tanh':'MLP tanh','bspline_cubic':'PIKAN B-spline'}
DOC_PATH=ROOT/'docs/brief-avance-investigacion-2026-09-23.docx'
NAVY='17324D'; TEAL='1B7F79'; PALE='EAF2F6'; LIGHT='F3F6F8'; GREY='5D6B78'; WHITE='FFFFFF'; ORANGE='C75B39'

doc=Document()
sec=doc.sections[0]
sec.top_margin=Inches(.62); sec.bottom_margin=Inches(.62); sec.left_margin=Inches(.72); sec.right_margin=Inches(.72)
sec.header_distance=Inches(.28); sec.footer_distance=Inches(.28)
styles=doc.styles
normal=styles['Normal']; normal.font.name='Aptos'; normal.font.size=Pt(9.5); normal.font.color.rgb=RGBColor.from_string('243746')
normal.paragraph_format.space_after=Pt(5); normal.paragraph_format.line_spacing=1.08
for stname,size,color in [('Title',28,NAVY),('Subtitle',13,GREY),('Heading 1',17,NAVY),('Heading 2',12,TEAL),('Heading 3',10,NAVY)]:
    st=styles[stname]; st.font.name='Aptos Display' if stname in ('Title','Heading 1','Heading 2') else 'Aptos'; st.font.size=Pt(size); st.font.bold=stname!='Subtitle'; st.font.color.rgb=RGBColor.from_string(color)
    if stname.startswith('Heading'):
        st.paragraph_format.keep_with_next=True; st.paragraph_format.space_before=Pt(9); st.paragraph_format.space_after=Pt(4)

# Header and footer.
h=sec.header.paragraphs[0]; h.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=h.add_run('AF_PIKANs   /   BRIEF DE AVANCE DE INVESTIGACIÓN'); r.bold=True; r.font.size=Pt(8); r.font.color.rgb=RGBColor.from_string(TEAL)
f=sec.footer.paragraphs[0]; f.alignment=WD_ALIGN_PARAGRAPH.CENTER
rr=f.add_run('Documento exploratorio · 23 septiembre 2026     |     '); rr.font.size=Pt(8); rr.font.color.rgb=RGBColor.from_string(GREY)
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); f._p.append(fld)

def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def set_cell_text(cell,text,bold=False,color=None,size=8.2):
    cell.text=''; p=cell.paragraphs[0]; p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.0
    r=p.add_run(str(text)); r.bold=bold; r.font.name='Aptos'; r.font.size=Pt(size)
    if color: r.font.color.rgb=RGBColor.from_string(color)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(headers,rows,widths=None,size=8.0):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'; t.autofit=False
    for i,x in enumerate(headers):
        set_cell_text(t.rows[0].cells[i],x,True,WHITE,size); shade(t.rows[0].cells[i],NAVY)
    for ridx,row in enumerate(rows):
        cells=t.add_row().cells
        for i,val in enumerate(row):
            set_cell_text(cells[i],val,False,None,size)
            if ridx%2==1: shade(cells[i],LIGHT)
    if widths:
        for row in t.rows:
            for i,w in enumerate(widths): row.cells[i].width=Inches(w)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
    return t

def p(text='',style=None,bold_prefix=None,italic=False):
    para=doc.add_paragraph(style=style)
    if bold_prefix and text.startswith(bold_prefix):
        r=para.add_run(bold_prefix); r.bold=True; para.add_run(text[len(bold_prefix):])
    else:
        r=para.add_run(text); r.italic=italic
    return para

def bullet(text,level=0):
    para=doc.add_paragraph(style='List Bullet' if level==0 else 'List Bullet 2'); para.paragraph_format.space_after=Pt(2); para.add_run(text); return para

def callout(label,text,fill=PALE):
    t=doc.add_table(rows=1,cols=1); t.alignment=WD_TABLE_ALIGNMENT.CENTER; c=t.cell(0,0); shade(c,fill); c.text=''
    q=c.paragraphs[0]; q.paragraph_format.space_after=Pt(1); a=q.add_run(label+'  '); a.bold=True; a.font.color.rgb=RGBColor.from_string(TEAL); q.add_run(text)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)

def image(name,width=6.8):
    path=OUT/'figures'/name
    para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER; para.paragraph_format.space_after=Pt(1)
    para.add_run().add_picture(str(path),width=Inches(width))

def caption(text):
    para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER; para.paragraph_format.space_after=Pt(4)
    r=para.add_run(text); r.italic=True; r.font.size=Pt(8); r.font.color.rgb=RGBColor.from_string(GREY)

def fmt(x, sig=3):
    if x==0: return '0'
    ax=abs(float(x))
    if ax<0.001 or ax>=10000: return f'{float(x):.{sig-1}e}'
    return f'{float(x):.{sig}g}'
def mean_sd(d): return f"{fmt(d['mean'])} ± {fmt(d['sd'])}"
def ref(text):
    q=doc.add_paragraph(); q.paragraph_format.left_indent=Inches(.18); q.paragraph_format.first_line_indent=Inches(-.18); q.paragraph_format.space_after=Pt(3)
    r=q.add_run(text); r.font.size=Pt(8.2); return q

# PAGE 1 — cover.
para=doc.add_paragraph(); para.paragraph_format.space_before=Pt(76); para.alignment=WD_ALIGN_PARAGRAPH.LEFT
r=para.add_run('AF_PIKANs'); r.bold=True; r.font.size=Pt(13); r.font.color.rgb=RGBColor.from_string(TEAL)
doc.add_paragraph('Brief de avance\nde investigación',style='Title')
doc.add_paragraph('Baselines, señales cuantificadas y ruta hacia evidencia comparable con SOTA',style='Subtitle')
para=doc.add_paragraph(); para.paragraph_format.space_before=Pt(22)
r=para.add_run('CORTE DE EVIDENCIA  ·  23 SEP 2026'); r.bold=True; r.font.size=Pt(10); r.font.color.rgb=RGBColor.from_string(TEAL)
para=doc.add_paragraph(); para.paragraph_format.space_before=Pt(16); para.paragraph_format.space_after=Pt(18)
r=para.add_run('2 campañas GPU revisadas     24 corridas PDE     2 problemas con baseline exploratorio'); r.font.size=Pt(11); r.font.color.rgb=RGBColor.from_string(NAVY)
callout('ESTADO','Exploratorio. No se reclama superioridad ni resultado SOTA. Las métricas están bajo contratos experimentales distintos; EXP-0006 no se ha enviado y Allen–Cahn/DFG aún no están desbloqueados.',fill='FFF3E9')
para=doc.add_paragraph(); para.paragraph_format.space_before=Pt(80)
r=para.add_run('Preparado desde datos y artefactos ya guardados en AF_PIKANs.\nSin entrenamiento nuevo ni uso de GPU para este brief.'); r.font.size=Pt(9); r.font.color.rgb=RGBColor.from_string(GREY)
doc.add_page_break()

# PAGE 2 — executive snapshot.
doc.add_heading('1. Lectura ejecutiva',level=1)
p('El trabajo ya produjo dos capas de evidencia en Heat y Burgers: un baseline RBF-PIKAN fijo con Sobol y un screening de tres arquitecturas. Los resultados muestran que el desempeño cambia por PDE: B-spline fue mejor en Heat y RBF-PIKAN en el Burgers transitorio del screening. No hay una arquitectura ganadora general.')
p('El análisis de señales cuantificó una asociación útil, aunque todavía post hoc: en Burgers, los puntos del decil superior del residual del RBF concentraron entre 2.60× y 4.84× el error absoluto medio. En Heat la concentración fue menor (1.21×–1.88×). Esta señal motiva probar RAD de forma controlada; no demuestra que muestrear adaptativamente vaya a mejorar el modelo.')
callout('LÍMITE DE INFERENCIA','Cada brazo del screening tiene 3 semillas. Las dispersiones describen este piloto, no una confirmación estadística ni comparabilidad directa con resultados publicados. Los presupuestos de parada efectivos difirieron entre arquitecturas.')
doc.add_heading('Qué está hecho',level=2)
for x in ['EXP-0002: 6 corridas PDE, 3 semillas por problema, dos T4 en paralelo, 6/6 sincronizadas con W&B.','EXP-0005: 18 corridas PDE, 3 arquitecturas × 2 problemas × 3 semillas, 18/18 sincronizadas con W&B.','Se guardaron pools Sobol, checkpoints, métricas por semilla, señales internas y telemetría. Validación: 129×65 en Heat y Burgers estacionario; 1025×129 para Burgers transitorio EXP-0005.','Se hizo análisis CPU candidato para la referencia de Allen–Cahn.']: bullet(x)
doc.add_heading('Qué no está hecho',level=2)
for x in ['No hay umbrales físicos/de costo firmados ni una campaña confirmatoria concluida.','RAD vs Sobol (EXP-0006) está preparado con diez jobs planeados, pero no se envió: faltan firma científica y freeze del baseline/umbrales. El bloqueo fail-closed está implementado en el notebook y en cada worker y sus pruebas pasan.','Allen–Cahn carece de criterio operacional completo de convergencia. DFG 2D-1 ya tiene geometría, BC y QoIs canónicos; faltan verificador FEM independiente y cabeza vectorial antes de puntuar el modelo.']: bullet(x)
callout('DECISIÓN DE DISEÑO','La unidad de investigación es el problema: elegir y validar focalización por las señales cuantificadas de cada PDE, en lugar de asumir que un indicador global servirá igual para Heat, Burgers, Allen–Cahn y DFG.')
doc.add_page_break()

# PAGE 3 — experiment inventory and controls.
doc.add_heading('2. Experimentos y protocolo ejecutado',level=1)
rows=[
['EXP-0002 · baseline','Heat en varilla + Burgers estacionario tanh','RBF-PIKAN fija; Sobol aleatorizado y pools fijos; 3 semillas por PDE','6 corridas PDE; 2 T4; 6/6 W&B','Exploratorio; no pasa gates científicos'],
['EXP-0005 · architecture screen','Heat + Burgers transitorio canónico','RBF-PIKAN, MLP tanh y PIKAN B-spline; mismos pools pareados; 3 semillas','18 corridas PDE; 2 T4; 18/18 W&B','Exploratorio; capacidad cercana (~3.65k parámetros), costo/parada desigual'],
['EXP-0006 · RAD vs Sobol','Burgers transitorio canónico','Cinco semillas pareadas, un cambio de muestreo, 2 T4 planeadas','Propuesta 10 jobs; aún sin envío','Firma/freeze pendientes; guardas fail-closed probadas'],
['Allen–Cahn · estudio de referencia','Difusión-reacción 1D','CPU: diferencias finitas centradas + BDF; mallas 513/1025/2049','Sin entrenamiento PIKAN','Referencia candidata, criterio de convergencia incompleto'],
['DFG 2D-1 · cilindro Re=20','Navier–Stokes incompresible 2D','Contrato y QoIs en FEATFLOW; falta verificador FEM independiente y cabezal vectorial','No hay corrida evaluada','Pendiente de implementación y verificación']]
table(['Estudio','Problema','Diseño/recursos','Registro','Estado'],rows,[1.05,1.18,2.15,1.02,1.48],7.5)
doc.add_heading('Instrumentación y puntos',level=2)
p('El baseline EXP-0002 usa Sobol scrambled, pools reproducibles almacenados y diagnóstico independiente. Por semilla: 4,096 puntos interiores para Heat; 8,192 para Burgers; 512 puntos por frontera, 1,024 iniciales y un pool diagnóstico de hasta 2,048. Su malla de evaluación independiente es 129×65. EXP-0005 conserva 129×65 para Heat y usa 1025×129 para Burgers transitorio, con más resolución en el frente. Se midieron discrepancia centrada, distancias entre vecinos, pérdidas, pasos/tiempo, derivadas por autodiferenciación, residual, cobertura RBF, saturación, sensibilidad de salida, GPU y memoria.')
p('El modelo RBF fijo usa arquitectura [2,16,16,1], 3,648 parámetros, 11 bases gaussianas por arista y centros/anchos congelados; float64, Adam lr=0.001. Límites: Heat 5,000 pasos o 300 s; Burgers 20,000 pasos o 1,200 s. La rutina registra qué límite detuvo cada corrida. La decisión completa de parámetros está en el manifiesto y en parameter-decisions.md del experimento.')
callout('REPRODUCIBILIDAD','Se conservaron códigos de campaña, semillas, pools, artefactos de W&B y hashes. Los resultados por semilla prevalecen sobre promedios de campaña para cualquier auditoría.')
doc.add_page_break()

# PAGE 4 — problem contracts.
doc.add_heading('3. Qué significa cada problema',level=1)
rows=[
['Heat · sólido / varilla','uₜ = αuₓₓ; α=0.1; x,t∈[0,1]. IC sin(πx), extremos a temperatura cero.','Referencia analítica: sin(πx)e^(−απ²t).','EXP-0002 y EXP-0005. Conducción en sólido; no fluido ni convección. [9]'],
['Burgers · histórico EXP-0002','uₜ + uuₓ − νuₓₓ=0; ν=0.02; onda estacionaria tanh, A=0.5, centro x₀=−0.25, c=0.','Solución exacta tanh del contrato local.','Baseline histórico. No es el benchmark transitorio estándar.'],
['Burgers · transitorio EXP-0005','uₜ+uuₓ−(0.01/π)uₓₓ=0; x∈[−1,1], t∈[0,1]; u(x,0)=−sin(πx), bordes cero.','Cole–Hopf; contrato compartido con tarea canónica PINNacle. [1,2]','18 corridas de screening. La tabla de resultados se mantiene separada de EXP-0002.'],
['Allen–Cahn · candidato','uₜ−0.001uₓₓ−5(u−u³)=0. Contrato de referencia en revisión.','BDF implícito + diferencias centradas; malla refinada. [13]','Solo referencia CPU candidata; ningún baseline de red.'],
['DFG / Navier–Stokes · 2D-1','Flujo incompresible alrededor de cilindro, Re=20; caso estacionario.','FEATFLOW / Schäfer–Turek define el contrato y los QoIs. Targets: Cd=5.579535, Cl=0.010619, Δp=0.117520. [10,11]','Turek et al. (2010) reporta otro caso Re=20 en un manuscrito FEM 3D; no es un target intercambiable con 2D-1. Falta FEM independiente. [12]']]
table(['Problema','Definición/condiciones','Referencia científica','Estado y alcance'],rows,[1.22,2.45,1.62,1.59],7.6)
doc.add_heading('Campo Burgers inspeccionado con PyVista',level=2)
image('figure_04_burgers_pyvista.png',6.65)
caption('EXP-0002, seed 101, malla independiente 129×65. Los campos de referencia/predicción se muestran como superficies; error y residual se proyectan sobre (x,t). Para el mapa logarítmico se representa log10(|error|), con piso de visualización 10⁻¹²; no cambia las métricas.')
callout('CONTRATO, NO ETIQUETA','Burgers transitorio es un modelo escalar de transporte no lineal con difusión viscosa. Sirve de puente hacia NS, pero no contiene el vector velocidad-presión ni las condiciones geométricas de un flujo 2D.')
doc.add_page_break()

# PAGE 5 — results.
doc.add_heading('4. Resultados cuantitativos',level=1)
doc.add_heading('EXP-0002 · baseline RBF-PIKAN',level=2)
rows=[]
for problem,label in [('heat','Heat · varilla'),('burgers','Burgers · onda estacionaria tanh')]:
    d=DATA['exp0002'][problem]
    front=mean_sd(d['front_field_rmse_normalized']) if 'front_field_rmse_normalized' in d else '—'
    rows.append([label,'3',mean_sd(d['field_relative_l2']),mean_sd(d['residual_rmse_abs']),front,f"{d['runtime_sec']['mean']:.1f} ± {d['runtime_sec']['sd']:.1f}"])
table(['Contrato','n','Error rel. L2','RMSE residual PDE','RMSE frente normalizado','Tiempo medio (s)'],rows,[1.56,.35,1.23,1.34,1.31,1.1],7.8)
p('Heat resolvió una varilla 1D con un modo analítico suave. Burgers aquí es un perfil estacionario tanh; error y residuo son bajo ese contrato y su propia solución exacta. Ninguno de estos valores es un benchmark transferible a otros parámetros o a Burgers transitorio.')
doc.add_heading('EXP-0005 · screening de arquitecturas',level=2)
rows=[]
for model,d in DATA['exp0005'].items():
    h=d['problems']['heat']; b=d['problems']['burgers']
    rows.append([d['label'],str(d['parameters']),mean_sd(h['field_relative_l2']),f"{h['runtime_sec']['mean']:.0f} s / {h['steps']['mean']:.0f}",mean_sd(b['field_relative_l2']),mean_sd(b['front_field_rmse_normalized']),f"{b['runtime_sec']['mean']:.0f} s / {b['steps']['mean']:.0f}"])
table(['Arquitectura','Parámetros','Heat: L2 rel.','Heat: tiempo/pasos','Burgers: L2 rel.','Burgers: frente norm.','Burgers: tiempo/pasos'],rows,[1.28,.63,1.02,1.08,1.0,1.12,1.18],7.1)
image('figure_01_architecture_errors.png',6.65)
caption('Puntos = valores por semilla; rombo y bigote = media ± desviación estándar muestral. Escala vertical logarítmica. EXP-0005; errores de Burgers bajo la formulación transitoria canónica.')
p('Lectura: B-spline obtuvo el menor L2 medio de Heat (2.30×10⁻⁵); RBF obtuvo el menor L2 de Burgers (5.40×10⁻²) y error del frente (0.292). La B-spline llegó al límite de 300 s en Heat con ≈4,462 pasos y a 1,200 s en Burgers con ≈11,197 pasos; RBF y MLP llegaron a los máximos de pasos en estos runs. Por ello, esta tabla es un screening de costo-calidad, no un contraste a presupuesto computacional idéntico.')
doc.add_page_break()

# PAGE 6 — signals.
doc.add_heading('5. Señales medidas y qué permiten concluir',level=1)
p('Señal significa una medición cuantitativa candidata a explicar error, estabilidad o costo. Los datos siguientes describen los checkpoints y puntos guardados; fueron analizados después de las corridas y no se usaron para escoger puntos durante EXP-0002/0005.')
image('figure_02_residual_localization.png',6.65)
caption('En cada semilla, rango de Spearman entre |residual| y |error| y razón entre el error medio del decil de mayor residual y el error medio total. Área interior de validación; cálculo post hoc.')
rows=[]
for model,d in DATA['signals']['localization'].items():
    for prob,plabel in [('heat','Heat'),('burgers','Burgers')]:
        q=d[prob]
        rows.append([LABELS[model],plabel,f"{q['spearman_mean']:.3f} ({min(q['spearman_values']):.3f}–{max(q['spearman_values']):.3f})",f"{q['enrichment_mean']:.2f}× ({min(q['enrichment_values']):.2f}–{max(q['enrichment_values']):.2f})"])
table(['Arquitectura','PDE','ρ Spearman promedio (rango semillas)','Error top-decile / total (promedio; rango)'],rows,[1.35,.68,2.43,2.55],7.5)
rows=[
['Pool y cobertura','Sobol fijo: 4,096 Heat / 8,192 Burgers. Discrepancia centrada media: 5.27×10⁻⁸ Heat; 1.39×10⁻⁸ Burgers. Distancia mediana al vecino: 0.00970 / 0.00764 en coord. normalizadas.'],
['Error global vs frente','RBF en EXP-0005: Burgers L2=0.0540±0.0393; frente normalizado=0.292±0.229. Reportar ambos evita ocultar la región crítica con la media de dominio.'],
['Reproducibilidad del hot spot','RBF Burgers: Jaccard del top-decile residual entre pares de semillas 0.179–0.260. Hay concentración del error, con solapamiento espacial limitado/modesto.'],
['Autodiferenciación y residuo','Se guardan u, uₓ, uₜ, uₓₓ y r en cada diagnóstico. RMSE residual evaluado por separado: Heat RBF 9.22×10⁻⁴ (EXP-0002); Burgers transitorio RBF 6.06×10⁻² (EXP-0005), bajo contratos distintos.'],
['Internas del modelo','En RBF, cobertura efectiva media de base en capas 1/2: Heat ≈58.5/54.3; Burgers ≈89.2/111.4. Se miden además ocupación, sensibilidad ‖∂u/∂θ‖ y saturación; todavía son descriptivos, sin relación causal demostrada.']]
table(['Familia de señal','Medición reportada'],rows,[1.42,5.6],7.6)
callout('SEÑAL PRIORIZABLE','El residual es una señal prometedora para Burgers, pero sus correlaciones varían por arquitectura y problema. Las relaciones más altas en Burgers aparecen también con MLP/B-spline; el próximo experimento debe aislar la política de muestreo con pesos, arquitectura y referencias constantes.')
doc.add_page_break()

# PAGE 7 — readiness and what is blocking.
doc.add_heading('6. Madurez por problema y brechas de evidencia',level=1)
rows=[
['Heat','Baseline RBF y screening de 3 arquitecturas, n=3 cada una; solución analítica y malla independiente.','Prueba suave; revisar mayor resolución/evaluación de generalización, congelar umbrales y confirmación.','No implica conducción con fluido ni benchmark térmico de materiales heterogéneos.'],
['Burgers 1D','Dos contratos documentados: estacionario histórico y transitorio canónico. Screening transitorio compara arquitectura.','EXP-0006: RAD vs Sobol, 5 pares, solo después de firma y fijar métrica primaria, guardas, costo.','Tamaño n=5 puede no tener potencia; reportar pares/IC y fallos.'],
['Allen–Cahn','Referencia CPU: diferencias relativas de campo 1.17×10⁻⁴ y 6.27×10⁻⁵ entre refinamientos; BDF más estricto 1.26×10⁻⁸. IC/BC y monotonía de energía pasan checks.','Definir “efecto” del criterio <10%, resolver convergencia operacional y agregar QoIs/test antes de desbloquear un baseline.','Referencia candidata, no declarada científicamente convergida.'],
['DFG / NS 2D','FEATFLOW fija el contrato 2D-1 y publica Cd=5.57953523384, Cl=0.010618948146 y Δp=0.11752016697. El informe Turek et al. 2010 contiene una tabla Re=20 bajo un manuscrito FEM Level-Set 3D; sus cifras no son directamente comparables con 2D-1. Trainer actual es escalar (x,t). [10–12]','Implementar FEM 2D independiente; validar geometría, BC, convención de fuerza y QoIs; crear campo vectorial (u,v,p).','No hay score PIKAN. No mezclar la tabla Re=20 del informe 3D con el target canónico 2D-1.'],
['Poisson','En el alcance anterior se dejó pendiente.','Mantener como posible control elíptico; añadir solo si distingue una hipótesis de tesis.','No se incluye entre las 4 líneas principales ahora: Heat, Burgers, Allen–Cahn y DFG.']]
table(['Problema','Evidencia disponible','Siguiente gate científico','Reserva'],rows,[.85,2.24,2.55,1.38],7.2)
doc.add_heading('Por qué aún no corresponde decir SOTA',level=2)
for x in ['El protocolo comparativo exige contrato, referencia, métricas y recursos compatibles; PDEBench, PDENNEval y PINNacle sirven de guía de evaluación, no de validación de nuestro resultado. [2–4]','El estudio publicado MLP–KAN destaca sensibilidad a arquitectura, semillas y presupuesto. El screening local confirma heterogeneidad por PDE y muestra parada desigual. [6]','EXP-0002/0005 tienen n=3, no preregistro confirmatorio firmado ni umbral de aceptación aprobado; evaluadores locales registran scientific_accepted=false / thresholds_pass=false.','Una afirmación SOTA requiere comparador fuerte (p. ej., MLP/PINN de referencia y una arquitectura PIKAN reciente), presupuesto común, referencia independiente, intervalos por semilla y artefactos reproducibles.']: bullet(x)
doc.add_page_break()

# PAGE 8 — plan.
doc.add_heading('7. Ruta recomendada para avanzar',level=1)
rows=[
['A · Cerrar lectura del baseline','Auditar errores por semilla, front mask, residuo físico/normalizado y costo; reportar sensibilidad a cuadrícula para no depender de una sola malla.','CPU; datos ya guardados.'],
['B · Congelar y firmar','Congelar métrica primaria (log ratio pareado de L2), guardas físicas, coste máximo y regla de faltantes; versionar firma/prerregistro y verificar las guardas fail-closed ya implementadas.','Requisito humano previo a EXP-0006.'],
['C · Intervención única','Cinco parejas nuevas; misma RBF, init/pool común y presupuesto; solo cambia Sobol fijo ↔ RAD acotado. Guardar ESS, puntos únicos, gasto de AD para refresco y telemetría.','Dos T4; diez jobs, un brazo por slot.'],
['D · Leer el resultado','Paired effect + IC t del 95%, todos los puntos individuales, coste y guardas. RAD avanza solo si el efecto útil predeclarado aparece y no rompe condiciones físicas/costo.','No convertir el piloto automáticamente en aceptación.'],
['E · Confirmar y comparar','Repetir/expandir semillas según variabilidad y potencia; añadir comparador MLP/PINN o PIKAN reciente bajo budget igualado y validación exacta común.','Aquí comienza evidencia comparable con SOTA.'],
['F · Desbloquear problemas','Cerrar Allen–Cahn reference y después DFG: contrato, verificador, solución/FEM y QoIs; medir señales adaptadas al frente/interfaz o a wake/cilindro, respectivamente.','Un contrato por problema; no transferir señales sin validación.']]
table(['Fase','Trabajo verificable','Recursos/gate'],rows,[1.22,4.43,1.37],7.5)
image('figure_03_cost_error.png',6.5)
caption('EXP-0005: cada punto es una corrida por PDE; forma distingue problema, color arquitectura. Tiempo y L2 en escala logarítmica. La dispersión temporal muestra por qué hay que equalizar presupuesto antes de atribuir eficiencia.')
callout('ESTADO DEL PRÓXIMO RUN','EXP-0006 no se envió: el prerregistro y el freeze de umbrales aún esperan firma científica. El notebook y cada worker aplican el bloqueo fail-closed; las pruebas verifican que no se despachen tareas ni se cree salida mientras el manifiesto siga sin firma. Este brief solo documenta resultados guardados; no consume GPU.')
doc.add_page_break()

# PAGE 9 — toolchain.
doc.add_heading('8. Herramientas de visualización por etapa',level=1)
p('La pila combina visualización de evidencia numérica existente con herramientas especializadas para las siguientes familias de experimentos. Las bibliotecas de etapas futuras se registran como opciones, no como dependencias obligatorias del baseline actual.')
rows=[
['PDE / SciML / materiales / proteínas','PyVista','Renderizado científico de campos/mallas, exportación off-screen. Esta entrega añade figura 4-panel desde NPZ guardado; útil cuando DFG y geometrías 2D/3D estén listos. [16]'],
['Partículas / Brownian / artificial life','VisPy o Taichi','VisPy para inspección interactiva científica Python; Taichi si la simulación visual requiere kernels acelerados/partículas. No hay carga de partículas actual. [19,20]'],
['SciML matemático serio','Julia + Makie','Opción si una etapa o referencia se implementa en Julia; Makie tiene paquete citable de visualización científica. El trainer actual continúa en PINA/PyTorch. [21]'],
['Notebooks interactivos rápidos','Plotly','Curvas y heatmaps con hover. El brief genera HTML complementario; la campaña anterior ya produjo dashboards por problema. [17]'],
['Bioimaging / células / volúmenes','napari','Especialista para arreglos de imagen n-dimensional. Usar cuando la tesis incorpore imagen volumétrica o bioimaging, no para campos escalares 1D. [22]'],
['Simulación como aplicación','Panel + PyVista / Plotly','Dashboard local para revisión posterior; la app Panel del repo ya existe. Conectar solo outputs versionados y estados de ejecución. [18]'],
['Visual experimental en navegador','Three.js/WebGPU o Taichi','Prototipo de interacción visual cuando exista una pregunta científica concreta; WebGPU aún se trata como plataforma web emergente. No usar para sustituir el evaluador numérico. [20,23]']]
table(['Etapa','Herramienta','Uso y estado en AF_PIKANs'],rows,[1.55,1.42,4.05],7.4)
doc.add_heading('Producción de esta entrega',level=2)
for x in ['PyVista: figura de predicción, referencia, error log10 y residuo desde validation.npz EXP-0002.','Plotly: dashboard HTML con resumen comparativo, errores, costo, localización residual y campo de Burgers; se puede abrir en navegador sin servicio.','Matplotlib: PNGs estáticos de error L2, señales y coste para integrar al DOCX.','Panel: `src/research_pipeline/remote/analysis_dashboard.py` y documentación ya están en repo; no se ejecutó GPU ni se modificó la app.']: bullet(x)
doc.add_page_break()

# PAGE 10+ — bibliography.
doc.add_heading('9. Bibliografía consultada',level=1)
p('La bibliografía separa publicaciones científicas, contratos/benchmarks PDE y documentación técnica. Los preprints se identifican como tales y no sustituyen validación ni implican superioridad. El BibTeX y las notas por fuente quedan en PIKAN-Vault/PIKAN-Vault/sources/AF_PIKANs_Research_Brief_References_2026-09-23.bib y 00_INDEX.md.')
doc.add_heading('Literatura científica',level=2)
refs=[
'[1] M. Raissi, P. Perdikaris y G. E. Karniadakis (2019). “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs.” Journal of Computational Physics 378, 686–707. DOI: 10.1016/j.jcp.2018.10.045. Vault: `papers/Raissi_2019_PINNs.md`.',
'[2] Z. Hao et al. (2024). “PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs.” NeurIPS Datasets and Benchmarks. arXiv:2306.08827. Vault: `papers/Hao_2024_PINNacle.md`.',
'[3] M. Takamoto et al. (2022). “PDEBench: An Extensive Benchmark for Scientific Machine Learning.” NeurIPS Datasets and Benchmarks. arXiv:2210.07182. Vault: `papers/Takamoto_2022_PDEBench.md`.',
'[4] P. Wei et al. (2024). “PDENNEval: A Comprehensive Evaluation of Neural Network Methods for Solving PDEs.” IJCAI 2024, 5181–5189. DOI: 10.24963/ijcai.2024/573. Vault: `papers/Wei_2024_PDENNEval.md`.',
'[5] C. Wu et al. (2023). “A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks.” Computer Methods in Applied Mechanics and Engineering 403, artículo 115671. DOI: 10.1016/j.cma.2022.115671. arXiv:2207.10289 (versión previa). Vault: `papers/Wu_2022_Adaptive_Sampling_PINNs.md`.',
'[6] K. Shukla, J. D. Toscano, Z. Wang, Z. Zou y G. E. Karniadakis (2024). “A comprehensive and FAIR comparison between MLP and KAN representations for differential equations and operator networks.” Computer Methods in Applied Mechanics and Engineering 431, artículo 117290. DOI: 10.1016/j.cma.2024.117290. arXiv:2406.02917 (versión previa). Vault: `papers/Shukla_2024_cPIKAN_Benchmark.md`.',
'[7] B. Jacob, A. A. Howard y P. Stinis (2025). “SPIKANs: Separable Physics-Informed Kolmogorov–Arnold Networks.” Machine Learning: Science and Technology 6(3), 035060. DOI: 10.1088/2632-2153/ae05af. Vault: `papers/Jacob_2025_SPIKANs.md`.',
'[8] S.-T. Chiu et al. (2026). “Free-RBF-KAN: Kolmogorov-Arnold Networks with Adaptive Radial Basis Functions for Efficient Function Learning.” arXiv:2601.07760 (preprint; arquitectura distinta del RBF fijo local). Vault: `papers/Chiu_2026_Free_RBF_KAN.md`.'
]
for text in refs: ref(text)
doc.add_heading('Contratos PDE y benchmarks',level=2)
refs=[
'[9] MIT OpenCourseWare, 18.085. “Problem Set 8 Solutions.” Fuente docente de contexto para la ecuación de calor 1D y su solución por modos. Vault: `sources/PDE_Contracts_Burgers_AllenCahn_DFG.md`.',
'[10] FEATFLOW. “DFG benchmark 2D-1 (Re=20, laminar).” Define el caso estacionario, geometría, condiciones de frontera, Cd, Cl, Δp y targets de referencia. Vault: `sources/PDE_Contracts_Burgers_AllenCahn_DFG.md`.',
'[11] M. Schäfer, S. Turek, F. Durst, E. Krause y R. Rannacher (1996). “Benchmark Computations of Laminar Flow Around a Cylinder.” En *Flow Simulation with High-Performance Computers II*, pp. 547–566. DOI: 10.1007/978-3-322-89849-4_39. Vault: `sources/PDE_Contracts_Burgers_AllenCahn_DFG.md`.',
'[12] S. Turek, O. Mierka, S. Hysing y D. Kuzmin (2010). “A high order 3D FEM-Level Set approach for multiphase flows with application to monodisperse droplet generation.” Ergebnisberichte des Instituts für Angewandte Mathematik, TU Dortmund, no. 411. §5.1/Table I reports a Re=20 cylinder case in a 3D FEM-Level Set context; its values are not interchangeable with the DFG 2D-1 target without reconciling case definition and normalization. Vault: `sources/PDE_Contracts_Burgers_AllenCahn_DFG.md`.',
'[13] DeepXDE (2024). Ejemplo documentado del problema forward Allen–Cahn. Documentación v1.12.2. Vault: `sources/PDE_Contracts_Burgers_AllenCahn_DFG.md`.'
]
for text in refs: ref(text)
doc.add_heading('Frameworks y documentación técnica',level=2)
refs=[
'[14] PINA Developers. Documentación oficial de PINA, framework SciML sobre PyTorch. Consultada 2026-09-23. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[15] Weights & Biases. Documentación de experiment tracking y artefactos. Consultada 2026-09-23. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[16] PyVista Developers. Guía y configuración off-screen de PyVista. Documentación oficial consultada 2026-09-23. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[17] Plotly. Documentación oficial de Plotly.py. Consultada 2026-09-23. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[18] HoloViz. Documentación oficial de Panel. Consultada 2026-09-23. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[19] VisPy Developers. Documentación oficial. Consultada 2026-09-23. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[20] Taichi Developers. Documentación de GUI / visualización de simulación. Consultada 2026-09-23. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[21] S. Danisch y J. Krumbiegel (2021). “Makie.jl: Flexible high-performance data visualization for Julia.” Journal of Open Source Software 6(65), 3349. DOI: 10.21105/joss.03349. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[22] napari contributors. Documentación oficial de napari viewer para arreglos n-dimensionales. Consultada 2026-09-23. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.',
'[23] World Wide Web Consortium (2026). WebGPU, Candidate Recommendation Draft, 15 septiembre 2026. Vault: `sources/Visualization_Toolchain_References_2026-09-23.md`.'
]
for text in refs: ref(text)
callout('REGISTRO BIBLIOGRÁFICO','Se corrigieron los DOI/versiones publicadas de Shukla y Wu, se añadió el capítulo canónico Schäfer–Turek, y se tituló correctamente el informe 411 de Turek et al. La tabla de ese informe se conserva con su alcance FEM 3D, separada de la referencia objetivo DFG 2D-1. Las entradas, notas y BibTeX están en la bóveda.')
p('Archivos: DOCX en docs/; scripts, figuras, HTML y JSON en var/agent_runs/research-progress-brief-20260923/; datos fuente en var/agent_runs/; BibTeX en PIKAN-Vault/PIKAN-Vault/sources/AF_PIKANs_Research_Brief_References_2026-09-23.bib. Maquetación revisada página por página desde Word.')

DOC_PATH.parent.mkdir(parents=True,exist_ok=True)
doc.save(DOC_PATH)
print(str(DOC_PATH))







