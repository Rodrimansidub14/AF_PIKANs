from pathlib import Path
import re,json,hashlib
import numpy as np
from pypdf import PdfReader
from scipy.interpolate import BSpline
ROOT=Path(__file__).resolve().parents[3];RUN=Path(__file__).resolve().parent
names=['z-main.tex','0-datos_estudiante.tex','1-opciones_adicionales.tex','2-paquetes_y_comandos_usuario.tex']
names += [p.name for p in ROOT.glob('[a-n]-*.tex')]
files=[ROOT/n for n in sorted(set(names))]+sorted((ROOT/'capitulos').rglob('*.tex'))
text='\n'.join(p.read_text(encoding='utf-8') for p in files)
active='\n'.join(re.sub(r'(?<!\\)%.*','',p.read_text(encoding='utf-8')) for p in files)
keys=set(re.findall(r'@\w+\{([^,]+),',(ROOT/'m-bibliografia.bib').read_text(encoding='utf-8')))
cites=set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',active) for k in group.split(','))
labels=re.findall(r'\\label\{([^}]+)\}',active)
refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',active))
assert not cites-keys,cites-keys
assert not refs-set(labels),refs-set(labels)
assert len(labels)==len(set(labels)), 'Duplicate labels'
assert not re.search(r'\\(?:textcolor|color)\{red\}|\bTODO\b|\[.*pendiente.*\]|placeholder',active)
figs=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{(revision2026/[^}]+)\}',active)
assert all((ROOT/'figuras'/f).exists() for f in figs)
for p in (ROOT/'capitulos').rglob('*.tex'):
    for fig in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',p.read_text(encoding='utf-8'),re.S):
        assert r'\cite{' in fig, f'Unattributed figure: {p}'
problems=re.findall(r'\\subsection\{P([1-5])\.',(ROOT/'capitulos/metodologia/01-problemas.tex').read_text(encoding='utf-8'))
assert problems==list('12345'),problems
theory='\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'capitulos/marco').glob('*.tex'))
assert not re.search(r'semillas|20000|\\subsection\{P[1-5]|H1,|\\widehat U_a',theory)
log=(ROOT/'z-main.log').read_text(encoding='utf-8',errors='replace')
bad=re.findall(r'^.*(?:undefined|multiply defined|Overfull|LaTeX Error|Rerun to get cross|Please \(re\)run Biber).*$',log,re.M)
assert not bad,bad
rng=np.random.default_rng(47);x=rng.uniform(-.95,.95,1000);t=rng.uniform(0,1,1000)
burg=[]
for c in [0,.5]:
 for nu in [.02,.005]:
    A=.5;q=A*(x-c*t+.25)/(2*nu);th=np.tanh(q);S=1-th**2
    u=c-A*th;ux=-A*A/(2*nu)*S;ut=-c*ux;uxx=A**3/(2*nu**2)*S*th
    residual=np.max(np.abs(ut+u*ux-nu*uxx));burg.append(float(residual));assert residual<1e-10
pvdfC=np.array([[2.1836e9,.6332e9,0],[.6332e9,2.1836e9,0],[0,0,.775e9]])
assert np.linalg.eigvalsh(pvdfC).min()>0
e=np.array([[0,0,0],[-.02904,-.05157,0]])
k=np.diag([1.062e-10,1.041e-10]);assert np.linalg.eigvalsh(k).min()>0
# Finite-difference derivatives of enthalpy independently recover both constitutive fields.
s=np.array([1e-5,-2e-5,3e-5]);Efield=np.array([2e4,1e5])
def enthalpy(strain,field):return .5*strain@pvdfC@strain-strain@e.T@field-.5*field@k@field
stress=pvdfC@s-e.T@Efield;electric=e@s+k@Efield
stress_fd=np.array([(enthalpy(s+np.eye(3)[i]*1e-8,Efield)-enthalpy(s-np.eye(3)[i]*1e-8,Efield))/2e-8 for i in range(3)])
electric_fd=-np.array([(enthalpy(s,Efield+np.eye(2)[i]*10)-enthalpy(s,Efield-np.eye(2)[i]*10))/20 for i in range(2)])
assert np.allclose(stress,stress_fd,rtol=1e-8,atol=1e-6)
assert np.allclose(electric,electric_fd,rtol=1e-8,atol=1e-12)
G=8;p=3;knots=np.r_[np.full(4,-1.05),np.linspace(-1.05,1.05,G+1)[1:-1],np.full(4,1.05)]
sp=BSpline(knots,np.eye(G+p),p);xx=np.linspace(-1.04,1.04,113)
partition=float(np.max(np.abs(sp(xx).sum(axis=1)-1)));assert partition<1e-12
dpartition=float(np.max(np.abs(sp.derivative()(xx).sum(axis=1))));assert dpartition<1e-12
assert (2*16+16*16+16)*12==3648
pdf=PdfReader(ROOT/'z-main.pdf');pages=[p.extract_text() or '' for p in pdf.pages]
pdftext='\n'.join(pages)
assert 'Daniel Gonzales' in pdftext and 'Gabriel Barrientos' in pdftext
assert '??' not in pdftext
assert not re.search(r'\[\d+\]\s*\[Nombre',pdftext)
out={'files':len(files),'pages':len(pages),'citations':len(cites),'equations':len(re.findall(r'\\begin\{(?:equation|align)\}',active)),'figures':len(figs),'canonical_problem_ids':problems,'missing_citations':sorted(cites-keys),'missing_refs':sorted(refs-set(labels)),'compile_issues':bad,'burgers_analytic_max_residuals':burg,'spline_partition_error':partition,'spline_derivative_partition_error':dpartition,'pvdf_C_min_eigenvalue':float(np.linalg.eigvalsh(pvdfC).min()),'scope':'Document and mathematical checks only; mechanism not trained'}
(RUN/'validacion.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
(RUN/'pdf-paginas.json').write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[ROOT/'m-bibliografia.bib',ROOT/'z-main.pdf']+[ROOT/'figuras'/f for f in figs]}
(RUN/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(out,indent=2,ensure_ascii=False))
for i,p in enumerate(pages,1):
 if any(s in p for s in ['Reproducción de la Figura','Cuadro 1:','Arquitectura propuesta','7.1.','6.1.']): print('INSPECCIONAR',i,p[:70].replace('\n',' '))
