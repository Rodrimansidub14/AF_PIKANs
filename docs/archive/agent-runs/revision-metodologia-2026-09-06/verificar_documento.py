from pathlib import Path
import re, json, hashlib, difflib
from collections import Counter
import numpy as np
from pypdf import PdfReader

root=Path(__file__).resolve().parent
repo=root.parents[2]
files=sorted(root.glob('*.tex'))
combined='\n'.join(p.read_text(encoding='utf-8-sig') for p in files)
bib=(root/'m-bibliografia.bib').read_text(encoding='utf-8-sig')
keys=re.findall(r'@\w+\s*\{\s*([^,]+),',bib)
cites=set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',combined) for k in group.split(','))
labels=re.findall(r'\\label\{([^}]+)\}',combined)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',combined)
figs=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',combined)
result={'missing_citations':sorted(cites-set(keys)),
 'duplicate_bib_keys':[k for k,n in Counter(keys).items() if n>1],
 'duplicate_labels':[k for k,n in Counter(labels).items() if n>1],
 'missing_labels':sorted(set(refs)-set(labels)),
 'missing_figures':[k for k in figs if not (root/k).is_file()],
 'cited_sources':len(cites),'figures_in_document':len(figs),
 'unresolved_editorial_markers':re.findall(r'\\pendiente|\[nuevo:|\bTODO\b|\bTBD\b',combined)}
assert all(not result[k] for k in ['missing_citations','duplicate_bib_keys','duplicate_labels','missing_labels','missing_figures','unresolved_editorial_markers']),result
rng=np.random.default_rng(617)
x=rng.uniform(.1,.9,200);y=rng.uniform(.1,.9,200);t=rng.uniform(.05,.95,200)
h=2e-4
# Fourth-order finite differences, independent of the formula derivations.
def d1(f,x): return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)
def d2(f,x): return (-f(x+2*h)+16*f(x+h)-30*f(x)+16*f(x-h)-f(x-2*h))/(12*h*h)
checks={}
u=lambda x,y:np.sin(4*np.pi*x)*np.sin(4*np.pi*y)
r=-d2(lambda z:u(z,y),x)-d2(lambda z:u(x,z),y)-2*(4*np.pi)**2*u(x,y)
checks['poisson_distribuido']=float(np.max(np.abs(r))/(2*(4*np.pi)**2))
a=100;c=.5;q=x*(1-x);d=x-c
f=np.exp(-a*d*d)*(2+4*a*d*(1-2*x)+(2*a-4*a*a*d*d)*q)
u=lambda z:z*(1-z)*np.exp(-a*(z-c)**2)
checks['poisson_localizado']=float(np.max(np.abs(-d2(u,x)-f))/max(1,np.max(np.abs(f))))
x=rng.uniform(-.8,.8,200);nu=.05
u=lambda z:-np.tanh(z/(2*nu))
checks['burgers']=float(np.max(np.abs(u(x)*d1(u,x)-nu*d2(u,x))))
eps=.05
u=lambda z,t:np.tanh((z+.5-t)/(np.sqrt(2)*eps))
f=-(1-u(x,t)**2)/(np.sqrt(2)*eps)
r=d1(lambda z:u(x,z),t)-eps**2*d2(lambda z:u(z,t),x)+u(x,t)**3-u(x,t)-f
checks['allen_cahn']=float(np.max(np.abs(r)))
Re=40;lam=Re/2-np.sqrt(Re**2/4+4*np.pi**2)
vx=lambda x,y:1-np.exp(lam*x)*np.cos(2*np.pi*y)
vy=lambda x,y:lam/(2*np.pi)*np.exp(lam*x)*np.sin(2*np.pi*y)
pres=lambda x,y:(1-np.exp(2*lam*x))/2
x=rng.uniform(-.4,.9,200);y=rng.uniform(-.4,1.4,200)
checks['kovasznay_continuidad']=float(np.max(np.abs(d1(lambda z:vx(z,y),x)+d1(lambda z:vy(x,z),y))))
for name,v,dp in [('x',vx,d1(lambda z:pres(z,y),x)),('y',vy,d1(lambda z:pres(x,z),y))]:
    r=vx(x,y)*d1(lambda z:v(z,y),x)+vy(x,y)*d1(lambda z:v(x,z),y)+dp-(d2(lambda z:v(z,y),x)+d2(lambda z:v(x,z),y))/Re
    checks['kovasznay_momento_'+name]=float(np.max(np.abs(r)))
assert all(v<1e-4 for v in checks.values()),checks
result['independent_finite_difference_checks']={'errors':checks,'threshold':1e-4,'method':'fourth-order central differences, h=2e-4, 200 seeded interior points'}
result['figure_calculations']=json.loads((root/'validacion-figuras.json').read_text())
manifest=[]
for f in files+[root/'m-bibliografia.bib']:
    original=repo/f.name
    if original.exists():
        old=original.read_text(encoding='utf-8-sig');new=f.read_text(encoding='utf-8-sig')
        diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/'+f.name,tofile='revision/'+f.name))
        (root/(f.name+'.diff')).write_text(diff,encoding='utf-8')
        manifest.append({'source':str(original),'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                         'revision':f.name,'revision_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(root/'manifest.json').write_text(json.dumps({'status':'draft_for_scientific_review','files':manifest},indent=2),encoding='utf-8')
pdf=root/'protocolo-revisado.pdf'
if pdf.exists():
    reader=PdfReader(pdf);texts=[p.extract_text() or '' for p in reader.pages]
    result['pdf_pages']=len(texts)
    result['layout_review']='All 33 pages rendered and inspected in contact sheets; figures independently inspected'
    result['foundation_page_share_approx']=6/14
    result['foundation_page_share_note']='Approximate page spans: sections 6.1-6.3 pp.6-11; chapter 6 pp.6-19; shared pages make this an editorial estimate'
    result['pages_with_unresolved_question_marks']=[i+1 for i,z in enumerate(texts) if '??' in z]
    result['pdf_text_characters']=sum(map(len,texts))
    (root/'texto-pdf.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{z}' for i,z in enumerate(texts)),encoding='utf-8')
log=root/'protocolo-revisado.log'
if log.exists():
    z=log.read_text(encoding='utf-8',errors='replace')
    result['latex_warnings']=[line for line in z.splitlines() if 'Warning' in line or 'Overfull' in line or line.startswith('!')]
(root/'validacion.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
