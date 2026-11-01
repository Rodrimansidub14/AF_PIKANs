from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[3]
shorts={'fig:fem':'Recreación de Poisson con FEM triangular.',
'fig:fem-orden':'Convergencia de la recreación FEM.',
'fig:arquitecturas':'Arquitecturas MLP y KAN: redibujo conceptual.',
'fig:perdida':'Construcción de la pérdida PINN.',
'fig:bases':'B-splines cúbicas y una base compacta ilustrativa.',
'fig:tesis-fem':'Desplazamiento vertical PINN--FEM en el antecedente piezoeléctrico.',
'fig:tesis-deformacion':'Deformación piezoeléctrica PINN--FEM en la tesis de referencia.',
'fig:canonicos':'Recreaciones de Burgers, Allen--Cahn y geometría DFG.',
'fig:mecanismo':'Arquitectura propuesta del controlador y su evaluación.'}
for p in (ROOT/'capitulos').rglob('*.tex'):
    text=p.read_text(encoding='utf-8')
    def sub(m):
        block=m[0];label=re.search(r'\\label\{([^}]+)\}',block)
        if label and label[1] in shorts and '\\caption[' not in block:
            block=block.replace('\\caption{','\\caption['+shorts[label[1]]+']{',1)
        return block
    text=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',sub,text,flags=re.S)
    p.write_text(text,encoding='utf-8')
p=ROOT/'var/agent_runs/integracion-zmain-2026-09-07/PROCEDENCIA.md'
p.write_text(p.read_text(encoding='utf-8').replace('[115 175 90 170]','[115 185 90 170]'),encoding='utf-8')
