import json
from pathlib import Path
p=json.loads(Path('var/agent_runs/integracion-zmain-2026-09-07/tesis-paginas.json').read_text(encoding='utf-8'))
for i,t in enumerate(p):
 if i>30 and any(w in t for w in ['Tabla 4:', 'Tabla 5:', 'Figura 24:', 'Figura 40:', 'Figura 41:', 'Figura 42:', 'Figura 43:', 'Figura 46:', 'Tabla 7:', 'Tabla 8:']):
  print('\nPDF',i+1,'\n',t[:10500])
