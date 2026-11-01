"""Check assembled review artifacts without changing canonical thesis sources."""
from pathlib import Path
import hashlib, json, re
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
checks = {}
snapshot = json.loads((HERE / 'source-snapshot.json').read_text())
checks['canonical_sources_unchanged'] = all(
    hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest
    for name, digest in snapshot.items()
)
required = ['architecture', 'pinn-loss', 'heat-rod'] + [f'p{k}-problem' for k in range(1, 6)]
checks['all_physical_and_architecture_figures'] = all(
    (HERE / 'illustrations' / (name + '.pdf')).is_file() for name in required
)
checks['all_solution_figures'] = all(
    (HERE / ('solutions' if k < 4 else 'dfg' if k == 4 else 'pvdf') / f'p{k}-solution.pdf').is_file()
    for k in range(1, 6)
)
checks['all_numerical_evidence_present'] = all(
    (HERE / folder / 'evidence.json').is_file() for folder in ['solutions', 'dfg', 'pvdf']
)
checks['numerical_percentages_escaped'] = all(
    re.search(r'\d%', path.read_text(encoding='utf-8')) is None
    for folder in ['dfg', 'pvdf'] for path in (HERE / folder).glob('*.tex')
)
pdfs = {}
for name in ['atlas-figuras', 'marco-teorico-ilustrado', 'tesis-revision']:
    path = HERE / (name + '.pdf')
    reader = PdfReader(path)
    pdfs[name] = {'pages': len(reader.pages), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    checks[name + '_navier_stokes_named'] = 'Navier' in text and 'Stokes' in text
    checks[name + '_no_undefined_references'] = '??' not in text
log = (HERE / 'tesis-revision.log').read_text(errors='replace')
checks['latex_no_unresolved_citations'] = 'There were undefined references' not in log and 'Citation ' not in log
checks['latex_no_overfull_boxes'] = 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log
result = {'checks': checks, 'pdfs': pdfs, 'all_passed': all(checks.values()),
          'scope': 'Artifact integrity and build checks; numerical diagnostics remain in each evidence.json.'}
(HERE / 'review-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
if not result['all_passed']:
    raise SystemExit(1)
