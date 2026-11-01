"""Infrastructure-only contract probe. These are NOT PIKAN research results."""
import json
import os
from pathlib import Path

run = Path(os.environ['HARNESS_RUN_DIR'])
seed = int(os.environ['HARNESS_SEED'])
# A measured arithmetic identity checks JSON/environment transport only.
error = abs(sum([0.25] * 4) - 1.0)
(run / 'metrics.json').write_text(json.dumps({'error': error}), encoding='utf-8')
(run / 'decision_note.md').write_text(
    f'Infrastructure smoke, seed {seed}. No training or scientific claim.\n', encoding='utf-8')
