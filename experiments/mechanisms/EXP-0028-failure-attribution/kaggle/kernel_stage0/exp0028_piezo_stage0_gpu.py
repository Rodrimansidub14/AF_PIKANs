"""EXP-0028 piezo, etapa 0: medicion de costo en T4 (s/paso, recalculo, concurrencia, CUDA graphs). Ejecuta stage0.py del bundle. No entrena modelos completos."""
import os
import subprocess
import sys
from pathlib import Path

bundle = next(Path("/kaggle/input").rglob("stage0.py")).parent
out = Path("/kaggle/working/stage0")
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, BUNDLE_DIR=str(bundle))
rc = subprocess.run([sys.executable, str(bundle / "stage0.py"), "--out", str(out)], cwd=str(bundle), env=env).returncode
print("stage0 rc", rc, flush=True)
