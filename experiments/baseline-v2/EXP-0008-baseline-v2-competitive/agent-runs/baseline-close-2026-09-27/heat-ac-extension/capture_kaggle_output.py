import os, subprocess
from pathlib import Path
py = Path(r'var/agent_runs/remote-env/Scripts/python.exe')
out = Path(r'var/agent_runs/baseline-close-2026-09-27/heat-ac-extension/debug-output-captured')
out.mkdir(parents=True, exist_ok=True)
command = [str(py), '-m', 'kaggle', 'kernels', 'output', 'rodrimandub/af-pikans-baseline-heat-ac-20260927', '-p', str(out)]
proc = subprocess.run(command, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
Path(r'var/agent_runs/baseline-close-2026-09-27/heat-ac-extension/kaggle-output-stdout.txt').write_text(proc.stdout.decode('utf-8', 'replace'), encoding='utf-8')
Path(r'var/agent_runs/baseline-close-2026-09-27/heat-ac-extension/kaggle-output-stderr.txt').write_text(proc.stderr.decode('utf-8', 'replace'), encoding='utf-8')
print('exit_code=', proc.returncode, 'stdout_bytes=', len(proc.stdout), 'stderr_bytes=', len(proc.stderr))
