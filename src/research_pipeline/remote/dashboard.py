"""Local, read-only monitor for the currently submitted Kaggle campaign."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / 'var' / 'agent_runs' / 'heat-burgers-001'
_CACHE = {'at': 0.0, 'value': None}


def _load_local_credentials():
    """Read only the allowlisted provider settings; never return or print them."""
    path = ROOT / '.env'
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if key in {'KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'WANDB_API_KEY', 'WANDB_ENTITY', 'WANDB_PROJECT'}:
            os.environ.setdefault(key, value.strip().strip('"\''))


def _tail(path: Path, limit: int = 40) -> list[str]:
    try:
        return path.read_text(encoding='utf-8', errors='replace').splitlines()[-limit:]
    except OSError:
        return []


def _campaign_paths():
    campaign = Path(os.environ.get('AF_PIKAN_CAMPAIGN', CAMPAIGN)).resolve()
    bundle = Path(os.environ.get('AF_PIKAN_BUNDLE', campaign / 'bundle')).resolve()
    status_script = Path(os.environ.get(
        'AF_PIKAN_STATUS_SCRIPT', ROOT / 'var' / 'agent_runs' / 'run_heat_burgers_remote.py'
    )).resolve()
    extracted_runs = campaign / 'download-extracted' / 'runs'
    if extracted_runs.is_dir():
        runs = extracted_runs
    else:
        candidates = [path for download in campaign.glob('download*') if download.is_dir()
                     for path in download.rglob('runs')
                     if path.is_dir() and any(path.glob('seed-*/run_summary.json'))]
        runs = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    return campaign, bundle, status_script, runs


def snapshot() -> dict:
    now = time.monotonic()
    if _CACHE['value'] is not None and now - _CACHE['at'] < 12:
        return _CACHE['value']

    _load_local_credentials()
    campaign, bundle, coordinator, runs_root = _campaign_paths()
    submission_path = bundle / 'submission.json'
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    submission = json.loads(submission_path.read_text(encoding='utf-8')) if submission_path.is_file() else {}
    submitted_at = datetime.fromtimestamp(submission_path.stat().st_mtime, timezone.utc) if submission_path.is_file() else None
    elapsed = max(0, (datetime.now(timezone.utc) - submitted_at).total_seconds()) if submitted_at else None

    status_text = 'not_submitted'
    error = None
    try:
        python = ROOT / 'var' / 'agent_runs' / 'remote-env' / 'Scripts' / 'python.exe'
        proc = subprocess.run([str(python), str(coordinator), 'status'], cwd=ROOT,
                              capture_output=True, text=True, timeout=25, check=False)
        if proc.returncode == 0:
            response = json.loads(proc.stdout)
            status_text = response.get('provider_status', 'unknown')
        else:
            error = 'No se pudo consultar Kaggle; reintentaré automáticamente.'
    except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
        error = 'No se pudo consultar Kaggle; reintentaré automáticamente.'

    run_rows = []
    if runs_root is not None:
        for path in runs_root.rglob('run_summary.json'):
            try:
                summary = json.loads(path.read_text(encoding='utf-8'))
                result = summary.get('result', {})
                run_rows.append({'seed': summary.get('seed'), 'status': result.get('status'),
                                 'elapsed_sec': result.get('elapsed_sec'),
                                 'metrics': result.get('metrics', {})})
            except (OSError, ValueError):
                continue
    logs = []
    if campaign.is_dir():
        for name in ('stdout.log', 'stderr.log'):
            for downloads in campaign.glob('download*'):
                if downloads.is_dir():
                    for path in list(downloads.rglob(name))[-6:]:
                        lines = _tail(path, 18)
                        if lines:
                            logs.append({'file': path.name, 'lines': lines})

    cfg = manifest.get('training', {})
    sync_path = campaign / 'wandb-sync.json'
    sync_status = json.loads(sync_path.read_text(encoding='utf-8')).get('status', 'unknown') \
        if sync_path.is_file() else 'unknown'
    result = {
        'kernel': submission.get('kernel') or manifest.get('iteration', {}).get('id'),
        'provider_state': status_text,
        'submission_state': submission.get('status', 'unknown'),
        'elapsed_sec': elapsed,
        'outer_budget_sec': cfg.get('timeout_sec_per_seed', 2100) * len(cfg.get('seeds', [])) +
                            manifest.get('remote', {}).get('setup_timeout_sec', 900),
        'seeds': cfg.get('seeds', []),
        'problem_sequence': manifest.get('problem', {}).get('sequence', []),
        'wandb_mode': manifest.get('tracking', {}).get('mode', 'unknown'),
        'wandb_sync_status': sync_status,
        'runs': sorted(run_rows, key=lambda item: item.get('seed') or 0),
        'logs': logs,
        'output_available': bool(run_rows or logs),
        'error': error,
        'refreshed_at': datetime.now(timezone.utc).isoformat(),
        'limitations': ('Esta campaña registra W&B en modo offline. Kaggle solo expone el estado del kernel; '
                        'los pasos y métricas aparecen cuando se descargan los artefactos al terminar.')
    }
    _CACHE.update(at=now, value=result)
    return result


PAGE = r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AF PIKANs · Run monitor</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2e;--line:#263451;--muted:#9baac4;--text:#eef3ff;--blue:#72a7ff;--green:#52d2a0;--amber:#ffc66d;--red:#ff7d86}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 15% -15%,#1e3765 0,transparent 50%),var(--bg);color:var(--text);font:15px/1.5 Inter,Segoe UI,sans-serif}.wrap{max-width:1120px;margin:auto;padding:34px 22px 60px}h1{font-size:30px;margin:0}.sub,.muted{color:var(--muted)}.sub{margin:4px 0 26px}.top{display:flex;align-items:center;justify-content:space-between;gap:20px}.badge{border:1px solid var(--line);border-radius:999px;padding:7px 13px;color:var(--blue);font-weight:650}.grid{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:12px}.card,.panel{background:linear-gradient(145deg,#141f36,#10182a);border:1px solid var(--line);border-radius:16px;padding:18px}.label{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}.value{font-size:22px;font-weight:700;margin-top:6px;overflow-wrap:anywhere}.panel{margin-top:15px}.panel h2{font-size:16px;margin:0 0 12px}.bar{height:8px;border-radius:8px;background:#202d45;overflow:hidden;margin-top:13px}.bar>i{display:block;background:linear-gradient(90deg,#4f86ff,#49d2a1);height:100%;width:0;transition:width .5s}.rows{display:grid;grid-template-columns:repeat(3,minmax(190px,1fr));gap:10px}.seed{border:1px solid var(--line);border-radius:12px;padding:13px}.seed b{display:block}.pill{color:var(--amber);font-size:12px}.notice{border-left:3px solid var(--amber);padding:10px 13px;background:#211e1b;color:#e9d7b7;border-radius:4px}.logs{max-height:330px;overflow:auto;background:#090d17;padding:12px;border-radius:10px;color:#b9c9e6;font:12px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.foot{margin-top:20px;color:var(--muted);font-size:12px}a{color:var(--blue)}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}.rows{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}
</style></head><body><main class="wrap"><div class="top"><div><h1>AF PIKANs</h1><div class="sub">Monitor de campaña · Heat + Burgers · Kaggle</div></div><span class="badge" id="state">consultando…</span></div>
<section class="grid"><article class="card"><div class="label">Estado remoto</div><div class="value" id="provider">—</div></article><article class="card"><div class="label">Tiempo transcurrido</div><div class="value" id="elapsed">—</div></article><article class="card"><div class="label">Presupuesto exterior</div><div class="value" id="budget">—</div></article><article class="card"><div class="label">W&B · modo y sync</div><div class="value" id="wandb">—</div></article></section>
<section class="panel"><h2>Campaña y semillas</h2><div class="muted" id="problems"></div><div class="bar"><i id="progress"></i></div><div class="muted" id="progress-note" style="margin-top:7px"></div><div class="rows" id="seeds" style="margin-top:15px"></div></section>
<section class="panel"><h2>Telemetría disponible</h2><div class="notice" id="limitation"></div><div id="runs"></div></section><section class="panel"><h2>Salida de entrenamiento</h2><div class="logs" id="logs">Aún no hay logs descargables localmente.</div></section><div class="foot">Actualización automática cada 15 s · última consulta: <span id="updated">—</span><br><span id="error"></span></div></main>
<script>
const fmt=s=>{if(s==null)return'—';s=Math.max(0,Math.floor(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),ss=s%60;return h?`${h} h ${m} min`:`${m} min ${ss} s`};
async function refresh(){try{const r=await fetch('/api/state',{cache:'no-store'}),d=await r.json();document.querySelector('#provider').textContent=d.provider_state;document.querySelector('#state').textContent=d.provider_state.includes('RUNNING')?'ACTIVA':d.provider_state.toUpperCase();document.querySelector('#elapsed').textContent=fmt(d.elapsed_sec);document.querySelector('#budget').textContent=fmt(d.outer_budget_sec);document.querySelector('#wandb').textContent=(d.wandb_mode||'unknown').toUpperCase()+' → '+(d.wandb_sync_status||'unknown').toUpperCase();document.querySelector('#problems').textContent=(d.problem_sequence||[]).join(' → ');const pct=d.elapsed_sec==null?0:Math.min(100,100*d.elapsed_sec/d.outer_budget_sec);document.querySelector('#progress').style.width=pct+'%';document.querySelector('#progress-note').textContent=`${Math.round(pct)}% del límite exterior estimado · no representa pasos completados ni tiempo de cola.`;document.querySelector('#limitation').textContent=d.limitations;document.querySelector('#updated').textContent=new Date(d.refreshed_at).toLocaleTimeString();document.querySelector('#error').textContent=d.error||'';document.querySelector('#seeds').innerHTML=(d.seeds||[]).map(s=>{const row=(d.runs||[]).find(x=>x.seed===s);return `<div class="seed"><b>Semilla ${s}</b><span class="pill">${row?.status||'estado detallado no publicado'}</span>${row?.elapsed_sec!=null?`<div class="muted">${fmt(row.elapsed_sec)}</div>`:''}</div>`}).join('');document.querySelector('#runs').innerHTML=(d.runs||[]).map(x=>`<div class="seed" style="margin-top:8px"><b>Semilla ${x.seed}: ${x.status}</b><span class="muted"> ${JSON.stringify(x.metrics||{})}</span></div>`).join('')||'<div class="muted">Sin artefactos descargados todavía.</div>';document.querySelector('#logs').textContent=d.logs?.length?d.logs.map(x=>`--- ${x.file} ---\n${x.lines.join('\n')}`).join('\n\n'):'Los logs permanecen en el entorno de Kaggle hasta que la ejecución termine y se descarguen sus artefactos.'}catch(e){document.querySelector('#error').textContent='Dashboard no pudo actualizarse: '+e}}refresh();setInterval(refresh,15000);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/state':
            payload = json.dumps(snapshot(), ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in ('/', '/index.html'):
            payload = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def log_message(self, _format, *_args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {'127.0.0.1', 'localhost'}:
        raise SystemExit('The monitor may only bind to localhost')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'AF PIKANs run monitor: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
