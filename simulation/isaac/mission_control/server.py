"""Local-only mission service. Launch with Isaac Python: -m mission_control.server."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .models import ROOT, DEFAULTS, validate_mission, policy_paths, default_mission

HERE = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/mission_control'
RUNS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='EDF Mission Control', docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
lock = threading.Lock()
process = None
active_id = None


def active_training_command():
    """Find this repository's active trainer without inspecting unrelated work."""
    for candidate in psutil.process_iter(['name']):
        if 'python' not in (candidate.info['name'] or '').lower():
            continue
        try:
            command = candidate.cmdline()
            if any(Path(arg).name == 'run_train_ppo.py' for arg in command):
                if str(ROOT).lower() in ' '.join(command).lower() or Path(candidate.cwd()).resolve() == ROOT:
                    return command
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def external_training_running():
    return active_training_command() is not None


def latest_jsonl(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 20000))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    except FileNotFoundError:
        pass
    return {}


def training_snapshot(command):
    if not command:
        return None
    base = ROOT / (command[command.index('--output-dir') + 1] if '--output-dir' in command else 'runs')
    if not base.resolve().is_relative_to((ROOT / 'runs').resolve()):
        return None
    runs = list(base.glob('*/args.json'))
    if not runs:
        return None
    run = max(runs, key=lambda path: path.stat().st_mtime).parent
    update = latest_jsonl(run / 'train_log.jsonl')
    evaluation = latest_jsonl(run / 'eval_log.jsonl')
    selected = dict(run=run.name, step=update.get('global_step'),
                    stage=update.get('spawn_stage_index'), stages=update.get('spawn_num_stages'),
                    stage_success=update.get('stage_success_fraction'),
                    full_eval_step=evaluation.get('global_step'),
                    full_success=evaluation.get('success_fraction'),
                    success_energy_wh=evaluation.get('success_mean_energy_wh'),
                    success_delta_v=evaluation.get('success_mean_propulsive_delta_v_m_s'))
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in selected.items()}


@app.middleware('http')
async def local_write_guard(request: Request, call_next):
    if request.method not in ('GET', 'HEAD'):
        if request.headers.get('x-mission-control') != 'local':
            return JSONResponse({'detail': 'Local mission header required'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Cross-origin writes forbidden'}, status_code=403)
        limit = 200_000_000 if request.url.path.endswith('/video') else 100_000
        if int(request.headers.get('content-length', '0')) > limit:
            return JSONResponse({'detail': 'Request too large'}, status_code=413)
    return await call_next(request)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def directory(mid):
    if not re.fullmatch(r'[a-f0-9]{12}', mid):
        raise HTTPException(404, 'Mission not found')
    p = RUNS / mid
    if not p.is_dir():
        raise HTTPException(404, 'Mission not found')
    return p


def status(mid):
    p = directory(mid)
    s = read_json(p / 'status.json', {'state': 'starting', 'phase': 'Launching Isaac', 'frames': 0})
    if mid == active_id and process and process.poll() is not None and s['state'] not in ('complete', 'failed', 'cancelled'):
        s.update(state='failed', error=f'Isaac process exited with code {process.returncode}; see mission log')
    elif mid != active_id and s['state'] in ('starting', 'running'):
        s.update(state='failed', error='Mission interrupted before this server session')
    return dict(id=mid, **s)


@app.get('/api/config')
def config():
    policies = {'ppo': 'Legacy PPO 26M · old hinges', 'ppo_deterministic': 'Legacy PPO 34M · old hinges', 'pid': 'PID · radial hinges'}
    if 'ppo_radial' in policy_paths():
        policies = {'ppo_radial': 'PPO · radial 8S / battery aware', **policies}
    if 'ppo_mission' in policy_paths():
        policies = {'ppo_mission': 'EXPERIMENTAL PPO · recovery + waypoints', **policies}
    training = active_training_command()
    return dict(defaults=default_mission(), hardware=read_json(HERE / 'hardware.json'),
                policies=policies,
                engine='NVIDIA Isaac Sim / PhysX', training=bool(training), training_metrics=training_snapshot(training),
                active=active_id if process and process.poll() is None else None)


@app.get('/api/missions')
def list_missions():
    ids = sorted((p for p in RUNS.iterdir() if p.is_dir() and re.fullmatch(r'[a-f0-9]{12}', p.name)), key=lambda p: p.stat().st_mtime, reverse=True)
    return [dict(status(p.name), request=read_json(p / 'request.json', {}),
                 hinge_layout=read_json(p / 'metadata.json', {}).get('hinge_layout')) for p in ids[:100]]


@app.post('/api/missions', status_code=202)
async def start(request: Request):
    global process, active_id
    try:
        spec = validate_mission(await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with lock:
        if external_training_running():
            raise HTTPException(409, 'PPO training is using Isaac. Replay is available; start a new mission after training finishes.')
        if process and process.poll() is None:
            raise HTTPException(409, 'An Isaac mission is already running')
        mid = uuid.uuid4().hex[:12]
        p = RUNS / mid
        p.mkdir()
        (p / 'request.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
        (p / 'created.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat()}))
        with (p / 'isaac.log').open('wb') as output:
            process = subprocess.Popen([sys.executable, str(ROOT / 'apps/run_mission.py'), '--request', str(p / 'request.json'), '--output', str(p)],
                                       cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        active_id = mid
        return status(mid)


@app.get('/api/missions/{mid}')
def mission(mid: str):
    p = directory(mid)
    return dict(status(mid), metadata=read_json(p / 'metadata.json'), request=read_json(p / 'request.json'), video=(p / 'landing.webm').exists())


@app.get('/api/missions/{mid}/frames')
def frames(mid: str, after: int = 0):
    if after < 0:
        raise HTTPException(422, 'Negative frame offset')
    p = directory(mid) / 'frames.jsonl'
    data = []
    if p.exists():
        with p.open(encoding='utf-8') as stream:
            for i, line in enumerate(stream):
                if i >= after and line.endswith('\n'):
                    data.append(json.loads(line))
    return dict(frames=data, next=after + len(data))


@app.post('/api/missions/{mid}/stop')
def stop(mid: str):
    if mid != active_id or not process or process.poll() is not None:
        raise HTTPException(409, 'Mission is not running')
    (directory(mid) / 'STOP').touch()
    return {'state': 'stopping'}


@app.get('/api/missions/{mid}/download')
def download(mid: str):
    p = directory(mid)
    data = dict(metadata=read_json(p / 'metadata.json'), summary=read_json(p / 'summary.json'), **frames(mid))
    return JSONResponse(data, headers={'Content-Disposition': f'attachment; filename="mission-{mid}.json"'})


@app.get('/api/missions/{mid}/log')
def log(mid: str):
    return FileResponse(directory(mid) / 'isaac.log', media_type='text/plain')


@app.post('/api/missions/{mid}/video')
async def save_video(mid: str, request: Request):
    p = directory(mid)
    if request.headers.get('content-type') != 'video/webm':
        raise HTTPException(415, 'WebM required')
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 200_000_000:
            raise HTTPException(413, 'Video exceeds 200 MB')
    if data[:4] != b'\x1a\x45\xdf\xa3':
        raise HTTPException(422, 'Invalid WebM header')
    temporary = p / ('video-' + uuid.uuid4().hex + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(p / 'landing.webm')
    return {'path': str(p / 'landing.webm'), 'bytes': len(data)}


@app.get('/api/missions/{mid}/video')
def video(mid: str):
    p = directory(mid) / 'landing.webm'
    if not p.exists():
        raise HTTPException(404, 'No video recorded for this mission')
    return FileResponse(p, media_type='video/webm', filename=f'landing-{mid}.webm')


@app.get('/')
def index():
    return FileResponse(HERE / 'static/index.html')


app.mount('/static', StaticFiles(directory=HERE / 'static'), name='static')


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8830)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port)
