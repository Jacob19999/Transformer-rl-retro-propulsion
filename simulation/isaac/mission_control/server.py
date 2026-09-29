"""Local-only mission service. Launch with Isaac Python: -m mission_control.server."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .models import ROOT, DEFAULTS, validate_mission, default_mission, convex_available, braking_envelopes

HERE = Path(__file__).resolve().parent
CONVEX_LABEL = 'CONVEX · SOCP powered-descent guidance'
CONVEX_NOTE = ('Deterministic classical controller, no checkpoint. A second-order cone program '
               '(Acikmese & Ploen 2007, lossless convexification) plans a minimum-energy thrust '
               'trajectory through the route to the pad with thrust, tilt, glide-slope, speed and '
               'thrust-rate constraints; it is re-solved every 0.5 s from the measured state and '
               'tracked by position feedback and a geometric attitude loop. Not flight-qualified.')
RUNS = ROOT / 'runs/mission_control'
RUNS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='EDF Mission Control', docs_url=None, redoc_url=None)
# Hosts the service answers to. LAN bind (default) adds this machine's names at
# startup; the middleware reads the list when the app builds its stack on first request.
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', 'testserver']
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
lock = threading.Lock()
process = None
active_id = None



class JsonlCache:
    """Complete JSONL records, re-reading only bytes appended since the last call.

    Mission logs are append-only, so polling endpoints cost
    O(new lines) instead of re-parsing multi-MB files. A shrunk or replaced
    file is re-read from the start. raw=True keeps each validated line as
    bytes so large frame streams can be returned without a decode/encode trip.
    """

    def __init__(self, raw=False, capacity=24):
        self.raw, self.capacity, self.entries, self.lock = raw, capacity, OrderedDict(), threading.Lock()

    def read(self, path):
        try:
            info = path.stat()
        except FileNotFoundError:
            with self.lock:
                self.entries.pop(path, None)
            return []
        identity = (info.st_dev, info.st_ino)
        with self.lock:
            entry = self.entries.get(path)
            if entry is None or entry['identity'] != identity or info.st_size < entry['offset']:
                entry = dict(identity=identity, offset=0, records=[])
            if info.st_size > entry['offset']:
                with path.open('rb') as stream:
                    stream.seek(entry['offset'])
                    data = stream.read(info.st_size - entry['offset'])
                complete = data.rfind(b'\n') + 1  # a trainer may be appending the final line
                for line in data[:complete].splitlines():
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    entry['records'].append(line if self.raw else record)
                entry['offset'] += complete
            self.entries[path] = entry
            self.entries.move_to_end(path)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return entry['records']


log_records = JsonlCache()
frame_lines = JsonlCache(raw=True)
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


def recorded_request(path, metadata=None):
    """Restore diagnostic requests from their recorded, resolved parameters."""
    if metadata is None:
        metadata = read_json(path / 'metadata.json', {}) or {}
    resolved = dict(metadata.get('request') or {})
    supplied = read_json(path / 'request.json', {}) or {}
    battery = {**resolved.get('battery', {}), **supplied.get('battery', {})}
    resolved.update(supplied)
    if battery:
        resolved['battery'] = battery
    if 'vane_model' not in resolved and 'dynamics' in metadata:
        # Recorded before the field existed: report the plant actually flown.
        coupled = ((metadata['dynamics'] or {}).get('coupled_jet') or {}).get('enabled')
        resolved['vane_model'] = 'momentum' if coupled else 'legacy'
    return resolved


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
    from .convex_parameters import schema
    from .disturbance_parameters import defaults as disturbance_defaults
    policies = {}
    if convex_available():
        policies['convex'] = CONVEX_LABEL
    return dict(defaults=default_mission(), hardware=read_json(HERE / 'hardware.json'),
                policies=policies, vehicles=braking_envelopes(), convex_parameters=schema(),
                disturbance_defaults=disturbance_defaults(),
                engine='NVIDIA Isaac Sim / PhysX',
                active=active_id if process and process.poll() is None else None)


@app.post('/api/flight-plan/validate')
async def validate_plan(request: Request):
    try:
        return validate_mission(await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/flight-plan/route')
async def validate_route(request: Request):
    """Route fields only: plans are independent of guidance and environment."""
    from .flight_plans import validate_route as route_only
    try:
        return route_only(await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/convex-profiles')
def convex_profiles():
    from .convex_parameters import list_profiles
    return list_profiles()


@app.get('/api/convex-presets')
def convex_presets():
    from .presets import convex_presets
    return convex_presets()


@app.get('/api/disturbance-presets')
def disturbance_presets():
    from .presets import disturbance_presets
    return disturbance_presets()


@app.get('/api/flight-plan-samples')
def flight_plan_samples():
    from .presets import list_samples
    return list_samples()


@app.get('/api/flight-plan-samples/{key}')
def flight_plan_sample(key: str):
    from .presets import read_sample
    try:
        return read_sample(key)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, 'Sample flight plan is missing or invalid') from exc


@app.get('/api/flight-plans')
def flight_plans():
    from .flight_plans import list_plans
    return list_plans()


@app.get('/api/flight-plans/{key}')
def flight_plan(key: str):
    from .flight_plans import read_plan
    try:
        return read_plan(key)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, 'Saved flight plan is missing or invalid') from exc


@app.post('/api/flight-plans')
async def save_flight_plan(request: Request):
    from .flight_plans import save_plan
    try:
        value = await request.json()
        with lock:
            return save_plan(value)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/convex-profiles')
async def save_convex_profile(request: Request):
    from .convex_parameters import save_profile
    try:
        body = await request.json()
        if not isinstance(body, dict) or set(body) - {'name', 'settings', 'note'}:
            raise ValueError('Expected profile name, settings and optional note')
        with lock:
            return save_profile(body.get('name'), body.get('settings'), body.get('note', ''))
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/missions')
def list_missions():
    ids = sorted((p for p in RUNS.iterdir() if p.is_dir() and re.fullmatch(r'[a-f0-9]{12}', p.name)), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for p in ids[:100]:
        metadata = read_json(p / 'metadata.json', {}) or {}
        result.append(dict(status(p.name), request=recorded_request(p, metadata), hinge_layout=metadata.get('hinge_layout')))
    return result


@app.post('/api/missions', status_code=202)
async def start(request: Request):
    global process, active_id
    try:
        spec = validate_mission(await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with lock:
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
    metadata = read_json(p / 'metadata.json')
    return dict(status(mid), metadata=metadata, request=recorded_request(p, metadata or {}), video=(p / 'landing.webm').exists())


def frame_slice(mid, after):
    """Recorded frame lines (already strict JSON from run_mission.py) from `after` on."""
    if after < 0:
        raise HTTPException(422, 'Negative frame offset')
    lines = frame_lines.read(directory(mid) / 'frames.jsonl')[after:]
    return b'[' + b','.join(lines) + b']', after + len(lines)


def raw_json(parts, headers=None):
    return Response(b''.join(parts), media_type='application/json', headers=headers)


@app.get('/api/missions/{mid}/frames')
def frames(mid: str, after: int = 0):
    # Frames pass through as recorded: decoding and re-encoding a full replay
    # (~3 MB) dominated load time, and every line was validated when cached.
    data, following = frame_slice(mid, after)
    return raw_json([b'{"frames":', data, b',"next":%d}' % following])


@app.post('/api/missions/{mid}/stop')
def stop(mid: str):
    if mid != active_id or not process or process.poll() is not None:
        raise HTTPException(409, 'Mission is not running')
    (directory(mid) / 'STOP').touch()
    return {'state': 'stopping'}


@app.get('/api/missions/{mid}/download')
def download(mid: str):
    p = directory(mid)
    head = json.dumps(dict(metadata=read_json(p / 'metadata.json'), summary=read_json(p / 'summary.json')))
    data, following = frame_slice(mid, 0)
    return raw_json([head[:-1].encode(), b',"frames":', data, b',"next":%d}' % following],
                    headers={'Content-Disposition': f'attachment; filename="mission-{mid}.json"'})


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
    # Keep the UI bundle and stylesheet in sync after a local update. The old
    # fixed query string let a cached pre-editor CSS leave the canvas 300x150.
    html = (HERE / 'static/index.html').read_text(encoding='utf-8')
    for filename in ('style.css', 'fin-labels.css', 'planner-workspace.css', 'app.js'):
        path = HERE / 'static' / filename
        if path.is_file():
            html = re.sub(r'/static/' + re.escape(filename) + r'(?:\?[^"\s]*)?',
                          f'/static/{filename}?v={path.stat().st_mtime_ns}', html)
    return HTMLResponse(html, headers={'Cache-Control': 'no-cache'})


app.mount('/static', StaticFiles(directory=HERE / 'static'), name='static')


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8830)
    parser.add_argument('--local', action='store_true',
                        help='Restrict to this machine (127.0.0.1). Default listens on every interface '
                             'so LAN devices can open the console; there is no login.')
    parser.add_argument('--lan', action='store_true',
                        help=argparse.SUPPRESS)  # kept for old start.ps1 callers; LAN is already default
    args = parser.parse_args()
    host = '127.0.0.1' if args.local else '0.0.0.0'
    if not args.local:
        import socket
        names = {socket.gethostname(), socket.getfqdn()}
        addresses = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
        ALLOWED_HOSTS.extend(sorted(names | addresses))
        print('Mission control on the LAN: ' + ', '.join(f'http://{a}:{args.port}' for a in sorted(addresses)))
    if not convex_available():
        # Controller dropdown omits convex when Clarabel is missing; Isaac missions
        # also need this env. start.ps1 / env_isaaclab Python is the supported launcher.
        print(f'WARNING: clarabel not found in {sys.executable}; convex controller hidden. '
              'Use env_isaaclab/Scripts/python.exe (or start.ps1).', flush=True)
    uvicorn.run(app, host=host, port=args.port)
