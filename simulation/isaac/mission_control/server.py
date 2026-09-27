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
import time
import uuid
from collections import OrderedDict
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .models import ROOT, DEFAULTS, validate_mission, policy_paths, default_mission, convex_available, braking_envelopes

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
# Hosts the service answers to. --lan adds this machine's LAN names at startup
# (the middleware reads the list when the app builds its stack, on first request).
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', 'testserver']
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
lock = threading.Lock()
process = None
active_id = None


# Trainers that own Isaac, with the output directory each uses by default.
TRAINERS = {'run_train_ppo.py': 'runs', 'run_train_waypoints.py': 'runs/waypoint_flight'}
MODEL_RUNS = ROOT / 'runs/waypoint_flight'


class JsonlCache:
    """Complete JSONL records, re-reading only bytes appended since the last call.

    Trainer and mission logs are append-only, so polling endpoints cost
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
TRAINING_SCAN_TTL_S = 3.
_training_scan = (0., None)


def scan_training_command():
    """Find this repository's active trainer without inspecting unrelated work."""
    for candidate in psutil.process_iter(['name']):
        if 'python' not in (candidate.info['name'] or '').lower():
            continue
        try:
            command = candidate.cmdline()
            if any(Path(arg).name in TRAINERS for arg in command):
                if str(ROOT).lower() in ' '.join(command).lower() or Path(candidate.cwd()).resolve() == ROOT:
                    return command
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def active_training_command():
    """Process scan (~30 ms on Windows) shared by the status polls for a few seconds."""
    global _training_scan
    stamp, command = _training_scan
    if time.monotonic() - stamp > TRAINING_SCAN_TTL_S:
        command = scan_training_command()
        _training_scan = (time.monotonic(), command)
    return command


def external_training_running():
    # Launch decisions always use a fresh scan, never the shared status cache.
    return scan_training_command() is not None


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


def finite_values(value):
    """JSON-safe copy: the trainers write NaN for metrics with no samples."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_values(item) for item in value]
    return value


def curriculum_stages(run):
    task = (read_json(run / 'task_config.json', {}) or {}).get('task', {})
    return task.get('waypoint_flight', {}).get('curriculum', {}).get('stages', [])


def training_snapshot(command):
    if not command:
        return None
    trainer = next(Path(arg).name for arg in command if Path(arg).name in TRAINERS)
    base = ROOT / (command[command.index('--output-dir') + 1] if '--output-dir' in command else TRAINERS[trainer])
    if not base.resolve().is_relative_to((ROOT / 'runs').resolve()):
        return None
    runs = list(base.glob('*/args.json'))
    if not runs:
        return None
    run = max(runs, key=lambda path: path.stat().st_mtime).parent
    update = latest_jsonl(run / 'train_log.jsonl')
    evaluation = latest_jsonl(run / 'eval_log.jsonl')
    if trainer == 'run_train_waypoints.py':
        selected = dict(task='waypoint_flight', run=run.name, step=update.get('global_step'),
                        stage=update.get('stage_index'), stages=len(curriculum_stages(run)) or None,
                        stage_name=update.get('stage_name'),
                        stage_success=update.get('stage_success_fraction'),
                        peak_yaw=update.get('mean_peak_yaw_deg_s'), sps=update.get('sps'),
                        full_eval_step=evaluation.get('global_step'),
                        full_success=evaluation.get('success_fraction'),
                        success_energy_wh=evaluation.get('success_mean_energy_wh'),
                        success_delta_v=None)
    else:
        selected = dict(task='landing', run=run.name, step=update.get('global_step'),
                        stage=update.get('spawn_stage_index'), stages=update.get('spawn_num_stages'),
                        stage_success=update.get('stage_success_fraction'),
                        full_eval_step=evaluation.get('global_step'),
                        full_success=evaluation.get('success_fraction'),
                        success_energy_wh=evaluation.get('success_mean_energy_wh'),
                        success_delta_v=evaluation.get('success_mean_propulsive_delta_v_m_s'))
    return finite_values(selected)


def read_jsonl(path):
    """Complete records only; a trainer may be appending the final line."""
    return log_records.read(path)


def train_updates(run):
    return [r for r in read_jsonl(run / 'train_log.jsonl') if r.get('type', 'train_update') == 'train_update']


def outcome_fractions(record):
    outcomes = record.get('outcomes') or {}
    total = sum(outcomes.values())
    return {key: value / total for key, value in outcomes.items()} if total else None


def model_summary(run):
    """Summarise one waypoint_flight run from its logs; no checkpoint is loaded."""
    updates = train_updates(run)
    last = updates[-1] if updates else {}
    checkpoints = []
    for path in run.glob('*.pt'):
        digits, info = path.stem.removeprefix('ppo_step_'), path.stat()
        checkpoints.append(dict(file=path.name, step=int(digits) if digits.isdigit() else None,
                                bytes=info.st_size, modified=info.st_mtime))
    args = read_json(run / 'args.json', {}) or {}
    return finite_values(dict(
        run=run.name, task='waypoint_flight', modified=run.stat().st_mtime,
        resume=args.get('resume'), num_envs=args.get('num_envs'), total_steps=args.get('total_steps'),
        step=last.get('global_step'), update=last.get('update'), sps=last.get('sps'),
        stage=last.get('stage_index'), stage_name=last.get('stage_name'),
        stages=[stage.get('name') for stage in curriculum_stages(run)],
        stage_success=last.get('stage_success_fraction'), outcomes=outcome_fractions(last),
        peak_yaw=last.get('mean_peak_yaw_deg_s'), throttle=last.get('throttle_mean'),
        explained_variance=last.get('explained_variance'),
        evaluation=read_json(run / 'eval_latest.json'),
        checkpoints=sorted(checkpoints, key=lambda c: c['modified'])))


@app.get('/api/models')
def list_models():
    """Mission-flyable policies plus the latest waypoint_flight training runs.

    The registered mission policy may be a waypoint_flight checkpoint
    (run_mission.py flies it on its own training task); other runs listed
    here are for inspection only.
    """
    flyable = []
    for key, path in policy_paths().items():
        registry = HERE / ('mission_policy_registry.json' if key == 'ppo_mission' else 'policy_registry.json')
        record = read_json(registry, {}) or {}
        flyable.append(dict(key=key, checkpoint=path, status=record.get('status'),
                            validated=record.get('validated', False), note=record.get('note'),
                            training_run=record.get('training_run')))
    if convex_available():
        flyable.append(dict(key='convex', checkpoint='configs/controllers/convex_guidance.yaml',
                            status='deterministic', validated=False, label=CONVEX_LABEL, note=CONVEX_NOTE))
    runs = sorted((p for p in MODEL_RUNS.glob('*') if (p / 'args.json').is_file()),
                  key=lambda p: p.stat().st_mtime, reverse=True) if MODEL_RUNS.is_dir() else []
    training = active_training_command()
    active_run = (training_snapshot(training) or {}).get('run')
    return dict(flyable=flyable, active_run=active_run,
                runs=[dict(model_summary(run), active=run.name == active_run) for run in runs[:8]])


@app.get('/api/models/{run}/history')
def model_history(run: str, points: int = 240):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run) or not (MODEL_RUNS / run / 'args.json').is_file():
        raise HTTPException(404, 'Training run not found')
    updates = train_updates(MODEL_RUNS / run)
    stride = max(1, math.ceil(len(updates) / max(10, min(points, 2000))))
    kept = updates[::stride]
    if updates and kept[-1] is not updates[-1]:
        kept.append(updates[-1])
    series = [dict(step=r.get('global_step'), stage=r.get('stage_index'),
                   stage_success=r.get('stage_success_fraction'),
                   rollout_success=r.get('rollout_success_fraction'),
                   spin=(outcome_fractions(r) or {}).get('SPIN'),
                   peak_yaw=r.get('mean_peak_yaw_deg_s'), reward=r.get('reward_mean'),
                   explained_variance=r.get('explained_variance'), throttle=r.get('throttle_mean'))
              for r in kept]
    evaluations = [dict(step=r.get('global_step'), success=r.get('success_fraction'))
                   for r in read_jsonl(MODEL_RUNS / run / 'eval_log.jsonl')]
    return finite_values(dict(run=run, series=series, evaluations=evaluations))


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
    paths = policy_paths()
    policies = {}
    if convex_available():
        policies['convex'] = CONVEX_LABEL
    if 'ppo_radial' in paths:
        policies = {'ppo_radial': 'PPO · radial 8S / battery aware', **policies}
    if 'ppo_mission' in paths:
        policies = {'ppo_mission': 'EXPERIMENTAL PPO · waypoint flight + landing', **policies}
    training = active_training_command()
    return dict(defaults=default_mission(paths), hardware=read_json(HERE / 'hardware.json'),
                policies=policies, vehicles=braking_envelopes(), convex_parameters=schema(),
                disturbance_defaults=disturbance_defaults(),
                engine='NVIDIA Isaac Sim / PhysX', training=bool(training), training_metrics=training_snapshot(training),
                active=active_id if process and process.poll() is None else None)


@app.post('/api/flight-plan/validate')
async def validate_plan(request: Request):
    try:
        return validate_mission(await request.json())
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
    parser.add_argument('--lan', action='store_true',
                        help='Listen on every interface so devices on the local network can open the console. '
                             'There is no login: anyone on the network can launch and stop Isaac missions.')
    args = parser.parse_args()
    host = '127.0.0.1'
    if args.lan:
        import socket
        host = '0.0.0.0'
        names = {socket.gethostname(), socket.getfqdn()}
        addresses = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
        ALLOWED_HOSTS.extend(sorted(names | addresses))
        print('Mission control on the LAN: ' + ', '.join(f'http://{a}:{args.port}' for a in sorted(addresses)))
    uvicorn.run(app, host=host, port=args.port)
