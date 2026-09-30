"""Live web portal for the ESP32-CAM (firmware: hardware/esp32_cam_stream).

Reads framed packets from the CP2102 serial port and serves them to a browser:
  CAM1  JPEG frames  -> MJPEG stream / snapshot (in FLOW mode: a low-rate preview of what the camera sees)
  FLW1  optical flow -> live vector, strip chart, match-quality indicators, optional CSV log
  MSG1  device log   -> shown on the page

    python hardware/camera_portal.py                       # auto-detect CP210x, http://127.0.0.1:8000
    python hardware/camera_portal.py --log flow.csv        # also write every flow sample to CSV
    python hardware/camera_portal.py --port COM3 --http-port 8080

Endpoints: /  (page)  /stream  (MJPEG)  /frame.jpg  /status  /flow?n=300  (JSON)
           POST /config {"framesize","quality"}  /mode {"mode":0|1}  /exposure {"auto","aec","gain"}
           POST /preview {"ms"}  /selftest

The ESP32 must be running the firmware with the IO0-GND jumper removed (press RESET after removing it).
"""
from __future__ import annotations

import argparse
import binascii
import collections
import csv
import struct
import threading
import time

import serial
from flask import Flask, Response, jsonify, request
from serial.tools import list_ports

MAGICS = (b"CAM1", b"FLW1", b"MSG1")
CAM_HEADER = struct.Struct("<III")  # payload_len, seq, esp_millis
FLOW_BODY = struct.Struct("<IIIhhHHHH")  # seq, t_us, dt_us, dx_q8, dy_q8, sad, tex, compute_us, crc
FLOW_CRC_SPAN = FLOW_BODY.size - 2
MAX_JPEG = 200_000
BOOT_BAUD = 921_600  # firmware always boots at this rate; the portal then asks it to switch up
KEEPALIVE = b"K\xa5"
# esp32-camera framesize_t values (STREAM mode)
FRAMESIZES = {"QQVGA 160x120": 1, "HQVGA 240x176": 3, "QVGA 320x240": 5, "CIF 400x296": 6, "VGA 640x480": 8}
FLOW_FIELDS = ("pc_time_s", "seq", "t_dev_s", "dt_ms", "dx_px", "dy_px", "sad", "tex", "compute_us")


class FrameReceiver(threading.Thread):
    """Resyncing parser: hunts for a packet magic, validates length + CRC, dispatches by packet type."""

    def __init__(self, port: str, baud: int, log_path: str | None = None):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.active_baud: int | None = None
        self.notice: str | None = None
        self.ser: serial.Serial | None = None
        self._tx_lock = threading.Lock()
        self.cond = threading.Condition()
        self.latest: bytes | None = None
        self.count = 0  # good JPEG frames
        self.crc_errors = 0
        self.lost = 0  # JPEG frames skipped according to firmware seq numbers
        self.bytes_rx = 0
        self.last_seq: int | None = None
        self.last_frame_t = 0.0
        self.fps = 0.0
        self.error: str | None = None
        self._fps_t = time.time()
        self._fps_n = 0
        # optical flow
        self.flow: collections.deque = collections.deque(maxlen=2000)
        self.flow_count = 0
        self.flow_fps = 0.0
        self.last_flow_t = 0.0
        self.t_dev_s = 0.0  # device capture time, accumulated from dt_us (starts at 0 at the first sample)
        self.cum_dx = 0.0
        self.cum_dy = 0.0
        self._flow_fps_t = time.time()
        self._flow_fps_n = 0
        self.log_lines: collections.deque = collections.deque(maxlen=60)
        self._csv_file = open(log_path, "w", newline="") if log_path else None
        self._csv = csv.writer(self._csv_file) if self._csv_file else None
        if self._csv:
            self._csv.writerow(FLOW_FIELDS)

    # ---- commands to the firmware -------------------------------------------------------------
    def _send(self, data: bytes) -> None:
        if self.ser and self.ser.is_open:
            with self._tx_lock:
                self.ser.write(data)

    def send_config(self, framesize: int, quality: int) -> None:
        self._send(bytes([ord("C"), framesize & 0xFF, quality & 0xFF]))

    def send_mode(self, mode: int) -> None:
        self._send(bytes([ord("M"), mode & 0xFF]))

    def send_preview(self, ms: int) -> None:
        self._send(b"V" + struct.pack("<H", max(0, min(ms, 65535))))

    def send_exposure(self, auto: bool, aec: int, gain: int) -> None:
        self._send(b"E" + struct.pack("<BHB", 1 if auto else 0, max(0, min(aec, 1200)), max(0, min(gain, 30))))

    def send_selftest(self) -> None:
        self._send(b"T")

    # ---- packet reader ------------------------------------------------------------------------
    def _read_exact(self, n: int, deadline: float | None = None) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            if deadline is not None and time.time() > deadline:
                return None
            chunk = self.ser.read(n - len(buf))
            if chunk:
                buf += chunk
                self.bytes_rx += len(chunk)
        return bytes(buf)

    def _sync(self, deadline: float | None = None) -> bytes | None:
        win = bytearray()
        while True:
            if deadline is not None and time.time() > deadline:
                return None
            b = self.ser.read(1)
            if not b:
                continue
            self.bytes_rx += 1
            win += b
            del win[:-4]
            if bytes(win) in MAGICS:
                return bytes(win)

    def _read_packet(self, deadline: float | None = None):
        """Return (magic, parsed) for the next CRC-valid packet, or None if `deadline` passes first."""
        while True:
            magic = self._sync(deadline)
            if magic is None:
                return None
            if magic == b"CAM1":
                hdr = self._read_exact(CAM_HEADER.size, deadline)
                if hdr is None:
                    return None
                length, seq, _ = CAM_HEADER.unpack(hdr)
                if length == 0 or length > MAX_JPEG:
                    continue  # false magic inside garbage; resync
                body = self._read_exact(length + 2, deadline)
                if body is None:
                    return None
                jpeg, (crc,) = body[:-2], struct.unpack("<H", body[-2:])
                if crc != binascii.crc_hqx(jpeg, 0) or not (jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"):
                    self.crc_errors += 1
                    continue
                return magic, (seq, jpeg)
            if magic == b"FLW1":
                body = self._read_exact(FLOW_BODY.size, deadline)
                if body is None:
                    return None
                fields = FLOW_BODY.unpack(body)
                if fields[-1] != binascii.crc_hqx(body[:FLOW_CRC_SPAN], 0):
                    self.crc_errors += 1
                    continue
                return magic, fields
            # MSG1
            raw = self._read_exact(2, deadline)
            if raw is None:
                return None
            (length,) = struct.unpack("<H", raw)
            if length > 200:
                continue
            rest = self._read_exact(length + 2, deadline)
            if rest is None:
                return None
            text, (crc,) = rest[:-2], struct.unpack("<H", rest[-2:])
            if crc != binascii.crc_hqx(text, binascii.crc_hqx(raw, 0)):
                self.crc_errors += 1
                continue
            return magic, text.decode("utf-8", errors="replace")

    def _open(self, baud: int) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(self.port, baud, timeout=0.2)

    def _has_frames(self, seconds: float = 1.5) -> bool:
        """True if any valid packet shows up within `seconds` at the current baud."""
        return self._read_packet(time.time() + seconds) is not None

    def _connect(self) -> None:
        """Open the port at whichever baud the firmware is currently using, switching to self.baud if needed."""
        for baud in dict.fromkeys((self.baud, BOOT_BAUD)):  # target first: board may already be switched
            self._open(baud)
            if not self._has_frames():
                continue
            if baud != self.baud:
                self._send(b"B" + struct.pack("<I", self.baud))
                self.ser.flush()
                time.sleep(0.5)  # firmware finishes its in-flight packet, then re-clocks its UART
                self.ser.baudrate = self.baud
                self.ser.reset_input_buffer()
                if not self._has_frames(3.0):
                    # The firmware reverts to BOOT_BAUD on its own once keepalives stop (3 s).
                    self.notice = f"{self.baud} baud failed (adapter/wiring limit?); using {BOOT_BAUD}"
                    self.baud = BOOT_BAUD
                    raise OSError(self.notice)
            self.active_baud = self.baud
            return
        raise OSError("no frames from ESP32 (IO0 jumper removed? press RESET)")

    def run(self) -> None:
        while True:
            try:
                self._connect()
                self.error = None
                self._loop()
            except (serial.SerialException, OSError) as e:
                self.error = str(e)
                time.sleep(1.0)  # port busy / unplugged / board silent: keep retrying

    def _loop(self) -> None:
        last_ka = 0.0
        while True:
            if time.time() - last_ka > 0.5:  # keeps the firmware at a non-boot baud
                self._send(KEEPALIVE)
                last_ka = time.time()
            pkt = self._read_packet(time.time() + 1.0)  # bounded so keepalives continue while the board is quiet
            if pkt is None:
                continue
            kind, data = pkt
            if kind == b"CAM1":
                self._on_jpeg(*data)
            elif kind == b"FLW1":
                self._on_flow(data)
            else:
                self.log_lines.append(f"{time.strftime('%H:%M:%S')}  {data}")

    def _on_jpeg(self, seq: int, jpeg: bytes) -> None:
        if self.last_seq is not None and seq > self.last_seq + 1:
            self.lost += seq - self.last_seq - 1
        self.last_seq = seq
        now = time.time()
        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0
        with self.cond:
            self.latest, self.count, self.last_frame_t = jpeg, self.count + 1, now
            self.cond.notify_all()

    def _on_flow(self, f) -> None:
        seq, _t_us, dt_us, dx_q8, dy_q8, sad, tex, compute_us, _crc = f
        now = time.time()
        dx, dy = dx_q8 / 256.0, dy_q8 / 256.0
        if self.flow_count == 0:
            self.t_dev_s = 0.0
        self.t_dev_s += dt_us * 1e-6
        self.cum_dx += dx
        self.cum_dy += dy
        self.flow_count += 1
        self.last_flow_t = now
        self._flow_fps_n += 1
        if now - self._flow_fps_t >= 1.0:
            self.flow_fps = self._flow_fps_n / (now - self._flow_fps_t)
            self._flow_fps_t, self._flow_fps_n = now, 0
        self.flow.append((seq, round(self.t_dev_s, 4), round(dt_us / 1000.0, 2), round(dx, 4), round(dy, 4),
                          round(sad / 16.0, 3), round(tex / 16.0, 3), compute_us))
        if self._csv:
            self._csv.writerow((f"{now:.4f}", seq, f"{self.t_dev_s:.6f}", f"{dt_us / 1000.0:.3f}", f"{dx:.4f}",
                                f"{dy:.4f}", f"{sad / 16.0:.3f}", f"{tex / 16.0:.3f}", compute_us))
            if seq % 25 == 0:
                self._csv_file.flush()


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ESP32-CAM Live</title>
<style>
:root{--bg:#f6f7f9;--fg:#1c2026;--card:#fff;--mut:#5d6672;--ok:#1a7f37;--bad:#c62828;--bd:#d8dce2;--x:#2a6fdb;--y:#d9822b;--acc:#2a6fdb}
@media(prefers-color-scheme:dark){:root{--bg:#14171b;--fg:#e6e8eb;--card:#1e2329;--mut:#98a1ad;--ok:#3fb950;--bad:#f85149;--bd:#333a43;--x:#6ea8ff;--y:#f0a04b;--acc:#6ea8ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}
h1{font-size:1.15rem;margin:0 0 12px}h2{font-size:.95rem;margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin-bottom:12px}
#view{display:block;width:100%;image-rendering:pixelated;background:#000;border-radius:6px;min-height:120px}
.row{display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center}
.stat b{display:block;font-size:1.25rem;font-variant-numeric:tabular-nums}.stat span{color:var(--mut);font-size:.8rem}
select,input,button{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--bd);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
#dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--bad);margin-right:6px}
#dot.ok{background:var(--ok)}
a{color:inherit}small,.mut{color:var(--mut)}
canvas{width:100%;background:var(--bg);border:1px solid var(--bd);border-radius:6px;display:block}
.grid{display:grid;grid-template-columns:1fr 150px;gap:12px}@media(max-width:640px){.grid{grid-template-columns:1fr}}
#log{margin:8px 0 0;max-height:9em;overflow:auto;font:12px ui-monospace,monospace;white-space:pre-wrap;color:var(--mut)}
.lx{color:var(--x)}.ly{color:var(--y)}
</style>
<main>
<h1><span id=dot></span>ESP32-CAM <span id=msg style="color:var(--mut);font-weight:400"></span></h1>
<div class=card><img id=view src=/stream alt="live stream"></div>
<div class="card row">
 <div class=stat><b id=fps>–</b><span>fps (jpeg)</span></div>
 <div class=stat><b id=kbps>–</b><span>KB/s</span></div>
 <div class=stat><b id=count>–</b><span>frames</span></div>
 <div class=stat><b id=crc>–</b><span>CRC errors</span></div>
 <div class=stat><b id=lost>–</b><span>dropped</span></div>
 <span style="flex:1"></span>
 <span class=row>Mode <button id=mStream>Stream</button><button id=mFlow>Flow</button></span>
</div>

<div class=card id=flowCard>
 <h2>Optical flow <small id=flowNote></small></h2>
 <div class=row style="margin-bottom:8px">
  <div class=stat><b id=fdx class=lx>–</b><span>dx px/frame</span></div>
  <div class=stat><b id=fdy class=ly>–</b><span>dy px/frame</span></div>
  <div class=stat><b id=fwx class=lx>–</b><span>dx °/s</span></div>
  <div class=stat><b id=fwy class=ly>–</b><span>dy °/s</span></div>
  <div class=stat><b id=ffps>–</b><span>flow Hz</span></div>
  <div class=stat><b id=fcomp>–</b><span>ARPS ms</span></div>
  <div class=stat><b id=fconf>–</b><span>confidence</span></div>
 </div>
 <div class=grid>
  <canvas id=chart width=560 height=200></canvas>
  <canvas id=vec width=150 height=150></canvas>
 </div>
 <div class=row style="margin-top:8px">
  <small>Image shift between consecutive frames (feature moves by +dx right, +dy down). Camera rotation is the opposite sign.</small>
  <label>°/pixel <input id=dpp type=number step=0.01 value=0.22 style="width:5em"></label>
  <small>(0.22 assumes a 66° diagonal lens – <b>uncalibrated</b>)</small>
  <span>cum <b id=cum class=mut>–</b> px</span>
  <button id=btnSelf>Self-test</button>
 </div>
</div>

<div class="card row">
 <label>Resolution <select id=fs>%FS%</select></label>
 <label>JPEG quality <input id=q type=number min=4 max=63 value=12 style="width:4.5em"> <small>lower = sharper</small></label>
 <button id=apply>Apply</button>
 <a href=/frame.jpg download=frame.jpg>Save snapshot</a>
</div>
<div class="card row">
 <label><input id=auto type=checkbox checked> Auto exposure/gain</label>
 <label>Exposure <input id=aec type=number min=0 max=1200 value=300 style="width:5em"></label>
 <label>Gain <input id=gain type=number min=0 max=30 value=8 style="width:4em"></label>
 <button id=applyExp>Apply</button>
 <small>Lock exposure for flow: fixed exposure avoids brightness pumping between frames.</small>
</div>
<div class=card><h2>Device log</h2><pre id=log></pre></div>
</main>
<script>
const $=id=>document.getElementById(id);let lastB=0,lastT=performance.now(),mode=0;
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
async function poll(){try{const s=await (await fetch('/status')).json();
 const now=performance.now(),kb=(s.bytes_rx-lastB)/1024/((now-lastT)/1000);lastB=s.bytes_rx;lastT=now;
 $('fps').textContent=s.fps.toFixed(1);$('kbps').textContent=kb.toFixed(0);$('count').textContent=s.frames;
 $('crc').textContent=s.crc_errors;$('lost').textContent=s.lost;
 const live=(s.age_s!==null&&s.age_s<2)||(s.flow_age_s!==null&&s.flow_age_s<2);$('dot').className=live?'ok':'';
 mode=s.flow_age_s!==null&&s.flow_age_s<2?1:0;
 $('mStream').className=mode?'':'on';$('mFlow').className=mode?'on':'';
 $('msg').textContent=s.error?('– '+s.error):(live?('– '+(s.baud/1e6).toFixed(2)+' Mbaud, '+(mode?'FLOW':'STREAM')+(s.notice?' ('+s.notice+')':'')):'– no frames (IO0 jumper removed? press RESET)');
 $('log').textContent=s.log.join('\\n');
}catch(e){$('msg').textContent='– portal unreachable'}}
setInterval(poll,1000);poll();
$('apply').onclick=()=>post('/config',{framesize:+$('fs').value,quality:+$('q').value});
$('applyExp').onclick=()=>post('/exposure',{auto:$('auto').checked,aec:+$('aec').value,gain:+$('gain').value});
$('mStream').onclick=()=>post('/mode',{mode:0});$('mFlow').onclick=()=>post('/mode',{mode:1});
$('btnSelf').onclick=()=>post('/selftest');

const chart=$('chart'),cx=chart.getContext('2d'),vec=$('vec'),vx=vec.getContext('2d');
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function drawChart(S){const W=chart.width,H=chart.height;cx.clearRect(0,0,W,H);
 if(!S.length)return;const t1=S[S.length-1][1],t0=t1-10;const pts=S.filter(s=>s[1]>=t0);
 let m=1;for(const s of pts)m=Math.max(m,Math.abs(s[3]),Math.abs(s[4]));m=Math.ceil(m);
 cx.strokeStyle=css('--bd');cx.fillStyle=css('--mut');cx.font='11px system-ui';cx.lineWidth=1;
 cx.beginPath();cx.moveTo(0,H/2);cx.lineTo(W,H/2);cx.stroke();cx.fillText('+'+m+' px',4,11);cx.fillText('-'+m+' px',4,H-4);
 for(const [k,c] of [[3,'--x'],[4,'--y']]){cx.strokeStyle=css(c);cx.lineWidth=1.5;cx.beginPath();
  pts.forEach((s,i)=>{const x=(s[1]-t0)/10*W,y=H/2-s[k]/m*(H/2-6);i?cx.lineTo(x,y):cx.moveTo(x,y)});cx.stroke()}}
function drawVec(dx,dy){const W=vec.width,H=vec.height,c=W/2;vx.clearRect(0,0,W,H);
 vx.strokeStyle=css('--bd');vx.beginPath();vx.arc(c,c,c-6,0,7);vx.moveTo(6,c);vx.lineTo(W-6,c);vx.moveTo(c,6);vx.lineTo(c,H-6);vx.stroke();
 const k=(c-10)/5,px=Math.max(-5,Math.min(5,dx)),py=Math.max(-5,Math.min(5,dy)),ex=c+px*k,ey=c+py*k;
 vx.strokeStyle=css('--acc');vx.fillStyle=css('--acc');vx.lineWidth=2.5;vx.beginPath();vx.moveTo(c,c);vx.lineTo(ex,ey);vx.stroke();
 vx.beginPath();vx.arc(ex,ey,4,0,7);vx.fill();vx.fillStyle=css('--mut');vx.font='10px system-ui';vx.fillText('±5 px',W-38,H-4)}
async function pollFlow(){try{const r=await (await fetch('/flow?n=400')).json();
 $('flowNote').textContent=r.active?'':'(switch Mode to Flow)';if(!r.samples.length)return;
 const S=r.samples,l=S[S.length-1],dpp=+$('dpp').value||0,dt=l[2]/1000;
 $('fdx').textContent=l[3].toFixed(2);$('fdy').textContent=l[4].toFixed(2);
 $('fwx').textContent=dt>0?(l[3]*dpp/dt).toFixed(1):'–';$('fwy').textContent=dt>0?(l[4]*dpp/dt).toFixed(1):'–';
 $('ffps').textContent=r.fps.toFixed(1);$('fcomp').textContent=(l[7]/1000).toFixed(1);
 const ratio=l[6]>0?l[5]/l[6]:9;$('fconf').textContent=Math.max(0,Math.min(1,1-ratio)).toFixed(2);
 $('cum').textContent=r.cum_dx.toFixed(1)+', '+r.cum_dy.toFixed(1);drawChart(S);drawVec(l[3],l[4]);
}catch(e){}}
setInterval(pollFlow,200);
</script>"""


def create_app(rx: FrameReceiver) -> Flask:
    app = Flask(__name__)
    fs_opts = "".join(f'<option value={v}{" selected" if v == 5 else ""}>{k}</option>' for k, v in FRAMESIZES.items())

    @app.get("/")
    def index():
        return PAGE.replace("%FS%", fs_opts)

    @app.get("/frame.jpg")
    def frame():
        if rx.latest is None:
            return Response("no frame yet", status=503)
        return Response(rx.latest, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/stream")
    def stream():
        def gen():
            seen = -1
            while True:
                with rx.cond:
                    if not rx.cond.wait_for(lambda: rx.count != seen and rx.latest is not None, timeout=5):
                        continue
                    jpeg, seen = rx.latest, rx.count
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg)) + jpeg + b"\r\n"

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/status")
    def status():
        now = time.time()
        age = now - rx.last_frame_t if rx.last_frame_t else None
        flow_age = now - rx.last_flow_t if rx.last_flow_t else None
        return jsonify(fps=rx.fps if age is not None and age < 2 else 0.0, frames=rx.count,
                       crc_errors=rx.crc_errors, lost=rx.lost, bytes_rx=rx.bytes_rx, age_s=age, error=rx.error,
                       baud=rx.active_baud, notice=rx.notice, flow_age_s=flow_age, log=list(rx.log_lines)[-12:])

    @app.get("/flow")
    def flow():
        n = min(max(int(request.args.get("n", 300)), 1), 2000)
        active = rx.last_flow_t and time.time() - rx.last_flow_t < 2
        return jsonify(samples=list(rx.flow)[-n:], active=bool(active), fps=rx.flow_fps if active else 0.0,
                       count=rx.flow_count, cum_dx=rx.cum_dx, cum_dy=rx.cum_dy)

    @app.post("/config")
    def config():
        body = request.get_json(silent=True) or {}
        fs, q = int(body.get("framesize", 5)), int(body.get("quality", 12))
        rx.send_config(fs, min(max(q, 4), 63))
        return jsonify(ok=True)

    @app.post("/mode")
    def mode():
        rx.send_mode(1 if (request.get_json(silent=True) or {}).get("mode") else 0)
        return jsonify(ok=True)

    @app.post("/exposure")
    def exposure():
        b = request.get_json(silent=True) or {}
        rx.send_exposure(bool(b.get("auto", True)), int(b.get("aec", 300)), int(b.get("gain", 8)))
        return jsonify(ok=True)

    @app.post("/preview")
    def preview():
        rx.send_preview(int((request.get_json(silent=True) or {}).get("ms", 500)))
        return jsonify(ok=True)

    @app.post("/selftest")
    def selftest():
        rx.send_selftest()
        return jsonify(ok=True)

    return app


def find_port() -> str:
    for p in list_ports.comports():
        if (p.vid, p.pid) == (0x10C4, 0xEA60):
            return p.device
    raise SystemExit("No CP210x adapter found; pass --port.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=1_500_000,
                    help="target UART baud (firmware boots at 921600; this CP2102 adapter fails above 1.5M)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--log", metavar="CSV", help="write every optical-flow sample to this CSV file")
    args = ap.parse_args()

    rx = FrameReceiver(args.port or find_port(), args.baud, args.log)
    rx.start()
    print(f"Serial {rx.port} @ {rx.baud}  ->  http://{args.host}:{args.http_port}")
    create_app(rx).run(host=args.host, port=args.http_port, threaded=True)


if __name__ == "__main__":
    main()
