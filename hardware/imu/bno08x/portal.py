"""Live web portal for a BNO08x IMU in UART-RVC mode, read through a CP2102 USB-UART adapter.

    python hardware/imu/bno08x/portal.py                    # auto-detect CP210x, http://127.0.0.1:8001
    python hardware/imu/bno08x/portal.py --port COM3 --log imu.csv
    python hardware/imu/bno08x/portal.py --simulate         # synthetic 100 Hz stream, no hardware

Wiring (see README.md): sensor P0 -> 3V (selects UART-RVC), sensor SDA -> adapter RXD, 3V -> 3V3, GND -> GND.
The camera portal uses port 8000 and the same CP210x VID/PID; give each its own --port if both adapters are plugged in.

Endpoints: /  (page)  /status  /samples?after=N  (JSON)  /samples.csv  (buffered history)
"""
from __future__ import annotations

import argparse
import csv
import io
import time

from flask import Flask, Response, jsonify, request

from rvc import CSV_FIELDS, FRAME_LEN, NOMINAL_HZ, RvcReader

PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>BNO08x Live</title>
<style>
:root{--bg:#f6f7f9;--fg:#1c2026;--card:#fff;--mut:#5d6672;--ok:#1a7f37;--bad:#c62828;--bd:#d8dce2;--acc:#2a6fdb;
--c1:#2a6fdb;--c2:#d9822b;--c3:#1a9e6b;--face:#dfe4ec;--warn:#fff4d6;--warnfg:#6b4e00}
@media(prefers-color-scheme:dark){:root{--bg:#14171b;--fg:#e6e8eb;--card:#1e2329;--mut:#98a1ad;--ok:#3fb950;--bad:#f85149;
--bd:#333a43;--acc:#6ea8ff;--c1:#6ea8ff;--c2:#f0a04b;--c3:#3fc994;--face:#39424e;--warn:#3a2f10;--warnfg:#f0d68a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:1.15rem;margin:0 0 12px}h2{font-size:.95rem;margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin-bottom:12px}
.row{display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center}
.stat b{display:block;font-size:1.25rem;font-variant-numeric:tabular-nums;min-width:4.6em}.stat span{color:var(--mut);font-size:.8rem}
select,input,button{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--bd);border-radius:6px;padding:4px 8px}
button{cursor:pointer}a{color:var(--acc)}small,.mut{color:var(--mut)}
#dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--bad);margin-right:6px}#dot.ok{background:var(--ok)}
canvas{width:100%;background:var(--bg);border:1px solid var(--bd);border-radius:6px;display:block}
.grid{display:grid;grid-template-columns:340px 1fr;gap:12px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
#warn{display:none;background:var(--warn);color:var(--warnfg);border-radius:8px;padding:10px 12px;margin-bottom:12px}
.c1{color:var(--c1)}.c2{color:var(--c2)}.c3{color:var(--c3)}
</style>
<main>
<h1><span id=dot></span>BNO08x <span id=msg class=mut style="font-weight:400"></span></h1>
<div id=warn></div>
<div class=grid>
 <div class=card><h2>Orientation</h2><canvas id=cube width=320 height=300></canvas>
  <div class=row style="margin-top:8px">
   <label><input type=checkbox id=iy>invert yaw</label><label><input type=checkbox id=ip>invert pitch</label>
   <label><input type=checkbox id=ir>invert roll</label><button id=zero>Zero yaw</button>
  </div>
  <small>Axes: <b style="color:#d33">X</b> <b style="color:#2a9">Y</b> <b style="color:#38f">Z</b> (sensor body). Rotation order Rz(yaw)·Ry(pitch)·Rx(roll);
  sign conventions are unverified on hardware &ndash; use the invert boxes until the tilt direction matches.</small></div>
 <div>
  <div class="card row">
   <div class=stat><b id=yaw class=c1>–</b><span>yaw °</span></div>
   <div class=stat><b id=pitch class=c2>–</b><span>pitch °</span></div>
   <div class=stat><b id=roll class=c3>–</b><span>roll °</span></div>
   <div class=stat><b id=amag>–</b><span>|a| m/s² (rest ≈ 9.81)</span></div>
  </div>
  <div class="card row">
   <div class=stat><b id=hz>–</b><span>frames/s (nominal %HZ%)</span></div>
   <div class=stat><b id=count>–</b><span>frames</span></div>
   <div class=stat><b id=crc>–</b><span>bad checksum</span></div>
   <div class=stat><b id=drop>–</b><span>dropped</span></div>
   <div class=stat><b id=bps>–</b><span>bytes/s</span></div>
   <span style="flex:1"></span><a href=/samples.csv download=bno08x.csv>Download CSV</a>
  </div>
  <div class=card><h2>Attitude <small>(10 s)</small></h2><canvas id=att width=600 height=170></canvas>
   <small><span class=c1>■</span> yaw <span class=c2>■</span> pitch <span class=c3>■</span> roll</small></div>
 </div>
</div>
<div class=card><h2>Acceleration <small>(10 s, sensor frame, includes gravity)</small></h2><canvas id=acc width=900 height=170></canvas>
 <small><span class=c1>■</span> x <span class=c2>■</span> y <span class=c3>■</span> z &nbsp; m/s²</small></div>
</main>
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const WIN=10,MAXBUF=1500;let buf=[],after=0,yawZero=0,lastBytes=0,lastBT=performance.now();
const inv={};for(const k of['iy','ip','ir']){try{$(k).checked=localStorage.getItem(k)==='1'}catch(e){}
 $(k).onchange=()=>{try{localStorage.setItem(k,$(k).checked?'1':'0')}catch(e){}}}
$('zero').onclick=()=>{if(buf.length)yawZero=buf[buf.length-1][3]};

async function pollSamples(){try{const r=await (await fetch('/samples?after='+after)).json();
 if(r.latest<after){buf=[];}          // portal restarted
 if(r.samples.length){after=r.samples[r.samples.length-1][0];buf=buf.concat(r.samples);
  if(buf.length>MAXBUF)buf=buf.slice(-MAXBUF)}else after=Math.min(after,r.latest);
}catch(e){}}
setInterval(pollSamples,50);

async function pollStatus(){try{const s=await (await fetch('/status')).json();
 const now=performance.now(),bps=(s.bytes_rx-lastBytes)/((now-lastBT)/1000);lastBytes=s.bytes_rx;lastBT=now;
 $('hz').textContent=s.hz.toFixed(1);$('count').textContent=s.frames;$('crc').textContent=s.crc_errors;
 $('drop').textContent=s.dropped;$('bps').textContent=Math.max(0,bps).toFixed(0);
 const live=s.age_s!==null&&s.age_s<1;$('dot').className=live?'ok':'';
 $('msg').textContent='– '+s.port_label+(live?' @ 115200 UART-RVC':'');
 const w=$('warn');w.style.display=s.diagnosis?'block':'none';w.textContent=s.diagnosis;
}catch(e){$('dot').className='';$('msg').textContent='– portal unreachable'}}
setInterval(pollStatus,1000);pollStatus();

// ---- strip charts
function strip(cv,cols,keys,lim,wrap){const c=cv.getContext('2d'),W=cv.width,H=cv.height;c.clearRect(0,0,W,H);
 c.fillStyle=css('--mut');c.strokeStyle=css('--bd');c.font='11px system-ui';c.lineWidth=1;
 if(!buf.length)return;const t1=buf[buf.length-1][1],t0=t1-WIN;
 let m=lim;if(!lim){m=1;for(const s of buf)if(s[1]>=t0)for(const k of keys)m=Math.max(m,Math.abs(s[k]));m=Math.ceil(m/2)*2}
 c.beginPath();c.moveTo(0,H/2);c.lineTo(W,H/2);c.stroke();c.fillText('+'+m,4,11);c.fillText('-'+m,4,H-4);
 keys.forEach((k,j)=>{c.strokeStyle=css(cols[j]);c.lineWidth=1.5;c.beginPath();let prev=null,pen=false;
  for(const s of buf){if(s[1]<t0)continue;let v=s[k];if(k===3&&wrap)v=wrapDeg(v-yawZero);
   const x=(s[1]-t0)/WIN*W,y=H/2-v/m*(H/2-6);
   if(wrap&&prev!==null&&Math.abs(v-prev)>180)pen=false;
   pen?c.lineTo(x,y):c.moveTo(x,y);pen=true;prev=v}c.stroke()})}
const wrapDeg=d=>((d+540)%360)-180;

// ---- 3D cube
const cv=$('cube'),cx=cv.getContext('2d');
const V=[[-1,-.6,-.15],[1,-.6,-.15],[1,.6,-.15],[-1,.6,-.15],[-1,-.6,.15],[1,-.6,.15],[1,.6,.15],[-1,.6,.15]];
const F=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]];
function rot(y,p,r){const cy=Math.cos(y),sy=Math.sin(y),cp=Math.cos(p),sp=Math.sin(p),cr=Math.cos(r),sr=Math.sin(r);
 return[[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]}
const A=-35*Math.PI/180,E=28*Math.PI/180;
function proj(v){const x1=v[0]*Math.cos(A)-v[1]*Math.sin(A),y1=v[0]*Math.sin(A)+v[1]*Math.cos(A);
 return{x:x1,u:v[2]*Math.cos(E)+y1*Math.sin(E),d:y1*Math.cos(E)-v[2]*Math.sin(E)}}
function drawCube(){const W=cv.width,H=cv.height,S=W*.2,ox=W/2,oy=H/2;cx.clearRect(0,0,W,H);
 const P=(v)=>{const q=proj(v);return{x:ox+q.x*S,y:oy-q.u*S,d:q.d}};
 cx.strokeStyle=css('--bd');cx.lineWidth=1;cx.beginPath();          // ground ring
 for(let i=0;i<=48;i++){const a=i/48*6.2832,q=P([1.9*Math.cos(a),1.9*Math.sin(a),-1.3]);i?cx.lineTo(q.x,q.y):cx.moveTo(q.x,q.y)}cx.stroke();
 if(!buf.length){cx.fillStyle=css('--mut');cx.font='13px system-ui';cx.textAlign='center';cx.fillText('no data',ox,oy);cx.textAlign='start';return}
 const l=buf[buf.length-1],sg=k=>$(k).checked?-1:1;
 const R=rot(wrapDeg(l[3]-yawZero)*sg('iy')*Math.PI/180,l[4]*sg('ip')*Math.PI/180,l[5]*sg('ir')*Math.PI/180);
 const M=v=>[R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]];
 const pts=V.map(v=>P(M(v)));
 const faces=F.map(f=>({f,d:f.reduce((a,i)=>a+pts[i].d,0)/4})).sort((a,b)=>b.d-a.d);
 faces.forEach((o,i)=>{cx.beginPath();o.f.forEach((k,j)=>j?cx.lineTo(pts[k].x,pts[k].y):cx.moveTo(pts[k].x,pts[k].y));cx.closePath();
  cx.globalAlpha=.88;cx.fillStyle=css('--face');cx.fill();cx.globalAlpha=1;cx.strokeStyle=css('--mut');cx.stroke()});
 const o0=P([0,0,0]);[[[1.6,0,0],'#d33','X'],[[0,1.3,0],'#2a9','Y'],[[0,0,1.1],'#38f','Z']].forEach(([v,col,name])=>{
  const q=P(M(v));cx.strokeStyle=col;cx.fillStyle=col;cx.lineWidth=2.5;cx.beginPath();cx.moveTo(o0.x,o0.y);cx.lineTo(q.x,q.y);cx.stroke();
  cx.font='bold 12px system-ui';cx.fillText(name,q.x+4,q.y-4)})}

function render(){
 if(buf.length){const l=buf[buf.length-1];
  $('yaw').textContent=wrapDeg(l[3]-yawZero).toFixed(1);$('pitch').textContent=l[4].toFixed(1);$('roll').textContent=l[5].toFixed(1);
  $('amag').textContent=Math.hypot(l[6],l[7],l[8]).toFixed(2)}
 drawCube();strip($('att'),['--c1','--c2','--c3'],[3,4,5],180,true);strip($('acc'),['--c1','--c2','--c3'],[6,7,8],0,false);
 requestAnimationFrame(render)}
requestAnimationFrame(render);
</script>"""


def create_app(rx: RvcReader) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE.replace("%HZ%", f"{NOMINAL_HZ:.0f}")

    @app.get("/status")
    def status():
        age = time.time() - rx.last_frame_t if rx.last_frame_t else None
        live = age is not None and age < 1.0
        return jsonify(hz=rx.hz if live else 0.0, frames=rx.count, bytes_rx=rx.bytes_rx, age_s=age,
                       crc_errors=rx.parser.crc_errors, dropped=rx.parser.dropped, junk_bytes=rx.parser.junk_bytes,
                       port_label=rx.port or "no port", diagnosis=rx.diagnosis(), frame_len=FRAME_LEN)

    @app.get("/samples")
    def samples():
        after = int(request.args.get("after", 0))
        return jsonify(latest=rx.count, samples=rx.since(after), fields=CSV_FIELDS[1:])

    @app.get("/samples.csv")
    def samples_csv():
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(("n", "t_dev_s", "index", "yaw_deg", "pitch_deg", "roll_deg", "ax_mps2", "ay_mps2", "az_mps2"))
        w.writerows(rx.since(0, limit=10**9))
        return Response(out.getvalue(), mimetype="text/csv")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: first CP210x found)")
    ap.add_argument("--simulate", action="store_true", help="synthetic RVC stream instead of hardware")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8001)
    ap.add_argument("--log", metavar="CSV", help="write every frame to this CSV file")
    args = ap.parse_args()

    rx = RvcReader(args.port, args.simulate, args.log)
    rx.start()
    print(f"BNO08x UART-RVC {'(simulated)' if args.simulate else args.port or '(auto-detect)'}"
          f"  ->  http://{args.host}:{args.http_port}")
    create_app(rx).run(host=args.host, port=args.http_port, threaded=True)


if __name__ == "__main__":
    main()
