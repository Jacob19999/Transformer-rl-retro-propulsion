"""Live sensor-fusion web portal: TFmini Plus (range) + WTGAHRS1 (IMU, magnetometer, barometer) -> EKF3-style fusion.

    python hardware/sensor_fusion/portal.py                       # auto-detect both adapters, http://127.0.0.1:8002
    python hardware/sensor_fusion/portal.py --tf-port COM3 --imu-port COM4 --log fusion.csv
    python hardware/sensor_fusion/portal.py --simulate            # synthetic sensors, no hardware

Run ``init_sensors.py`` once first (puts the WTGAHRS1 at 115200 baud / 100 Hz). Ports 8000 (camera) and 8001 (BNO08x)
are taken by the other portals. Endpoints: /  /status  /samples?after=N  /samples.csv  POST /api/reset
and the setup API: GET|POST /api/config  POST /api/calib/{start,cancel,finish_mag,reset}  POST /api/imu/gyro_autozero
(simulate only: POST /api/sim/pose). Calibration is stored in calibration.json (calibration_sim.json with --simulate).
"""
from __future__ import annotations

import argparse
import csv
import io
import threading
import time
from dataclasses import asdict
from pathlib import Path

from flask import Flask, Response, jsonify, request

import detect
import sim
import tfmini
import witmotion
from calibration import FACES, PATH as CALIB_PATH
from common import SensorReader, SerialStream, cp210x_ports
from frames import MOUNTS, mount_label
from fusion import FusionParams
from hub import KINDS, ROW_FIELDS, FusionHub

SIM_POSES = ("moving", "level", "nose_up") + FACES

PAGE_PATH = Path(__file__).with_name("portal.html")  # the page; read per request so edits show on reload


class Locator:
    """Finds which CP210x carries which sensor, and re-probes when a sensor goes quiet (e.g. after a power cycle)."""

    def __init__(self, tf_port: str | None, imu_port: str | None, imu_baud: int | None):
        self.lock = threading.Lock()
        self.known: dict[str, tuple[str, int] | None] = {
            "tfmini": (tf_port, tfmini.BAUD) if tf_port else None,
            "witmotion": (imu_port, imu_baud or witmotion.DEFAULT_BAUD) if imu_port else None}
        self.explicit = {"tfmini": bool(tf_port), "witmotion": bool(imu_port)}
        self.last_open: dict[str, float] = {}

    def opener(self, kind: str):
        def open_stream():
            with self.lock:
                recent = time.time() - self.last_open.get(kind, 0) < 30  # reopened soon after the last open: re-probe
                hit = self.known[kind]
                if hit is None or (recent and not self.explicit[kind]):
                    taken = {v[0] for k, v in self.known.items() if v and k != kind}
                    free = [p for p in cp210x_ports() if p not in taken]
                    found = detect.find_sensors(free, log=lambda m: print(f"[detect] {m}"))
                    hit = self.known[kind] = found.get(kind)
                self.last_open[kind] = time.time()
                if hit is None:
                    raise OSError(f"no {kind} found on any CP210x adapter (pass --{'tf' if kind == 'tfmini' else 'imu'}-port)")
                return SerialStream(*hit), hit[0], hit[1]
        return open_stream


class SimOpener:
    def __init__(self, factory):
        self.factory = factory

    def __call__(self):
        return self.factory(), "simulator", 0


def build(args) -> tuple[FusionHub, SensorReader, SensorReader]:
    calib_path = CALIB_PATH.with_name("calibration_sim.json") if args.simulate else args.calibration
    hub = FusionHub(FusionParams.from_profile(), log_path=args.log, calib_path=calib_path, kind=args.filter,
                    bench_aiding=not args.no_bench_aiding, mag_yaw=args.mag_yaw,
                    ekf_overrides=dict(zupt_guard=False, fast_reset=False) if args.legacy_stationary else None)
    if args.simulate:
        errors = sim.SimErrors(mount=args.sim_mount)
        tf_open, imu_open = SimOpener(sim.SimTfmini), SimOpener(lambda: sim.SimWitmotion(errors=errors))
    else:
        loc = Locator(args.tf_port, args.imu_port, args.imu_baud)
        if not (args.tf_port and args.imu_port):
            print("Detecting sensors on the CP210x adapters...")
            found = detect.find_sensors()
            for kind in ("tfmini", "witmotion"):
                loc.known[kind] = loc.known[kind] or found.get(kind)
        tf_open, imu_open = loc.opener("tfmini"), loc.opener("witmotion")
    tf = SensorReader("TFmini", tf_open, tfmini.TfParser, hub.on_tf,
                      "port open but the TFmini is silent: check 5 V (red) / GND (black), TFmini TXD (green) -> adapter RXD.",
                      "bytes arriving but no valid TFmini frames: wrong baud (needs 115200) or wrong sensor on this port.",
                      tfmini.NOMINAL_HZ)
    imu = SensorReader("WTGAHRS1", imu_open, witmotion.WitParser, hub.on_imu,
                       "port open but the WTGAHRS1 is silent: check VCC (red) / GND (black), sensor TX (yellow) -> adapter RXD.",
                       "bytes arriving but no valid WitMotion packets: wrong baud (run init_sensors.py) or wrong sensor on this port.",
                       witmotion.DEFAULT_HZ)
    return hub, tf, imu


def link_status(r: SensorReader, name: str, parser_stats: dict) -> dict:
    return dict(name=name, port=r.port, baud=r.baud or None, hz=r.hz if r.live else 0.0, frames=r.count,
                bytes_rx=r.bytes_rx, age_s=r.age_s, diagnosis=r.diagnosis(), **parser_stats)


def create_app(hub: FusionHub, tf: SensorReader, imu: SensorReader, simulate: bool = False) -> Flask:
    app = Flask(__name__)

    def body() -> dict:
        d = request.get_json(silent=True)
        return d if isinstance(d, dict) else {}

    def bad(e: Exception):
        return jsonify(ok=False, error=str(e)), 400

    @app.get("/")
    def index():
        return Response(PAGE_PATH.read_text(encoding="utf-8"), mimetype="text/html")

    @app.get("/status")
    def status():
        p, w = tf.parser, imu.parser
        return jsonify(tf_link=link_status(tf, "TFmini", dict(crc=p.crc_errors, junk=p.junk_bytes, invalid=p.invalid)),
                       imu_link=link_status(imu, "WTGAHRS1", dict(crc=w.crc_errors, junk=w.junk_bytes,
                                                                  packets={hex(k): v for k, v in w.packets.items()})),
                       sim=simulate, **hub.snapshot())

    @app.get("/samples")
    def samples():
        after = int(request.args.get("after", 0))
        limit = max(1, min(int(request.args.get("limit", 600)), 6000))      # the page polls small; analysis tools ask for the history
        return jsonify(latest=hub.count, samples=hub.since(after, limit), fields=ROW_FIELDS)

    @app.get("/samples.csv")
    def samples_csv():
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(ROW_FIELDS)
        w.writerows(hub.since(0, limit=10 ** 9))
        return Response(out.getvalue(), mimetype="text/csv")

    @app.post("/api/reset")
    def reset():
        hub.reset()
        return jsonify(ok=True)

    # ---- setup -------------------------------------------------------------------------------------------------
    @app.get("/api/config")
    def get_config():
        with hub.lock:
            return jsonify(calibration=asdict(hub.calib), settings=hub.settings(), faces=FACES, sim=simulate,
                           mounts=[dict(key=k, label=mount_label(k)) for k in MOUNTS], sim_poses=SIM_POSES)

    @app.post("/api/config")
    def set_config():
        d = body()
        try:
            if any(k in d for k in ("mount", "trim_deg", "tf_offset_m")):
                hub.set_calibration_fields(d.get("mount"), d.get("trim_deg"), d.get("tf_offset_m"))
            hub.set_settings(d.get("kind"), d.get("bench_aiding"), d.get("mag_yaw"))
        except (ValueError, TypeError, KeyError) as e:
            return bad(e)
        return jsonify(ok=True)

    @app.post("/api/calib/start")
    def calib_start():
        d = body()
        try:
            with hub.lock:
                hub.session.start(str(d.get("task")), d.get("face"))
        except ValueError as e:
            return bad(e)
        return jsonify(ok=True)

    @app.post("/api/calib/cancel")
    def calib_cancel():
        with hub.lock:
            hub.session.cancel()
        return jsonify(ok=True)

    @app.post("/api/calib/finish_mag")
    def calib_finish_mag():
        with hub.lock:
            hub.session.finish_mag()
        return jsonify(ok=True)

    @app.post("/api/calib/reset")
    def calib_reset():
        hub.reset_calibration()
        return jsonify(ok=True)

    @app.post("/api/imu/gyro_autozero")
    def gyro_autozero():
        enabled = body().get("enabled")
        if not isinstance(enabled, bool):
            return bad(ValueError("enabled must be true or false"))
        imu.send(witmotion.UNLOCK, witmotion.config(0x63, 0 if enabled else 1), witmotion.SAVE)  # 5.2.7: 1 = removed
        return jsonify(ok=True)

    @app.post("/api/sim/pose")
    def sim_pose():
        if not simulate:
            return jsonify(ok=False, error="not in simulate mode"), 404
        pose = body().get("pose")
        if pose not in SIM_POSES:
            return bad(ValueError(f"pose must be one of {SIM_POSES}"))
        sim.CONTROL["pose"] = pose
        return jsonify(ok=True)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf-port", help="TFmini serial port (default: auto-detect)")
    ap.add_argument("--imu-port", help="WTGAHRS1 serial port (default: auto-detect)")
    ap.add_argument("--imu-baud", type=int, help=f"WTGAHRS1 baud (default {witmotion.DEFAULT_BAUD}; re-detected if silent)")
    ap.add_argument("--filter", choices=KINDS, default="ekf3", help="ekf3 (raw IMU, default) or the simple vertical KF")
    ap.add_argument("--no-bench-aiding", action="store_true", help="EKF3 without gravity / stationary aiding (flight-like)")
    ap.add_argument("--mag-yaw", action="store_true", help="fuse magnetic yaw (needs the magnetometer calibration)")
    ap.add_argument("--legacy-stationary", action="store_true",
                    help="A/B only: the earlier stationary aid without the range guard and fast reset (freezes on slow descents)")
    ap.add_argument("--calibration", type=Path, default=CALIB_PATH,
                    help=f"calibration file (default {CALIB_PATH.name} next to this script)")
    ap.add_argument("--simulate", action="store_true", help="synthetic sensors instead of hardware")
    ap.add_argument("--sim-mount", default=sim.SimErrors().mount, choices=sorted(MOUNTS),
                    help="how the simulated IMU is bolted on (forward/right/down sensor axes)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8002)
    ap.add_argument("--log", metavar="CSV", help="write every fused row to this CSV file")
    args = ap.parse_args()

    hub, tf, imu = build(args)
    tf.start()
    imu.start()
    print(f"Sensor fusion portal ({'simulated' if args.simulate else 'hardware'}, {args.filter})  ->  "
          f"http://{args.host}:{args.http_port}")
    create_app(hub, tf, imu, args.simulate).run(host=args.host, port=args.http_port, threaded=True)


if __name__ == "__main__":
    main()
