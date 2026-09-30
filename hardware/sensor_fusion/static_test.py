"""Static (stationary) characterisation of the WTGAHRS1: record raw data at rest, then analyse it.

    python hardware/sensor_fusion/static_test.py record --minutes 30 --disable-autozero   # stop the portal first
    python hardware/sensor_fusion/static_test.py analyze static_runs/<run>.csv

The sensor must sit untouched on a stable surface for the whole capture. ``analyze`` writes a text report and plots next
to the log and compares the fitted noise terms with simulation/isaac/configs/sensors/imu_wtgahrs1.yaml. The Allan fit is
the simulator's own (tvc_env/dynamics/imu_allan.py), so the numbers drop straight into that profile. A 30 min log pins
down the white-noise terms; bias instability and correlation time need hours.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import serial

import witmotion
from detect import find_sensors

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).with_name("static_runs")
PROFILE = ROOT / "simulation/isaac/configs/sensors/imu_wtgahrs1.yaml"
FIELDS = ("t", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "mx", "my", "mz", "pressure_pa", "baro_h_m",
          "temp_c", "roll_deg", "pitch_deg", "yaw_deg")
GYRO_LSB = 2000 / 32768
FIT_TAU_MIN_S = 0.3   # the sensor low-passes its output (~20 Hz): below ~0.1 s the Allan curve bends, which the
                      # white + Gauss-Markov + random-walk model cannot represent, so shorter taus are left out of the fit
ACC_LSB_G = 16 / 32768


def _allan():
    """Load imu_allan.py by path so the simulator package (and torch) is not imported."""
    path = ROOT / "simulation/isaac/tvc_env/dynamics/imu_allan.py"
    spec = importlib.util.spec_from_file_location("imu_allan", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- record --------------------------------------------------------------------------------------------------
def record(minutes: float, port: str | None, disable_autozero: bool, out: Path | None) -> Path:
    if port is None:
        found = find_sensors(log=lambda m: print("[detect]", m))
        if "witmotion" not in found:
            raise SystemExit("WTGAHRS1 not found (is the portal still holding the port?)")
        port, baud = found["witmotion"]
    else:
        baud = witmotion.DEFAULT_BAUD
    RUNS.mkdir(exist_ok=True)
    out = out or RUNS / time.strftime("wtgahrs1_static_%Y%m%d_%H%M%S.csv")
    parser = witmotion.WitParser()
    with serial.Serial(port, baud, timeout=0.05) as ser:
        if disable_autozero:
            for cmd in (witmotion.UNLOCK, witmotion.config(0x63, 1), witmotion.SAVE):  # datasheet 5.2.7: 1 = removed
                ser.write(cmd)
                ser.flush()
                time.sleep(0.25)
            print("gyro auto-zero disabled and saved")
        ser.reset_input_buffer()
        n, t0, next_report = 0, time.time(), 60.0
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            print(f"recording {port} @ {baud} for {minutes:g} min -> {out}", flush=True)
            while time.time() - t0 < minutes * 60:
                for s in parser.feed(ser.read(max(1, ser.in_waiting)), time.time()):
                    if s.acc is None or s.gyro is None:
                        continue
                    acc_g = [a / witmotion.G for a in s.acc]
                    w.writerow([f"{s.t - t0:.4f}", *(f"{v:.6f}" for v in acc_g), *(f"{v:.5f}" for v in s.gyro),
                                *(s.mag or ("", "", "")), s.pressure_pa or "", s.baro_h_m if s.baro_h_m is not None else "",
                                s.temp_c if s.temp_c is not None else "", *(f"{v:.3f}" for v in (s.rpy or ("", "", "")))])
                    n += 1
                el = time.time() - t0
                if el >= next_report:
                    fh.flush()
                    print(f"  {el / 60:4.1f} min  {n} samples ({n / el:.1f} Hz)  bad checksums {parser.crc_errors}", flush=True)
                    next_report += 60.0
    print(f"done: {n} samples, {parser.crc_errors} bad checksums -> {out}")
    return out


# ---- analyze -------------------------------------------------------------------------------------------------
def _load(path: Path) -> dict:
    with path.open() as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    cols = {}
    for i, name in enumerate(head):
        cols[name] = np.array([float(r[i]) if i < len(r) and r[i] != "" else np.nan for r in body])
    return cols


def _profile() -> dict:
    try:
        import yaml
        return yaml.safe_load(PROFILE.read_text())["imu"]
    except Exception:
        return {}


def _motion_windows(g: np.ndarray, rate: float, thresh_dps: float = 1.5) -> np.ndarray:
    """Boolean mask of 1 s windows where the gyro moved (someone touched the bench)."""
    w = max(int(rate), 1)
    k = len(g) // w
    peak = np.abs(g[:k * w] - np.median(g, axis=0)).reshape(k, w, 3).max(axis=(1, 2))
    return peak > thresh_dps


def analyze(path: Path) -> str:
    A = _allan()
    d = _load(path)
    t = d["t"]
    n = len(t)
    dur = t[-1] - t[0]
    dt = np.diff(t)
    rate = (n - 1) / dur
    gaps = int(np.sum(dt > 3.0 / rate))
    gyro = np.stack([d["gx_dps"], d["gy_dps"], d["gz_dps"]], 1)
    acc = np.stack([d["ax_g"], d["ay_g"], d["az_g"]], 1)
    prof = _profile()
    pg, pa = prof.get("gyro", {}), prof.get("accel", {})
    L: list[str] = []
    p = L.append

    p(f"WTGAHRS1 static characterisation — {path.name}")
    p("=" * 78)
    p(f"duration {dur / 60:.1f} min · {n} samples · {rate:.2f} Hz effective · {gaps} gaps > 3 samples")
    temp = d["temp_c"]
    p(f"temperature {np.nanmin(temp):.2f} … {np.nanmax(temp):.2f} °C (start {temp[0]:.2f}, end {temp[-1]:.2f})")
    moved = _motion_windows(gyro, rate)
    p(f"motion check: {int(moved.sum())} of {len(moved)} one-second windows exceed 1.5 °/s"
      + ("  <-- disturbed, treat the long-tau terms with caution" if moved.any() else "  (clean)"))
    zero_frac = (gyro == 0).mean(0)
    p(f"gyro samples exactly 0: x {zero_frac[0]:.1%} y {zero_frac[1]:.1%} z {zero_frac[2]:.1%}"
      + ("  <-- auto-zero clamp still active?" if zero_frac.max() > 0.5 else ""))
    p("")

    # ---- raw statistics
    p("Raw statistics (sensor axes)")
    p(f"{'':12}{'mean':>12}{'std':>11}{'std/LSB':>9}{'levels':>8}")
    for i, ax in enumerate("xyz"):
        g = gyro[:, i]
        p(f"gyro {ax} °/s {g.mean():12.4f}{g.std():11.4f}{g.std() / GYRO_LSB:9.2f}{len(np.unique(g)):8d}")
    for i, ax in enumerate("xyz"):
        a = acc[:, i]
        p(f"acc  {ax} mg  {a.mean() * 1e3:12.3f}{a.std() * 1e3:11.3f}{a.std() / ACC_LSB_G:9.2f}{len(np.unique(a)):8d}")
    gmag = np.linalg.norm(acc.mean(0))
    p(f"|a| = {gmag:.5f} g  (1.0000 expected; deviation {(gmag - 1) * 1e3:+.2f} mg = scale/offset error on the up axis)")
    tilt = math.degrees(math.acos(min(1.0, abs(acc.mean(0)[2]) / gmag)))
    p(f"tilt of the bench from the accelerometer: {tilt:.3f}°")
    p("")

    # ---- Allan
    fits = {}
    p("Allan-variance fit (simulator parameterisation) vs imu_wtgahrs1.yaml")
    p(f"{'':30}{'x':>10}{'y':>10}{'z':>10}{'mean':>10}{'profile':>10}")
    curves = {}
    for name, data, scale in (("gyro", gyro, 1.0), ("accel", acc, 1.0)):
        fits[name] = []
        curves[name] = []
        for i in range(3):
            taus, adev = A.overlapping_allan_deviation(data[:, i] * scale, rate)
            curves[name].append((taus, adev))
            # a constant channel (e.g. a firmware-zeroed gyro axis) has no Allan curve to fit
            keep = taus >= FIT_TAU_MIN_S
            fits[name].append(A.fit_allan(taus[keep], adev[keep]) if np.all(adev > 0) else None)

    def row(label, key, name, k, prof_val, fmt="{:10.4g}"):
        vals = [None if f is None else f[key] * k for f in fits[name]]
        good = [v for v in vals if v is not None]
        mean = float(np.mean(good)) if good else float("nan")
        pv = "—" if prof_val is None else fmt.format(prof_val).strip()
        p(f"{label:30}" + "".join(f"{'const':>10}" if v is None else fmt.format(v) for v in vals) + fmt.format(mean) + f"{pv:>10}")
        return mean

    res = {}
    res["g_nd"] = row("gyro noise density °/s/√Hz", "noise_density", "gyro", 1, pg.get("noise_density_dps_rthz"))
    res["g_bi"] = row("gyro bias instability °/s", "bias_instability", "gyro", 1, pg.get("bias_instability_dps"))
    res["g_tc"] = row("gyro bias corr. time s", "bias_correlation_time_s", "gyro", 1, pg.get("bias_correlation_time_s"))
    res["g_rw"] = row("gyro rate random walk °/s/√s", "rate_random_walk", "gyro", 1, pg.get("rate_random_walk_dps_rt_s"))
    res["a_nd"] = row("accel noise density µg/√Hz", "noise_density", "accel", 1e6, pa.get("noise_density_ug_rthz"))
    res["a_bi"] = row("accel bias instability mg", "bias_instability", "accel", 1e3, pa.get("bias_instability_mg"))
    res["a_tc"] = row("accel bias corr. time s", "bias_correlation_time_s", "accel", 1, pa.get("bias_correlation_time_s"))
    res["a_rw"] = row("accel random walk µg/√s", "rate_random_walk", "accel", 1e6, pa.get("random_walk_ug_rt_s"))
    for name, k, unit in (("gyro", 1, "°/s"), ("accel", 1e3, "mg")):
        s1 = []
        for taus, adev in curves[name]:
            s1.append(float(np.interp(0.0, np.log(taus), adev)) * k if np.all(adev > 0) else None)   # sigma at tau = 1 s
        mins = [(float(adev.min()) * k, float(taus[np.argmin(adev)])) for taus, adev in curves[name] if np.all(adev > 0)]
        p(f"{name} σ(1 s): " + ", ".join("const" if v is None else f"{v:.4g}" for v in s1) + f" {unit}  (noise density = √2·σ(1 s))"
          + " · Allan floor: " + ", ".join(f"{m:.3g} {unit} @ {t:.0f} s" for m, t in mins))
    p(f"fit uses τ ≥ {FIT_TAU_MIN_S} s (the on-sensor low-pass shapes shorter τ)")
    rr = lambda fs: ", ".join("—" if f is None else f"{f['relative_rms_error']:.3f}" for f in fs)  # noqa: E731
    p(f"fit residual (relative rms): gyro {rr(fits['gyro'])} · accel {rr(fits['accel'])}")
    for name in ("gyro", "accel"):
        const = ["xyz"[i] for i, f in enumerate(fits[name]) if f is None]
        if const:
            p(f"{name} axis {', '.join(const)} never changed value: no noise information (means exclude it)")
    arw = res["g_nd"] / math.sqrt(2) * 60                       # N (°/√s) -> °/√h
    p(f"angle random walk ≈ {arw:.3f} °/√h · velocity random walk ≈ {res['a_nd'] / 1e6 / math.sqrt(2) * 9.80665 * 60:.4f} m/s/√h")
    for name, lsb, unit in (("gyro", GYRO_LSB, "°/s"), ("accel", ACC_LSB_G, "g")):
        q = lsb / math.sqrt(12) / math.sqrt(rate / 2)          # quantisation noise density at this output rate
        p(f"{name} quantisation floor ≈ {q:.3g} {unit}/√Hz (LSB {lsb:.3g} {unit}): results near this are LSB-limited")
    p(f"note: the longest averaging time is {curves['gyro'][0][0][-1]:.0f} s; bias instability / correlation time / random "
      f"walk are only constrained below that (the repo tool asks for >= 6 h).")
    p("")

    # ---- temperature sensitivity
    if np.nanmax(temp) - np.nanmin(temp) > 0.5:
        k = np.isfinite(temp)
        p("Bias vs temperature (linear fit over the run)")
        for i, ax in enumerate("xyz"):
            sg = np.polyfit(temp[k], gyro[k, i], 1)[0] if gyro[k, i].std() > 0 else 0.0
            sa = np.polyfit(temp[k], acc[k, i], 1)[0] * 1e3
            p(f"  {ax}: gyro {sg:+.4f} °/s/K   accel {sa:+.3f} mg/K")
        p(f"  profile assumes {pg.get('tco_dps_per_k', '—')} °/s/K and {pa.get('tco_mg_per_k', '—')} mg/K")
    else:
        p(f"Temperature changed only {np.nanmax(temp) - np.nanmin(temp):.2f} K: no temperature coefficient measurable.")
    p("")

    # ---- onboard attitude, magnetometer, barometer
    roll, pitch, yaw = d["roll_deg"], d["pitch_deg"], d["yaw_deg"]
    yaw_u = np.degrees(np.unwrap(np.radians(yaw)))
    drift = np.polyfit(t / 60, yaw_u, 1)[0]
    p("Onboard filter (sensor's own attitude output)")
    p(f"  roll  mean {np.nanmean(roll):+.3f}° std {np.nanstd(roll):.4f}° span {np.nanmax(roll) - np.nanmin(roll):.3f}°")
    p(f"  pitch mean {np.nanmean(pitch):+.3f}° std {np.nanstd(pitch):.4f}° span {np.nanmax(pitch) - np.nanmin(pitch):.3f}°")
    p(f"  yaw   drift {drift:+.3f} °/min  (total {yaw_u[-1] - yaw_u[0]:+.2f}° over the run; 9-axis yaw is held by the magnetometer)")
    mag = np.stack([d["mx"], d["my"], d["mz"]], 1)
    if np.isfinite(mag).all():
        p(f"Magnetometer: mean {np.round(mag.mean(0), 1).tolist()} counts, noise std {np.round(mag.std(0), 2).tolist()} counts, "
          f"|m| {np.linalg.norm(mag.mean(0)):.1f} counts")
    pr, bh = d["pressure_pa"], d["baro_h_m"]
    if np.isfinite(pr).any():
        k = np.isfinite(pr)
        tt = t[k]
        trend = np.polyfit(tt / 60, bh[k], 1)[0]
        w = max(int(rate * 3), 1)
        short = np.nanmedian([np.std(bh[k][i:i + w]) for i in range(0, k.sum() - w, w)])
        p(f"Barometer: pressure {np.nanmean(pr) / 100:.2f} hPa, std {np.nanstd(pr):.2f} Pa, "
          f"{len(np.unique(pr[k]))} distinct values")
        p(f"  height noise (3 s windows) {short:.3f} m (profile noise_std 0.15 m) · drift {trend * 60:+.2f} m/h "
          f"(total {bh[k][-1] - bh[k][0]:+.2f} m; weather + warm-up)")
    report = "\n".join(L)

    # ---- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, name, k, unit in ((axs[0], "gyro", 1, "°/s"), (axs[1], "accel", 1e3, "mg")):
            for i, (c, lab) in enumerate(zip(("#2563eb", "#d97706", "#0d9488"), "xyz")):
                taus, adev = curves[name][i]
                if np.all(adev > 0):
                    ax.loglog(taus, adev * k, color=c, lw=1.6, label=lab)
            ax.set_title(f"{name} Allan deviation")
            ax.set_xlabel("τ (s)")
            ax.set_ylabel(f"σ(τ) ({unit})")
            ax.grid(True, which="both", alpha=.25)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".allan.png"), dpi=130)
        fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        tm = t / 60
        ds = max(len(t) // 4000, 1)
        for i, (c, lab) in enumerate(zip(("#2563eb", "#d97706", "#0d9488"), "xyz")):
            axs[0].plot(tm[::ds], gyro[::ds, i] - gyro[:, i].mean(), color=c, lw=.6, label=f"{lab} − {gyro[:, i].mean():+.3f}")
            axs[1].plot(tm[::ds], (acc[::ds, i] - acc[:, i].mean()) * 1e3, color=c, lw=.6, label=lab)
        axs[0].set_ylabel("gyro − mean (°/s)")
        axs[1].set_ylabel("accel − mean (mg)")
        axs[2].plot(tm, temp, color="#7c3aed", lw=1.2)
        axs[2].set_ylabel("temperature (°C)")
        axs[2].set_xlabel("time (min)")
        for a in axs:
            a.grid(True, alpha=.25)
        axs[0].legend(frameon=False, ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".timeseries.png"), dpi=130)
    except Exception as e:  # plots are optional
        report += f"\n(plots skipped: {e})"
    path.with_suffix(".report.txt").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--minutes", type=float, default=30.0)
    r.add_argument("--port")
    r.add_argument("--disable-autozero", action="store_true", help="send FF AA 63 01 00 + save first (writes the sensor's flash)")
    r.add_argument("--out", type=Path)
    r.add_argument("--analyze", action="store_true", help="run the analysis when the capture ends")
    a = sub.add_parser("analyze")
    a.add_argument("log", type=Path)
    args = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "record":
        out = record(args.minutes, args.port, args.disable_autozero, args.out)
        if args.analyze:
            print(analyze(out))
    else:
        print(analyze(args.log))


if __name__ == "__main__":
    main()
