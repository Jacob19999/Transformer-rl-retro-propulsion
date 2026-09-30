"""How quickly does the fused height follow the vertical motion? Analyse the portal's last 60 s (or a fusion.csv).

    python hardware/sensor_fusion/response_check.py                 # live portal on :8002
    python hardware/sensor_fusion/response_check.py fusion.csv      # a CSV from the page's Export or --log

Do a few quick descents (and climbs) by hand first. Reports, over the moving parts only: the time the fused height lags the
tilt-compensated raw TFmini range, the largest disagreement, and how often the filter stopped using good range readings
("freezes": consecutive readings inside 0.1-8 m that the innovation gate rejected). A healthy result is a lag of a few tens
of ms, a disagreement of a few cm and no freezes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request

import numpy as np


def load(src: str | None, url: str) -> dict[str, np.ndarray]:
    if src:
        with open(src, newline="") as fh:
            rows = list(csv.reader(fh))
        head = rows[0][1:] if rows[0][0] == "pc_time_s" else rows[0]
        body = [r[1:] if rows[0][0] == "pc_time_s" else r for r in rows[1:]]
    else:
        with urllib.request.urlopen(f"{url}/samples?after=0&limit=6000", timeout=10) as r:
            d = json.load(r)
        head, body = d["fields"], d["samples"]
    def col(name):
        i = head.index(name)
        return np.array([float(r[i]) if r[i] not in ("", None) else np.nan for r in body], float)
    return {k: col(k) for k in ("t", "range_m", "range_ok", "h_fused", "v_fused", "roll", "pitch", "nis", "mode")}


def analyse(d: dict[str, np.ndarray]) -> str:
    t = d["t"]
    order = np.argsort(t)
    d = {k: v[order] for k, v in d.items()}
    t = d["t"]
    dur = t[-1] - t[0]
    out = [f"{len(t)} rows over {dur:.1f} s ({(len(t) - 1) / dur:.0f} Hz)"]
    tilt = np.cos(np.radians(d["roll"])) * np.cos(np.radians(d["pitch"]))
    raw = d["range_m"] * np.where(np.isfinite(tilt), tilt, 1.0)             # tilt-compensated raw height
    inband = np.isfinite(d["range_m"]) & (d["range_m"] >= 0.10) & (d["range_m"] <= 8.0)
    # speed of the raw range, smoothed over ~0.15 s
    k = max(int(0.15 * (len(t) - 1) / dur), 1)
    ok = np.isfinite(raw)
    rs = np.where(ok, raw, np.nan)
    speed = np.full(len(t), np.nan)
    for i in range(k, len(t) - k):
        if np.isfinite(rs[i - k]) and np.isfinite(rs[i + k]):
            speed[i] = (rs[i + k] - rs[i - k]) / (t[i + k] - t[i - k])
    moving = np.abs(np.nan_to_num(speed)) > 0.3
    # dilate by 0.25 s so the settle after a stop is included
    kk = int(0.25 * (len(t) - 1) / dur)
    moving = np.convolve(moving.astype(float), np.ones(2 * kk + 1), "same") > 0
    out.append(f"raw range speed: max {np.nanmax(np.abs(speed)):.2f} m/s · moving (>0.3 m/s) for {moving.mean() * dur:.1f} s")
    # freezes: runs of in-band readings the filter did not use
    # only while the filter has a height (not the start-up alignment) and the vehicle is within the tilt limit
    rejected = inband & (d["range_ok"] == 0) & np.isfinite(d["h_fused"]) & (np.nan_to_num(tilt, nan=1.0) > math.cos(math.radians(60)))
    runs, i = [], 0
    while i < len(t):
        if rejected[i]:
            j = i
            while j + 1 < len(t) and rejected[j + 1]:
                j += 1
            runs.append((t[i], t[j] - t[i] + 1.0 / ((len(t) - 1) / dur)))
            i = j + 1
        else:
            i += 1
    long_runs = [r for r in runs if r[1] >= 0.05]
    out.append(f"freezes (good range not used): {len(long_runs)} runs ≥ 50 ms"
               + (f", longest {max(r[1] for r in long_runs) * 1000:.0f} ms" if long_runs else "")
               + f" · {int(rejected.sum())} of {int(inband.sum())} in-band readings rejected in total")
    if moving.sum() < 30:
        out.append("not enough vertical motion to measure lag: raise/lower the vehicle by ≥ 0.3 m at ≥ 0.3 m/s and rerun")
        return "\n".join(out)
    m = moving & ok & np.isfinite(d["h_fused"])
    err = d["h_fused"][m] - raw[m]
    out.append(f"while moving: fused − raw range  mean {err.mean() * 100:+.1f} cm · rms {np.sqrt(np.mean(err ** 2)) * 100:.1f} cm · "
               f"max {np.abs(err).max() * 100:.1f} cm")
    best = None
    for lag in np.arange(-0.10, 0.401, 0.005):
        est = np.interp(t[m] - lag, t, d["h_fused"])                        # fused delayed by `lag` vs the raw range
        e = np.mean((est - raw[m]) ** 2)
        if best is None or e < best[0]:
            best = (e, lag)
    out.append(f"time offset of the fused height relative to the raw range: {best[1] * 1000:+.0f} ms "
               f"(negative = fused leads the raw range, which is what IMU aiding should do; positive = fused lags)")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    args = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    print(analyse(load(args.csv, args.url)))


if __name__ == "__main__":
    main()
