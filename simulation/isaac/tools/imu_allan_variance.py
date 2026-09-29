"""Fit IMU noise parameters from a static log and print a profile snippet.

    python tools/imu_allan_variance.py log.csv --rate-hz 200 \
        --gyro gx,gy,gz --gyro-unit dps --accel ax,ay,az --accel-unit g

The log must be recorded with the sensor at rest (>= 6 h recommended), temperature
settled, EDF off; repeat with the EDF spinning to see vibration effects. --gyro/--accel
take three header names or column indices. The printed YAML keys match
configs/sensors/imu_wtgahrs1.yaml; paste the values over its PLACEHOLDER lines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvc_env.dynamics.imu_allan import fit_allan, overlapping_allan_deviation  # noqa: E402

# Scale from each log unit to the profile's working unit (dps for gyro, g for accelerometer).
UNITS = {"dps": 1.0, "rads": 180.0 / np.pi, "g": 1.0, "mg": 1e-3}


def load_columns(path: Path, names: list[str]) -> np.ndarray:
    header = path.open().readline().strip().split(",")
    has_header = not all(_is_number(h) for h in header)
    data = np.genfromtxt(path, delimiter=",", skip_header=1 if has_header else 0)
    cols = []
    for name in names:
        if name.isdigit():
            cols.append(int(name))
        elif name in header:
            cols.append(header.index(name))
        else:
            raise SystemExit(f"column {name!r} not in header {header}")
    return data[:, cols]


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def characterise(data: np.ndarray, rate_hz: float, scale: float):
    """Per-axis fits of an (N, 3) log, converted by ``scale`` into the profile's units."""
    fits = []
    for axis in range(3):
        taus, adev = overlapping_allan_deviation(data[:, axis] * scale, rate_hz)
        fits.append(fit_allan(taus, adev))
    return fits


def _mean(fits, key):
    return float(np.mean([f[key] for f in fits]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--rate-hz", type=float, required=True)
    parser.add_argument("--gyro", help="three gyro column names or indices")
    parser.add_argument("--gyro-unit", choices=["dps", "rads"], default="dps")
    parser.add_argument("--accel", help="three accelerometer column names or indices")
    parser.add_argument("--accel-unit", choices=["g", "mg"], default="g")
    args = parser.parse_args()
    if not args.gyro and not args.accel:
        parser.error("give --gyro and/or --accel")

    print("imu:")
    if args.gyro:
        fits = characterise(load_columns(args.log, args.gyro.split(",")), args.rate_hz, UNITS[args.gyro_unit])
        print("  gyro:")
        print(f"    noise_density_dps_rthz: {_mean(fits, 'noise_density'):.5g}")
        print(f"    bias_instability_dps: {_mean(fits, 'bias_instability'):.5g}")
        print(f"    bias_correlation_time_s: {_mean(fits, 'bias_correlation_time_s'):.5g}")
        print(f"    rate_random_walk_dps_rt_s: {_mean(fits, 'rate_random_walk'):.5g}")
        print(f"    # per-axis fit residual (relative rms): {[round(f['relative_rms_error'], 3) for f in fits]}")
    if args.accel:
        fits = characterise(load_columns(args.log, args.accel.split(",")), args.rate_hz, UNITS[args.accel_unit])
        print("  accel:")
        print(f"    noise_density_ug_rthz: {_mean(fits, 'noise_density') * 1e6:.5g}")
        print(f"    bias_instability_mg: {_mean(fits, 'bias_instability') * 1e3:.5g}")
        print(f"    bias_correlation_time_s: {_mean(fits, 'bias_correlation_time_s'):.5g}")
        print(f"    random_walk_ug_rt_s: {_mean(fits, 'rate_random_walk') * 1e6:.5g}")
        print(f"    # per-axis fit residual (relative rms): {[round(f['relative_rms_error'], 3) for f in fits]}")
    print("# turn_on_bias / scale_factor / misalignment are not observable from a static log: measure them "
          "with a multi-position calibration and a turntable.")


if __name__ == "__main__":
    main()
