"""
benchmark_envs.py — Throughput benchmark across env counts (T030).

Implements T030 / User Story 3 verification:
- Run 1000 steps at num_envs ∈ [1, 128, 512, 1024]
- Log wall-clock steps/s to CSV
- Assert 128-env throughput ≥ 10× single-env (SC-004)

Usage::
    python -m simulation.isaac.scripts.benchmark_envs
    python -m simulation.isaac.scripts.benchmark_envs --steps 500 --output benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

_ENV_COUNTS = [1, 128, 512, 1024]
_DEFAULT_STEPS = 1000


def benchmark_num_envs(num_envs: int, steps: int) -> float:
    """Run `steps` environment steps with `num_envs` parallel envs; return steps/s."""
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    # Pick appropriate config
    if num_envs == 1:
        cfg = REPO_ROOT / "simulation" / "isaac" / "configs" / "isaac_env_single.yaml"
    elif num_envs <= 128:
        cfg = REPO_ROOT / "simulation" / "isaac" / "configs" / "isaac_env_128.yaml"
    else:
        cfg = REPO_ROOT / "simulation" / "isaac" / "configs" / "isaac_env_1024.yaml"

    env = EDFIsaacEnv(config_path=cfg, seed=0)
    env.reset()

    if num_envs == 1:
        action = np.zeros(5, dtype=np.float32)
    else:
        action = np.zeros((num_envs, 5), dtype=np.float32)

    # Warm-up
    for _ in range(10):
        env.step(action)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(action)
    elapsed = time.perf_counter() - t0

    env.close()
    return (num_envs * steps) / elapsed  # environment-steps per second


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim parallel env throughput benchmark")
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS,
                        help="Steps per num_envs configuration")
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="CSV output path")
    parser.add_argument(
        "--env-counts", nargs="+", type=int, default=_ENV_COUNTS,
        help="List of num_envs to benchmark",
    )
    args = parser.parse_args()

    results: list[dict] = []
    baseline_sps: float | None = None

    for n in args.env_counts:
        print(f"[benchmark] num_envs={n:4d} — running {args.steps} steps ...", end="", flush=True)
        try:
            sps = benchmark_num_envs(n, args.steps)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"num_envs": n, "steps_per_s": float("nan"), "speedup_vs_1": float("nan")})
            continue

        if n == 1:
            baseline_sps = sps

        speedup = sps / baseline_sps if baseline_sps else float("nan")
        print(f"  {sps:,.0f} steps/s  (×{speedup:.1f} vs single-env)")
        results.append({"num_envs": n, "steps_per_s": sps, "speedup_vs_1": speedup})

    # Write CSV
    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_envs", "steps_per_s", "speedup_vs_1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[benchmark] Results saved to: {out_path}")

    # SC-004 assertion: 128-env ≥ 10× single-env
    r128 = next((r for r in results if r["num_envs"] == 128), None)
    if r128 and not np.isnan(r128["speedup_vs_1"]):
        if r128["speedup_vs_1"] >= 10.0:
            print(f"[benchmark] PASS SC-004: 128-env speedup = {r128['speedup_vs_1']:.1f}× ≥ 10×")
        else:
            print(f"[benchmark] FAIL SC-004: 128-env speedup = {r128['speedup_vs_1']:.1f}× < 10×")
            sys.exit(1)


if __name__ == "__main__":
    main()
