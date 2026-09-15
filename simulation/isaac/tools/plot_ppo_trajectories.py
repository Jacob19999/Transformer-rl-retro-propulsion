"""Plot saved single-environment PPO flight traces without replaying physics."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(s) for s in (args.run / 'trajectory.jsonl').read_text().splitlines()]
    summary = json.loads((args.run / 'summary.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ep in sorted({r['episode'] for r in records}):
        rows = [r for r in records if r['episode'] == ep]
        t = np.array([r['time_s'] for r in rows])
        p = np.array([r['position'] for r in rows])
        v = np.array([r['velocity_world'] for r in rows])
        q = np.array([r['observation_after'][3:7] for r in rows])
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1:3] ** 2).sum(1), -1, 1)))
        line, = axes[0, 0].plot(t, p[:, 2], label=f'Trial {ep}')
        color = line.get_color()
        axes[0, 1].plot(p[:, 0], p[:, 1], color=color)
        axes[0, 1].scatter(p[0, 0], p[0, 1], color=color, marker='x', s=35)
        axes[0, 1].scatter(p[-1, 0], p[-1, 1], color=color, s=18)
        axes[1, 0].plot(t, -v[:, 2], color=color)
        axes[1, 1].plot(t, tilt, color=color)
    axes[0, 0].set(xlabel='Time (s)', ylabel='Root altitude (m)')
    axes[0, 0].legend(fontsize=8, ncol=2)
    radius = summary['success_max_pad_distance']
    axes[0, 1].add_patch(plt.Circle((0, 0), radius, facecolor='#94a3b8', alpha=.18, edgecolor='#475569'))
    axes[0, 1].set(xlabel='World X (m)', ylabel='World Y (m)', title='X = spawn; dot = final position')
    axes[0, 1].set_aspect('equal', adjustable='datalim')
    axes[1, 0].set(xlabel='Time (s)', ylabel='Downward speed (m/s)')
    axes[1, 0].axhline(summary['success_max_touchdown_speed'], color='#475569', linestyle=':', linewidth=1)
    axes[1, 1].set(xlabel='Time (s)', ylabel='Tilt from upright (degrees)')
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.suptitle(f"PPO landing replay: {summary.get('action_mode', 'deterministic')} actions, "
                 f"{summary['episodes']} trials, seed {summary.get('seed', '?')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == '__main__':
    main()
