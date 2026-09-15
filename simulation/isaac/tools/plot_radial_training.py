"""Plot full-task reliability and physical efficiency without mixing spawn stages."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluations = {}
    modes = set()
    for run in args.runs:
        settings = json.loads((run / 'args.json').read_text(encoding='utf-8'))
        modes.add(settings.get('eval_action_mode', 'deterministic'))
        path = run / 'eval_log.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                evaluations[row['global_step']] = row
    if len(modes) != 1:
        raise ValueError('Compare one inference mode at a time')
    rows = sorted(evaluations.values(), key=lambda r: r['global_step'])
    x = [r['global_step'] / 1e6 for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    fig.suptitle(f'Radial hinges / planned 8S / {next(iter(modes))} PPO\nFull 16–20 m cold-rotor task', fontsize=15)
    axes[0,0].plot(x, [100*r['success_fraction'] for r in rows], 'o-', color='#147d73')
    axes[0,0].axhline(80, color='#64748b', linestyle=':', label='80% gate')
    axes[0,0].set_ylabel('Successful landings (%)')
    axes[0,0].set_ylim(-2,102)
    axes[0,0].legend(fontsize=9)
    axes[0,1].plot(x, [100*r['crashed_fraction'] for r in rows], 'o-', color='#b85546', label='Crash / safety failure')
    axes[0,1].plot(x, [100*r['timeout_fraction'] for r in rows], 's--', color='#a98440', label='Timeout')
    axes[0,1].axhline(5, color='#64748b', linestyle=':', label='5% crash gate')
    axes[0,1].set_ylabel('Episodes (%)')
    axes[0,1].set_ylim(-2,102)
    axes[0,1].legend(fontsize=9)
    for ax, key, label, color in (
        (axes[1,0], 'success_mean_energy_wh', 'Mean electrical work (Wh)', '#335f9e'),
        (axes[1,1], 'success_mean_propulsive_delta_v_m_s', 'Mean propulsive delta-v (m/s)', '#7c5295')):
        ax.plot(x, [r.get(key, float('nan')) for r in rows], 'o-', color=color)
        ax.set_title('Successful episodes only', fontsize=10)
        ax.set_ylabel(label)
        ax.set_xlabel('Cumulative training transitions (millions)')
        ax.set_ylim(bottom=0)
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == '__main__':
    main()
