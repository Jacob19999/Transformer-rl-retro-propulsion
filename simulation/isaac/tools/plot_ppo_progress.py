"""Plot comparable full-task evaluation metrics across PPO run directories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for run in args.runs:
        records = [json.loads(line) for line in (run / 'eval_log.jsonl').read_text().splitlines()]
        # final_eval may repeat the last evaluation; one point per checkpoint.
        records = list({r['global_step']: r for r in records}.values())
        x = [r['global_step'] / 1e6 for r in records]
        label = run.parent.name
        line, = axes[0].plot(x, [100 * r['success_fraction'] for r in records], 'o-', label=label)
        axes[1].plot(x, [100 * r['crashed_fraction'] for r in records], 'o-', color=line.get_color(), label=label + ': crashes')
        axes[1].plot(x, [100 * r['timeout_fraction'] for r in records], '--', color=line.get_color(), label=label + ': timeouts')
    axes[0].axhline(80, color='#64748b', linewidth=1, linestyle=':', label='80% success gate')
    axes[0].set_title('Deterministic evaluation on the full cold-start landing task')
    axes[0].set_ylabel('Successful landings (%)')
    axes[1].set_ylabel('Episode fraction (%)')
    axes[1].set_xlabel('Cumulative training transitions (millions)')
    for ax in axes:
        ax.set_ylim(-2, 102)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc='best')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == '__main__':
    main()
