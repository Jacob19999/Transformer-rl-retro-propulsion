"""Plot curriculum and full-task PPO outcomes without conflating them."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_records(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train = read_records(args.run/'train_log.jsonl')
    full = read_records(args.run/'eval_log.jsonl')
    stages = read_records(args.run/'curriculum_eval.jsonl')
    if not train:
        raise ValueError('No training updates found')
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    x = np.array([r['global_step']/1e6 for r in train])
    axes[0,0].plot(x, [r['stage_success_count']/max(r['stage_termination_count'],1)
                       for r in train], color='#0284c7', label='Training spawn: rollout success')
    if full:
        axes[0,0].plot([r['global_step']/1e6 for r in full],
                       [r['success_fraction'] for r in full], 'o-', color='#111827', label='Full cold task: evaluation')
    axes[0,0].set(ylabel='Soft on-pad success fraction', ylim=(-.03,1.03))
    axes[0,0].legend(fontsize=8)
    for key, label, color in [('rollout_crashed_count','Crash','#dc2626'),
                              ('rollout_timeout_count','Timeout','#d97706')]:
        axes[0,1].plot(x, [r[key]/max(r['stage_termination_count'],1) for r in train],
                       label=label, color=color)
    axes[0,1].set(ylabel='Training terminal fraction', ylim=(-.03,1.03))
    axes[0,1].legend(fontsize=8)
    axes[1,0].plot(x, [r['explained_variance'] for r in train], color='#7c3aed')
    axes[1,0].set(ylabel='Critic explained variance', ylim=(-.1,1.03))
    axes[1,1].plot(x, [r['policy_kl_final'] for r in train], color='#7c3aed')
    axes[1,1].set(ylabel='KL after PPO update')
    for ax, key, label in [(axes[2,0], 'success_mean_energy_wh', 'Successful episode energy (Wh)'),
                            (axes[2,1], 'success_mean_propulsive_delta_v_m_s', 'Successful propulsive impulse/mass (m/s)')]:
        for name, rows, color in [('Current spawn evaluation', stages, '#0284c7'),
                                  ('Full cold task evaluation', full, '#111827')]:
            valid = [r for r in rows if np.isfinite(r.get(key, np.nan))]
            if valid:
                ax.plot([r['global_step']/1e6 for r in valid], [r[key] for r in valid],
                        'o-', color=color, label=name)
        ax.set(ylabel=label)
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(.5,.5,'No successful evaluation episodes yet',ha='center',transform=ax.transAxes)
    transitions = [r for r in train if r['stage_advanced_this_update']]
    for ax in axes.flat:
        ax.set_xlabel('Total transitions (millions, including resumed checkpoint)')
        ax.grid(alpha=.18)
        for row in transitions:
            ax.axvline(row['global_step']/1e6,color='#64748b',ls=':',lw=.8)
    for row in transitions:
        axes[0,0].annotate(f"Stage {row['spawn_stage_index']}",
            (row['global_step']/1e6,.98), fontsize=7,rotation=90,va='top')
    fig.suptitle('Direct-action PPO on the corrected 8S simulation\n'
                 'Curriculum success does not establish full cold-start landing performance',fontsize=13)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
