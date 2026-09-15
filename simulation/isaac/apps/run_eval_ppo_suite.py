"""Sequential independent landing, disturbance and battery validation suite."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--comparison-checkpoint')
    parser.add_argument('--env-config', default='configs/env/train_512_8s_radial.yaml')
    parser.add_argument('--action-mode', choices=['stochastic', 'deterministic', 'mean'], default='stochastic')
    parser.add_argument('--nominal-episodes', type=int, default=2048)
    parser.add_argument('--disturbance-episodes', type=int, default=512)
    parser.add_argument('--swirl-sensitivity', action='store_true',
                        help='Add residual swirl fractions 0 and 0.2 for the coupled-jet plant')
    parser.add_argument('--seed', type=int, default=7411)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / 'suite_config.json').write_text(json.dumps(vars(args), indent=2))
    cases = [('nominal', 'nominal', 1., args.nominal_episodes),
             ('wind', 'wind', 1., args.disturbance_episodes),
             ('sensor_noise', 'sensor_noise', 1., args.disturbance_episodes),
             ('com_shift', 'com_shift', 1., args.disturbance_episodes),
             ('combined', 'combined', 1., args.disturbance_episodes),
             ('charge_50', 'nominal', .5, args.disturbance_episodes),
             ('combined_charge_80', 'combined', .8, args.disturbance_episodes)]
    summaries = {}
    def evaluate(name, disturbance, soc, episodes, checkpoint, residual_swirl=None):
        folder = output / name
        command = [sys.executable, str(root / 'apps/run_eval_ppo_batch.py'),
                   '--checkpoint', checkpoint, '--env-config', args.env_config,
                   '--disturbance', f'configs/disturbances/{disturbance}.yaml',
                   '--battery-soc', str(soc), '--episodes', str(episodes),
                   '--seed', str(args.seed), '--action-mode', args.action_mode,
                   '--output-dir', str(folder), '--max-wall-time', '1200']
        if residual_swirl is not None:
            command += ['--residual-swirl', str(residual_swirl)]
        print(f'START {name}: {episodes} independent episodes, SOC {soc:.0%}', flush=True)
        with (output / f'{name}.log').open('w') as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'{name} evaluator failed with code {completed.returncode}; see its log')
        summary = json.loads((folder / 'summary.json').read_text())
        summaries[name] = summary
        (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
        print(f"DONE {name}: success={summary['success_fraction']:.2%}, crashes={summary['crashed_fraction']:.2%}, pass={summary['passed']}", flush=True)
    for case in cases:
        evaluate(*case, args.checkpoint)
    if args.swirl_sensitivity:
        for fraction in (0., .2):
            evaluate(f'swirl_{fraction:.1f}', 'nominal', 1., args.disturbance_episodes,
                     args.checkpoint, residual_swirl=fraction)
    if args.comparison_checkpoint:
        evaluate('comparison_nominal', 'nominal', 1., args.nominal_episodes, args.comparison_checkpoint)
        records = lambda name: {r['episode']: r for r in
                               (json.loads(line) for line in (output / name / 'episodes.jsonl').read_text().splitlines())}
        before, after = records('comparison_nominal'), records('nominal')
        # Equal reset seeds alone are not proof of equal initial trials: also
        # check the recorded states before publishing paired efficiency gains.
        for episode, old in before.items():
            for field in ('reset_seed', 'batch_env', 'spawn_position', 'spawn_quaternion',
                          'spawn_body_velocity', 'spawn_motor_fraction'):
                if old[field] != after[episode][field]:
                    raise ValueError(f'Comparison trial {episode} differs in {field}')
        common = [i for i in before if before[i]['success'] and after[i]['success']]
        comparison = dict(common_success_episodes=len(common), total_episodes=args.nominal_episodes,
                          success_before=summaries['comparison_nominal']['success_fraction'],
                          success_after=summaries['nominal']['success_fraction'],
                          interpretation='Efficiency conditional on both policies succeeding on matched initial trials; compare success rates separately.')
        for key in ('energy_wh', 'propulsive_delta_v_m_s', 'duration_s'):
            if common:
                old = sum(before[i][key] for i in common) / len(common)
                new = sum(after[i][key] for i in common) / len(common)
                comparison[key] = dict(before=old, after=new, reduction_percent=100 * (old-new) / old if old else None)
            else:
                comparison[key] = None
        (output / 'efficiency_comparison.json').write_text(json.dumps(comparison, indent=2))
    print(f'Validation artifacts: {output}', flush=True)


if __name__ == '__main__':
    main()
