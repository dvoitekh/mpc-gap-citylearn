"""
Full 3-phase evaluation for CityLearn 2022 Challenge.

Phase 1: Buildings 1-5, days 0-120 (weight 0.2)
Phase 2: Buildings 1-5, days 120-240 (weight 0.3)
Phase 3: Buildings 6-17 (no 12,15), days 240-365 (weight 0.5)
"""

import numpy as np
from mpcgap.environment import create_env, evaluate_env


PHASES = [
    {
        'name': 'Phase 1',
        'buildings': [f'Building_{i}' for i in range(1, 6)],
        'sim_start': 0,
        'sim_end': 24 * 120 - 1,
        'weight': 0.2,
    },
    {
        'name': 'Phase 2',
        'buildings': [f'Building_{i}' for i in range(1, 6)],
        'sim_start': 24 * 120,
        'sim_end': 24 * 240 - 1,
        'weight': 0.3,
    },
    {
        'name': 'Phase 3',
        'buildings': [f'Building_{i}' for i in range(1, 18) if i not in [12, 15]],
        'sim_start': 24 * 240,
        'sim_end': 24 * 365 - 1,
        'weight': 0.5,
    },
]


def evaluate_baseline():
    """Evaluate baseline (no control, action=0)."""
    print("Evaluating Baseline (no control)...")
    phase_results = []
    weighted_score = 0

    for phase in PHASES:
        config = {
            'dataset_name': 'citylearn_challenge_2022_phase_all',
            'buildings': phase['buildings'],
            'sim_start': phase['sim_start'],
            'sim_end': phase['sim_end'],
            'central_agent': True,
        }

        env = create_env(config)
        obs, _ = env.reset()
        n_buildings = len(phase['buildings'])

        while not env.terminated:
            action = [[0.0] * n_buildings]
            obs, _, _, _, _ = env.step(action)

        metrics = evaluate_env(env)
        phase_results.append(metrics)
        weighted_score += phase['weight'] * metrics['score']

        print(f"  {phase['name']}: Score={metrics['score']:.4f}")

    print(f"  Weighted: {weighted_score:.4f}")
    return weighted_score, phase_results


def print_results_table(results):
    """Print formatted results table."""
    print(f"\n{'Experiment':<30} {'P1':>8} {'P2':>8} {'P3':>8} {'Weighted':>10}")
    print("-" * 70)
    for name, (weighted, phases) in results.items():
        p1 = phases[0]['score'] if len(phases) > 0 else float('nan')
        p2 = phases[1]['score'] if len(phases) > 1 else float('nan')
        p3 = phases[2]['score'] if len(phases) > 2 else float('nan')
        print(f"{name:<30} {p1:>8.4f} {p2:>8.4f} {p3:>8.4f} {weighted:>10.4f}")


if __name__ == '__main__':
    score, results = evaluate_baseline()
    print(f"\nBaseline Weighted Score: {score:.4f}")
