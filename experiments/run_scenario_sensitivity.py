"""
Scenario-count sensitivity of the stochastic MPC with the LightGBM forecaster
(S = 1, 5, 10, 15; three phases each).

Usage (from the repository root):
    python -m experiments.run_scenario_sensitivity

Output: results/scenario_sensitivity_results.txt
"""

import time
import numpy as np

from evaluate_full import PHASES
from online_mpc import run_mpc_phase
from run_experiments import make_lgb_forecaster


def make_lgb(n_buildings, sim_start=0):
    for phase in PHASES:
        if len(phase['buildings']) == n_buildings and phase['sim_start'] == sim_start:
            return make_lgb_forecaster(n_buildings, sim_start, phase['buildings'])
    return make_lgb_forecaster(n_buildings, sim_start)


def run_scenario_test(n_scenarios):
    """Run MPC+LGB with given scenario count on all 3 phases."""
    weighted_score = 0
    phase_results = []

    for phase in PHASES:
        N = len(phase['buildings'])
        forecaster = make_lgb(N, phase['sim_start'])

        metrics = run_mpc_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'],
            forecaster, n_scenarios=n_scenarios,
            action_smooth_weight=0.1,
            verbose=True,
        )
        phase_results.append(metrics)
        weighted_score += phase['weight'] * metrics['score']

    return weighted_score, phase_results


def main():
    start_time = time.time()
    scenario_counts = [1, 5, 10, 15]

    print("=" * 70)
    print("SCENARIO COUNT SENSITIVITY ANALYSIS (MPC + LightGBM)")
    print("=" * 70)

    results = {}
    for S in scenario_counts:
        print(f"\n--- S = {S} scenarios ---")
        t0 = time.time()
        score, phase_results = run_scenario_test(S)
        elapsed = time.time() - t0

        p_scores = [pr['score'] for pr in phase_results]
        results[S] = (score, p_scores)
        print(f"  S={S}: Weighted={score:.4f}, "
              f"P1={p_scores[0]:.4f}, P2={p_scores[1]:.4f}, P3={p_scores[2]:.4f} "
              f"({elapsed:.0f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("SCENARIO SENSITIVITY RESULTS")
    print("=" * 70)
    print(f"\n{'S':>4} {'P1':>8} {'P2':>8} {'P3':>8} {'Weighted':>10}")
    print("-" * 42)
    for S, (score, p_scores) in results.items():
        print(f"{S:>4} {p_scores[0]:>8.4f} {p_scores[1]:>8.4f} {p_scores[2]:>8.4f} {score:>10.4f}")

    # Save
    with open('results/scenario_sensitivity_results.txt', 'w') as f:
        f.write("Scenario Count Sensitivity (MPC + LightGBM)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'S':>4} {'P1':>8} {'P2':>8} {'P3':>8} {'Weighted':>10}\n")
        f.write("-" * 42 + "\n")
        for S, (score, p_scores) in results.items():
            f.write(f"{S:>4} {p_scores[0]:>8.4f} {p_scores[1]:>8.4f} {p_scores[2]:>8.4f} {score:>10.4f}\n")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    print("Results saved to results/scenario_sensitivity_results.txt")


if __name__ == '__main__':
    main()
