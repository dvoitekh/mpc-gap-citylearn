"""
Scenario-noise sensitivity of the stochastic MPC with the LightGBM forecaster.

Sweeps the flat multiplicative noise level sigma over
0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5 (three phases each).

Usage (from the repository root):
    python -m experiments.run_noise_sensitivity

Output: results/noise_sensitivity_results.txt
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


def run_noise_test(sigma):
    """Run MPC+LGB with given flat noise level on all 3 phases."""
    noise_profile = np.full(24, sigma)
    weighted_score = 0
    phase_results = []

    for phase in PHASES:
        N = len(phase['buildings'])
        forecaster = make_lgb(N, phase['sim_start'])

        metrics = run_mpc_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'],
            forecaster, n_scenarios=5,
            action_smooth_weight=0.1,
            noise_profile=noise_profile,
            verbose=True,
        )
        phase_results.append(metrics)
        weighted_score += phase['weight'] * metrics['score']

    return weighted_score, phase_results


def main():
    start_time = time.time()
    sigma_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    print("=" * 70)
    print("NOISE LEVEL SENSITIVITY ANALYSIS (MPC + LightGBM)")
    print("=" * 70)

    results = {}
    for sigma in sigma_values:
        print(f"\n--- sigma = {sigma} ---")
        t0 = time.time()
        score, phase_results = run_noise_test(sigma)
        elapsed = time.time() - t0

        p_scores = [pr['score'] for pr in phase_results]
        results[sigma] = (score, p_scores)
        print(f"  sigma={sigma}: Weighted={score:.4f}, "
              f"P1={p_scores[0]:.4f}, P2={p_scores[1]:.4f}, P3={p_scores[2]:.4f} "
              f"({elapsed:.0f}s)")

    print("\n" + "=" * 70)
    print("NOISE SENSITIVITY RESULTS")
    print("=" * 70)
    print(f"\n{'sigma':>8} {'P1':>8} {'P2':>8} {'P3':>8} {'Weighted':>10}")
    print("-" * 42)
    for sigma, (score, p_scores) in results.items():
        print(f"{sigma:>8.2f} {p_scores[0]:>8.4f} {p_scores[1]:>8.4f} {p_scores[2]:>8.4f} {score:>10.4f}")

    with open('results/noise_sensitivity_results.txt', 'w') as f:
        f.write("Noise Level Sensitivity (MPC + LightGBM, flat noise)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'sigma':>8} {'P1':>8} {'P2':>8} {'P3':>8} {'Weighted':>10}\n")
        f.write("-" * 42 + "\n")
        for sigma, (score, p_scores) in results.items():
            f.write(f"{sigma:>8.2f} {p_scores[0]:>8.4f} {p_scores[1]:>8.4f} {p_scores[2]:>8.4f} {score:>10.4f}\n")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    print("Results saved to results/noise_sensitivity_results.txt")


if __name__ == '__main__':
    main()
