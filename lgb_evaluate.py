"""
Detailed evaluation of LightGBM forecaster.

Analyses:
1. Overall MAPE comparison (all forecasters)
2. MAPE by forecast horizon (h=1..24)
3. MAPE by hour of day
4. End-to-end MPC evaluation (3 phases)
5. Ablation experiments

Usage:
    python lgb_evaluate.py              # Run all evaluations
    python lgb_evaluate.py --forecast   # Forecast quality only
    python lgb_evaluate.py --mpc        # MPC evaluation only
    python lgb_evaluate.py --ablation   # Ablation study only
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

from citylearn.data import DataSet
from forecasters import (
    PerfectForecaster, PersistenceForecaster, HoltWintersForecaster
)
from lgb_forecaster import LGBForecaster
from run_experiments import load_building_data
from evaluate_full import PHASES

warnings.filterwarnings('ignore')


def evaluate_forecast_by_horizon(forecaster, load_truth, solar_truth,
                                  n_buildings, warmup=48, name=''):
    """
    Compute MAPE broken down by forecast horizon (h=1..24).

    Returns:
        load_mape_by_h: array (24,) — MAPE for each horizon step
        solar_mape_by_h: array (24,) — MAPE for each horizon step
    """
    T = len(load_truth[0])
    load_errors = [[] for _ in range(24)]
    solar_errors = [[] for _ in range(24)]

    forecaster.reset()

    for t in range(T):
        hour = t % 24

        for b in range(n_buildings):
            forecaster.update(b, load_truth[b][t], solar_truth[b][t])

        if t >= warmup and t + 24 < T and hour == 0:
            for b in range(n_buildings):
                load_pred, solar_pred = forecaster.predict(b, t, hour)
                load_actual = load_truth[b][t + 1:t + 25]
                solar_actual = solar_truth[b][t + 1:t + 25]

                for h in range(24):
                    if load_actual[h] > 0.1:
                        err = abs(load_pred[h] - load_actual[h]) / load_actual[h]
                        load_errors[h].append(err)
                    if solar_actual[h] > 0.1:
                        err = abs(solar_pred[h] - solar_actual[h]) / solar_actual[h]
                        solar_errors[h].append(err)

    load_mape_by_h = np.array([
        np.mean(errs) * 100 if errs else 0.0 for errs in load_errors
    ])
    solar_mape_by_h = np.array([
        np.mean(errs) * 100 if errs else 0.0 for errs in solar_errors
    ])

    return load_mape_by_h, solar_mape_by_h


def evaluate_forecast_by_hour_of_day(forecaster, load_truth, solar_truth,
                                      n_buildings, warmup=48):
    """
    Compute MAPE broken down by hour of day (0-23) at which prediction is made.

    This averages over all horizons for each prediction start hour.
    """
    T = len(load_truth[0])
    load_errors_by_hour = [[] for _ in range(24)]
    solar_errors_by_hour = [[] for _ in range(24)]

    forecaster.reset()

    for t in range(T):
        hour = t % 24

        for b in range(n_buildings):
            forecaster.update(b, load_truth[b][t], solar_truth[b][t])

        if t >= warmup and t + 24 < T and hour == 0:
            for b in range(n_buildings):
                load_pred, solar_pred = forecaster.predict(b, t, hour)
                load_actual = load_truth[b][t + 1:t + 25]
                solar_actual = solar_truth[b][t + 1:t + 25]

                for h in range(24):
                    target_hour = (hour + h + 1) % 24
                    if load_actual[h] > 0.1:
                        err = abs(load_pred[h] - load_actual[h]) / load_actual[h]
                        load_errors_by_hour[target_hour].append(err)
                    if solar_actual[h] > 0.1:
                        err = abs(solar_pred[h] - solar_actual[h]) / solar_actual[h]
                        solar_errors_by_hour[target_hour].append(err)

    load_mape = np.array([
        np.mean(errs) * 100 if errs else 0.0 for errs in load_errors_by_hour
    ])
    solar_mape = np.array([
        np.mean(errs) * 100 if errs else 0.0 for errs in solar_errors_by_hour
    ])

    return load_mape, solar_mape


def run_forecast_evaluation():
    """Run comprehensive forecast quality evaluation."""
    print("=" * 70)
    print("FORECAST QUALITY EVALUATION")
    print("=" * 70)

    load_data, solar_data = load_building_data()
    n_buildings = 5
    start = 24 * 120
    end = 24 * 240

    load_p2 = {i: load_data[i][start:end] for i in range(n_buildings)}
    solar_p2 = {i: solar_data[i][start:end] for i in range(n_buildings)}

    building_names = [f'Building_{i}' for i in range(1, 6)]

    # Create forecasters
    forecasters = {}

    pf = PerfectForecaster(
        {i: load_data[i][start:end + 24] for i in range(n_buildings)},
        {i: solar_data[i][start:end + 24] for i in range(n_buildings)},
    )
    forecasters['Perfect'] = pf
    forecasters['Persistence'] = PersistenceForecaster(n_buildings)
    forecasters['Holt-Winters'] = HoltWintersForecaster(n_buildings)

    if os.path.exists('models/lgb_models.pkl'):
        forecasters['LightGBM'] = LGBForecaster(
            n_buildings, sim_start=start, building_names=building_names
        )

    # 1. Overall MAPE
    print("\n--- Overall MAPE (Phase 2) ---")
    print(f"{'Forecaster':<20} {'Load MAPE':>12} {'Solar MAPE':>12}")
    print("-" * 48)

    overall_results = {}
    for name, fc in forecasters.items():
        from forecasters import compute_forecast_mape
        lm, sm = compute_forecast_mape(fc, load_p2, solar_p2, n_buildings)
        overall_results[name] = (lm, sm)
        print(f"{name:<20} {lm:>10.1f}% {sm:>10.1f}%")

    # 2. MAPE by horizon
    print("\n--- MAPE by Forecast Horizon ---")
    horizon_results = {}
    for name, fc in forecasters.items():
        if name == 'Perfect':
            continue
        lm_h, sm_h = evaluate_forecast_by_horizon(
            fc, load_p2, solar_p2, n_buildings, name=name
        )
        horizon_results[name] = (lm_h, sm_h)

    print(f"\n{'Horizon':<8}", end='')
    for name in horizon_results:
        print(f" {name+' Load':>16} {name+' Solar':>16}", end='')
    print()
    print("-" * (8 + 32 * len(horizon_results)))

    for h in range(24):
        print(f"h={h+1:<5}", end='')
        for name, (lm_h, sm_h) in horizon_results.items():
            print(f" {lm_h[h]:>14.1f}% {sm_h[h]:>14.1f}%", end='')
        print()

    # 3. MAPE by hour of day
    print("\n--- MAPE by Hour of Day ---")
    hour_results = {}
    for name, fc in forecasters.items():
        if name == 'Perfect':
            continue
        lm_hod, sm_hod = evaluate_forecast_by_hour_of_day(
            fc, load_p2, solar_p2, n_buildings
        )
        hour_results[name] = (lm_hod, sm_hod)

    print(f"\n{'Hour':<6}", end='')
    for name in hour_results:
        print(f" {name+' Load':>16} {name+' Solar':>16}", end='')
    print()
    print("-" * (6 + 32 * len(hour_results)))

    for h in range(24):
        print(f"{h:<6}", end='')
        for name, (lm_hod, sm_hod) in hour_results.items():
            print(f" {lm_hod[h]:>14.1f}% {sm_hod[h]:>14.1f}%", end='')
        print()

    return overall_results, horizon_results, hour_results


def run_mpc_evaluation():
    """Run MPC evaluation with LightGBM forecaster."""
    print("\n" + "=" * 70)
    print("MPC + LightGBM EVALUATION (3 phases)")
    print("=" * 70)

    if not os.path.exists('models/lgb_models.pkl'):
        print("  No trained model found. Run 'python lgb_train.py' first.")
        return None

    from online_mpc import run_mpc_phase

    weighted_score = 0
    phase_results = []

    for phase in PHASES:
        N = len(phase['buildings'])
        forecaster = LGBForecaster(
            N, sim_start=phase['sim_start'],
            building_names=phase['buildings'],
        )

        print(f"\n  {phase['name']} ({N} buildings)...")
        metrics = run_mpc_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'],
            forecaster, n_scenarios=5, verbose=True
        )
        phase_results.append(metrics)
        weighted_score += phase['weight'] * metrics['score']
        print(f"  {phase['name']}: Score={metrics['score']:.4f}, "
              f"Cost={metrics['cost']:.4f}, Emis={metrics['emissions']:.4f}, "
              f"Grid={metrics['grid']:.4f}")

    print(f"\n  MPC + LightGBM Weighted Score: {weighted_score:.4f}")
    return weighted_score, phase_results


def run_ablation_study():
    """Run ablation experiments with LightGBM variants."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    if not os.path.exists('models/lgb_models.pkl'):
        print("  No trained model found. Run 'python lgb_train.py' first.")
        return None

    from online_mpc import run_mpc_phase

    # Only run on Phase 2 for speed
    phase = PHASES[1]  # Phase 2
    N = len(phase['buildings'])

    results = {}

    # Full model
    print("\n  [1/3] Full LightGBM model...")
    fc = LGBForecaster(N, sim_start=phase['sim_start'],
                        building_names=phase['buildings'],
                        online_correction=True)
    metrics = run_mpc_phase(
        phase['buildings'], phase['sim_start'], phase['sim_end'],
        fc, n_scenarios=5, verbose=False
    )
    results['Full model'] = metrics['score']
    print(f"    Score: {metrics['score']:.4f}")

    # No online correction
    print("  [2/3] No online correction...")
    fc = LGBForecaster(N, sim_start=phase['sim_start'],
                        building_names=phase['buildings'],
                        online_correction=False)
    metrics = run_mpc_phase(
        phase['buildings'], phase['sim_start'], phase['sim_end'],
        fc, n_scenarios=5, verbose=False
    )
    results['No correction'] = metrics['score']
    print(f"    Score: {metrics['score']:.4f}")

    # Persistence baseline
    print("  [3/3] Persistence baseline...")
    fc = PersistenceForecaster(N)
    metrics = run_mpc_phase(
        phase['buildings'], phase['sim_start'], phase['sim_end'],
        fc, n_scenarios=5, verbose=False
    )
    results['Persistence'] = metrics['score']
    print(f"    Score: {metrics['score']:.4f}")

    print("\n--- Ablation Results (Phase 2) ---")
    print(f"{'Variant':<25} {'Score':>8}")
    print("-" * 35)
    for name, score in results.items():
        print(f"{name:<25} {score:>8.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast', action='store_true', help='Forecast quality only')
    parser.add_argument('--mpc', action='store_true', help='MPC evaluation only')
    parser.add_argument('--ablation', action='store_true', help='Ablation study only')
    args = parser.parse_args()

    run_all = not (args.forecast or args.mpc or args.ablation)

    start = time.time()

    if run_all or args.forecast:
        run_forecast_evaluation()

    if run_all or args.mpc:
        run_mpc_evaluation()

    if run_all or args.ablation:
        run_ablation_study()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
