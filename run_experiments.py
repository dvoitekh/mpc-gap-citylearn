"""
Run the full experiment suite: LP reference, online MPC with every forecaster,
no-control baseline, and forecast accuracy (MAPE) for every forecaster.

Experiments:
1. LP Perfect Foresight (perfect-foresight reference)
2. Online MPC + Perfect Forecaster (MPC upper bound)
3. Online MPC + Persistence Forecaster
4. Online MPC + Holt-Winters Forecaster
5. Baseline (no control)

Each experiment runs on all 3 phases with weighted scoring.
Also computes forecast MAPE for each forecaster.
"""

import time
import numpy as np
import pandas as pd
from citylearn.data import DataSet

from environment import create_env, evaluate_env
from perfect_foresight_lp import load_citylearn_data, solve_lp_phase, evaluate_lp_phase
from online_mpc import OnlineMPC, run_mpc_phase
from forecasters import (PerfectForecaster, PersistenceForecaster, HoltWintersForecaster,
                          WeeklySeasonalityForecaster, EnsembleForecaster,
                          HybridForecaster, compute_forecast_mape)
from evaluate_full import PHASES, evaluate_baseline
from lgb_forecaster import LGBForecaster


def load_building_data():
    """Load raw load/solar data for all buildings (for forecaster evaluation)."""
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')
    root_dir = schema['root_directory']

    all_buildings = [f'Building_{i}' for i in range(1, 18)]
    pv_caps = {}
    for b in all_buildings:
        if b in schema['buildings']:
            pv_caps[b] = schema['buildings'][b].get('pv', {}).get('nominal_power', 4.0)

    load_data = {}
    solar_data = {}
    for i, b in enumerate(all_buildings):
        df = pd.read_csv(f'{root_dir}/{b}.csv')
        load_data[i] = df['non_shiftable_load'].values
        solar_data[i] = df['solar_generation'].values * pv_caps.get(b, 4.0) / 1000.0

    return load_data, solar_data


def run_lp_experiment():
    """Run LP perfect foresight on all phases."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: LP PERFECT FORESIGHT")
    print("=" * 70)

    net_load, price, carbon, _ = load_citylearn_data()

    weighted_score = 0
    phase_results = []

    for phase in PHASES:
        print(f"\n  Solving {phase['name']}...")
        actions = solve_lp_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'],
            net_load, price, carbon
        )
        metrics = evaluate_lp_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'], actions
        )
        phase_results.append(metrics)
        weighted_score += phase['weight'] * metrics['score']
        print(f"  {phase['name']}: Score={metrics['score']:.4f}, "
              f"Cost={metrics['cost']:.4f}, Emis={metrics['emissions']:.4f}, "
              f"Grid={metrics['grid']:.4f}")

    print(f"\n  LP Weighted Score: {weighted_score:.4f}")
    return weighted_score, phase_results


def run_mpc_experiment(forecaster_name, make_forecaster, action_smooth_weight=0.1,
                       reopt_interval=8, peak_penalty_weight=0.0,
                       horizon_discount=1.0, noise_profile=None, random_seed=2022):
    """
    Run Online MPC with a specific forecaster on all phases.

    Args:
        forecaster_name: str, display name
        make_forecaster: callable(n_buildings, sim_start) -> BaseForecaster
        action_smooth_weight: smoothness penalty weight (0 = no smoothing)
        reopt_interval: re-optimize every N hours
        peak_penalty_weight: weight for peak-exceeding penalty (0 = disabled)
        horizon_discount: gamma for exponential time discount (1.0 = none)
        noise_profile: per-horizon noise std array (24,), None = flat 0.3
    """
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: ONLINE MPC + {forecaster_name}")
    print("=" * 70)

    weighted_score = 0
    phase_results = []

    for phase in PHASES:
        N = len(phase['buildings'])
        forecaster = make_forecaster(N, phase['sim_start'])

        print(f"\n  {phase['name']} ({N} buildings)...")
        metrics = run_mpc_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'],
            forecaster, n_scenarios=5,
            action_smooth_weight=action_smooth_weight,
            reopt_interval=reopt_interval,
            peak_penalty_weight=peak_penalty_weight,
            horizon_discount=horizon_discount,
            noise_profile=noise_profile, verbose=True,
            random_seed=random_seed
        )
        phase_results.append(metrics)
        weighted_score += phase['weight'] * metrics['score']
        print(f"  {phase['name']}: Score={metrics['score']:.4f}")

    print(f"\n  {forecaster_name} Weighted Score: {weighted_score:.4f}")
    return weighted_score, phase_results


def evaluate_forecast_quality():
    """Compute MAPE for each forecaster on Phase 2 data."""
    print("\n" + "=" * 70)
    print("FORECAST QUALITY EVALUATION (Phase 2 data)")
    print("=" * 70)

    load_data, solar_data = load_building_data()
    n_buildings = 5  # Phase 2 uses buildings 0-4
    start = 24 * 120
    end = 24 * 240

    # Slice Phase 2 data
    load_p2 = {i: load_data[i][start:end] for i in range(n_buildings)}
    solar_p2 = {i: solar_data[i][start:end] for i in range(n_buildings)}

    results = {}

    # Perfect forecaster
    pf = PerfectForecaster(
        {i: load_data[i][start:end + 24] for i in range(n_buildings)},
        {i: solar_data[i][start:end + 24] for i in range(n_buildings)},
    )
    lm, sm = compute_forecast_mape(pf, load_p2, solar_p2, n_buildings)
    results['Perfect'] = (lm, sm)
    print(f"  Perfect:     Load MAPE={lm:.1f}%, Solar MAPE={sm:.1f}%")

    # Persistence
    pers = PersistenceForecaster(n_buildings)
    lm, sm = compute_forecast_mape(pers, load_p2, solar_p2, n_buildings)
    results['Persistence'] = (lm, sm)
    print(f"  Persistence: Load MAPE={lm:.1f}%, Solar MAPE={sm:.1f}%")

    # Holt-Winters
    hw = HoltWintersForecaster(n_buildings)
    lm, sm = compute_forecast_mape(hw, load_p2, solar_p2, n_buildings)
    results['Holt-Winters'] = (lm, sm)
    print(f"  Holt-Winters: Load MAPE={lm:.1f}%, Solar MAPE={sm:.1f}%")

    # Weekly Seasonality
    weekly = WeeklySeasonalityForecaster(n_buildings)
    lm, sm = compute_forecast_mape(weekly, load_p2, solar_p2, n_buildings)
    results['Weekly'] = (lm, sm)
    print(f"  Weekly:      Load MAPE={lm:.1f}%, Solar MAPE={sm:.1f}%")

    # Ensemble (persistence + weekly + holt-winters, equal weights)
    ensemble = EnsembleForecaster([
        PersistenceForecaster(n_buildings),
        WeeklySeasonalityForecaster(n_buildings),
        HoltWintersForecaster(n_buildings),
    ])
    lm, sm = compute_forecast_mape(ensemble, load_p2, solar_p2, n_buildings)
    results['Ensemble'] = (lm, sm)
    print(f"  Ensemble:    Load MAPE={lm:.1f}%, Solar MAPE={sm:.1f}%")

    # LightGBM
    import os
    if os.path.exists('models/lgb_models.pkl'):
        building_names = [f'Building_{i}' for i in range(1, 6)]
        lgb_f = LGBForecaster(n_buildings, sim_start=start,
                               building_names=building_names)
        lm, sm = compute_forecast_mape(lgb_f, load_p2, solar_p2, n_buildings)
        results['LightGBM'] = (lm, sm)
        print(f"  LightGBM:    Load MAPE={lm:.1f}%, Solar MAPE={sm:.1f}%")
    else:
        print("  LightGBM:    (skipped — no trained model found)")

    return results


def create_perfect_forecaster_factory():
    """Create factory that returns PerfectForecaster with pre-loaded data."""
    load_data, solar_data = load_building_data()

    def make_perfect(n_buildings, sim_start=0):
        return PerfectForecaster(
            {i: load_data[i] for i in range(n_buildings)},
            {i: solar_data[i] for i in range(n_buildings)},
            sim_start=sim_start,
        )
    return make_perfect


def make_lgb_forecaster(n_buildings, sim_start=0, phase_buildings=None):
    """Create LGBForecaster with correct building names."""
    return LGBForecaster(
        n_buildings, sim_start=sim_start,
        building_names=phase_buildings,
    )


def main():
    start_time = time.time()

    print("=" * 70)
    print("GAP ANALYSIS: LP Perfect Foresight vs Online MPC")
    print("CityLearn 2022 Challenge - 3-Phase Weighted Evaluation")
    print("=" * 70)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Skip LP and non-essential forecasters')
    args = parser.parse_args()

    all_results = {}

    # 1. Baseline
    score, results = evaluate_baseline()
    all_results['Baseline (no control)'] = (score, results)

    if not args.fast:
        # 2. LP Perfect Foresight
        score, results = run_lp_experiment()
        all_results['LP Perfect Foresight'] = (score, results)

    # 3. MPC + Perfect Forecaster (no smoothing — hurts with perfect forecast)
    make_perfect = create_perfect_forecaster_factory()
    score, results = run_mpc_experiment('Perfect Forecaster', make_perfect,
                                         action_smooth_weight=0.0)
    all_results['MPC + Perfect'] = (score, results)

    # 4. MPC + Persistence (with action smoothing)
    score, results = run_mpc_experiment(
        'Persistence Forecaster',
        lambda n, s=0: PersistenceForecaster(n)
    )
    all_results['MPC + Persistence'] = (score, results)

    if not args.fast:
        # 5. MPC + Holt-Winters
        score, results = run_mpc_experiment(
            'Holt-Winters Forecaster',
            lambda n, s=0: HoltWintersForecaster(n)
        )
        all_results['MPC + Holt-Winters'] = (score, results)

        # 6. MPC + Weekly Seasonality
        score, results = run_mpc_experiment(
            'Weekly Forecaster',
            lambda n, s=0: WeeklySeasonalityForecaster(n)
        )
        all_results['MPC + Weekly'] = (score, results)

        # 7. MPC + Ensemble (persistence + weekly + holt-winters)
        score, results = run_mpc_experiment(
            'Ensemble Forecaster',
            lambda n, s=0: EnsembleForecaster([
                PersistenceForecaster(n),
                WeeklySeasonalityForecaster(n),
                HoltWintersForecaster(n),
            ])
        )
        all_results['MPC + Ensemble'] = (score, results)

    # 8. MPC + LightGBM (if trained model exists)
    import os
    if os.path.exists('models/lgb_models.pkl'):
        def make_lgb(n_buildings, sim_start=0):
            for phase in PHASES:
                if len(phase['buildings']) == n_buildings and phase['sim_start'] == sim_start:
                    return make_lgb_forecaster(n_buildings, sim_start, phase['buildings'])
            return make_lgb_forecaster(n_buildings, sim_start)

        score, results = run_mpc_experiment('LightGBM Forecaster', make_lgb)
        all_results['MPC + LightGBM'] = (score, results)

        # 9. MPC + Hybrid (LGB load + Persistence solar + night constraint)
        def make_hybrid(n_buildings, sim_start=0):
            phase_buildings = None
            for phase in PHASES:
                if len(phase['buildings']) == n_buildings and phase['sim_start'] == sim_start:
                    phase_buildings = phase['buildings']
                    break
            lgb_fc = make_lgb_forecaster(n_buildings, sim_start, phase_buildings)
            persist_fc = PersistenceForecaster(n_buildings)
            return HybridForecaster(lgb_fc, persist_fc)

        score, results = run_mpc_experiment('Hybrid (LGB+Persist)', make_hybrid)
        all_results['MPC + Hybrid'] = (score, results)

        # 10. MPC + LGB with hourly re-optimization (reopt_interval=4)
        score, results = run_mpc_experiment(
            'LGB + reopt=4h', make_lgb, reopt_interval=4)
        all_results['MPC + LGB (reopt=4h)'] = (score, results)

        # 11-13. MPC + LGB with peak-aware penalty (sweep weights)
        for pw in [0.25, 0.5, 1.0]:
            name = f'LGB + peak={pw}'
            score, results = run_mpc_experiment(
                name, make_lgb, peak_penalty_weight=pw)
            all_results[f'MPC + LGB (peak={pw})'] = (score, results)
    else:
        print("\n  LightGBM experiment skipped — no trained model found.")
        print("  Run 'python lgb_train.py' first.")

    # 9. Forecast quality
    forecast_results = evaluate_forecast_quality()

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Experiment':<30} {'P1':>8} {'P2':>8} {'P3':>8} {'Weighted':>10}")
    print("-" * 70)
    for name, (weighted, phases) in all_results.items():
        p1 = phases[0]['score']
        p2 = phases[1]['score']
        p3 = phases[2]['score']
        print(f"{name:<30} {p1:>8.4f} {p2:>8.4f} {p3:>8.4f} {weighted:>10.4f}")

    # Per-metric breakdown (weighted across phases)
    print(f"\nPer-Metric Breakdown (weighted):")
    metric_names = ['cost', 'emissions', 'ramping', 'load_factor', 'daily_peak', 'all_time_peak']
    header = f"{'Experiment':<30}" + "".join(f" {m:>12}" for m in metric_names)
    print(header)
    print("-" * (30 + 13 * len(metric_names)))
    for name, (weighted, phases) in all_results.items():
        vals = []
        for m in metric_names:
            wval = sum(p['weight'] * phases[i].get(m, 0) for i, p in enumerate(PHASES))
            vals.append(wval)
        row = f"{name:<30}" + "".join(f" {v:>12.4f}" for v in vals)
        print(row)

    print(f"\nForecast Quality (Phase 2):")
    print(f"{'Forecaster':<20} {'Load MAPE':>12} {'Solar MAPE':>12}")
    print("-" * 48)
    for name, (lm, sm) in forecast_results.items():
        print(f"{name:<20} {lm:>10.1f}% {sm:>10.1f}%")

    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    # Save results
    with open('results/experiment_results.txt', 'w') as f:
        f.write("GAP ANALYSIS RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Experiment':<30} {'P1':>8} {'P2':>8} {'P3':>8} {'Weighted':>10}\n")
        f.write("-" * 70 + "\n")
        for name, (weighted, phases) in all_results.items():
            p1 = phases[0]['score']
            p2 = phases[1]['score']
            p3 = phases[2]['score']
            f.write(f"{name:<30} {p1:>8.4f} {p2:>8.4f} {p3:>8.4f} {weighted:>10.4f}\n")

        f.write(f"\nPer-Metric Breakdown (weighted):\n")
        metric_names = ['cost', 'emissions', 'ramping', 'load_factor', 'daily_peak', 'all_time_peak']
        f.write(f"{'Experiment':<30}" + "".join(f" {m:>12}" for m in metric_names) + "\n")
        f.write("-" * (30 + 13 * len(metric_names)) + "\n")
        for name, (weighted, phases) in all_results.items():
            vals = []
            for m in metric_names:
                wval = sum(p['weight'] * phases[i].get(m, 0) for i, p in enumerate(PHASES))
                vals.append(wval)
            f.write(f"{name:<30}" + "".join(f" {v:>12.4f}" for v in vals) + "\n")

        f.write(f"\nForecast Quality (Phase 2):\n")
        for name, (lm, sm) in forecast_results.items():
            f.write(f"  {name}: Load MAPE={lm:.1f}%, Solar MAPE={sm:.1f}%\n")

    print("\nResults saved to results/experiment_results.txt")


if __name__ == '__main__':
    main()
