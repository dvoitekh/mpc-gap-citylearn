"""
Task-loss analysis: forecast accuracy (MAPE) versus MPC score.

Quantifies how well MAPE predicts control performance across forecasters and
analyses which kinds of forecast error hurt the MPC most.
"""

import numpy as np
import pandas as pd
import time
from citylearn.data import DataSet

from mpcgap.evaluate_full import PHASES
from mpcgap.online_mpc import run_mpc_phase
from mpcgap.forecasters import (PerfectForecaster, PersistenceForecaster,
                          HoltWintersForecaster, WeeklySeasonalityForecaster,
                          compute_forecast_mape)


def load_building_data():
    """Load raw load/solar data for all buildings."""
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


def compute_mape_by_hour(forecaster, load_truth, solar_truth, n_buildings,
                         warmup=48):
    """
    Compute MAPE broken down by forecast horizon (h=1..24).

    Returns:
        load_mape_by_h: array (24,) — MAPE for each horizon hour
        solar_mape_by_h: array (24,) — MAPE for each horizon hour
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
                        load_errors[h].append(
                            abs(load_pred[h] - load_actual[h]) / load_actual[h]
                        )
                    if solar_actual[h] > 0.1:
                        solar_errors[h].append(
                            abs(solar_pred[h] - solar_actual[h]) / solar_actual[h]
                        )

    load_mape = np.array([
        np.mean(errs) * 100 if errs else 0.0 for errs in load_errors
    ])
    solar_mape = np.array([
        np.mean(errs) * 100 if errs else 0.0 for errs in solar_errors
    ])
    return load_mape, solar_mape


def compute_cost_weighted_mape(forecaster, load_truth, solar_truth,
                               n_buildings, price, carbon, sim_start,
                               warmup=48):
    """
    Compute MAPE weighted by electricity price and carbon intensity.

    Errors during expensive/dirty hours count more.

    Returns:
        weighted_load_mape: float
        weighted_solar_mape: float
        unweighted_load_mape: float
        unweighted_solar_mape: float
    """
    T = len(load_truth[0])
    load_errors_w = []
    load_errors_u = []
    solar_errors_w = []
    solar_errors_u = []

    # Normalize price and carbon to [0, 1] range
    price_norm = price / (np.max(price) + 1e-8)
    carbon_norm = carbon / (np.max(carbon) + 1e-8)

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
                    g = sim_start + t + 1 + h
                    if g >= len(price):
                        continue
                    # Cost weight: combination of price and carbon
                    w = 0.5 * price_norm[g] + 0.5 * carbon_norm[g]

                    if load_actual[h] > 0.1:
                        ape = abs(load_pred[h] - load_actual[h]) / load_actual[h]
                        load_errors_u.append(ape)
                        load_errors_w.append(ape * w)

                    if solar_actual[h] > 0.1:
                        ape = abs(solar_pred[h] - solar_actual[h]) / solar_actual[h]
                        solar_errors_u.append(ape)
                        solar_errors_w.append(ape * w)

    wl = np.mean(load_errors_w) * 100 if load_errors_w else float('inf')
    ul = np.mean(load_errors_u) * 100 if load_errors_u else float('inf')
    ws = np.mean(solar_errors_w) * 100 if solar_errors_w else float('inf')
    us = np.mean(solar_errors_u) * 100 if solar_errors_u else float('inf')
    return wl, ws, ul, us


def run_mpc_for_forecaster(name, make_forecaster, smooth_weight=0.1):
    """Run MPC on all 3 phases and return weighted score + per-phase metrics."""
    print(f"\n  Running MPC + {name}...")
    weighted = 0
    phase_results = []
    for phase in PHASES:
        N = len(phase['buildings'])
        forecaster = make_forecaster(N, phase['sim_start'])
        metrics = run_mpc_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'],
            forecaster, n_scenarios=5,
            action_smooth_weight=smooth_weight,
            reopt_interval=8, verbose=False
        )
        phase_results.append(metrics)
        weighted += phase['weight'] * metrics['score']
    print(f"    Score: {weighted:.4f}")
    return weighted, phase_results


def main():
    print("=" * 70)
    print("PHASE 1: TASK-LOSS ANALYSIS — MAPE vs MPC Score")
    print("=" * 70)

    start_time = time.time()

    # Load data
    load_data, solar_data = load_building_data()

    # Price/carbon for cost-weighted MAPE
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')
    root_dir = schema['root_directory']
    price = pd.read_csv(f'{root_dir}/pricing.csv')['electricity_pricing'].values
    carbon = pd.read_csv(f'{root_dir}/carbon_intensity.csv')['carbon_intensity'].values

    # Phase 2 data for MAPE evaluation
    n_buildings = 5
    start = 24 * 120
    end = 24 * 240
    load_p2 = {i: load_data[i][start:end] for i in range(n_buildings)}
    solar_p2 = {i: solar_data[i][start:end] for i in range(n_buildings)}

    # Define forecasters
    def make_perfect(n, s=0):
        return PerfectForecaster(
            {i: load_data[i] for i in range(n)},
            {i: solar_data[i] for i in range(n)},
            sim_start=s,
        )

    forecasters = {
        'Perfect': {
            'make': make_perfect,
            'eval': PerfectForecaster(
                {i: load_data[i][start:end + 24] for i in range(n_buildings)},
                {i: solar_data[i][start:end + 24] for i in range(n_buildings)},
            ),
            'smooth': 0.0,
        },
        'Persistence': {
            'make': lambda n, s=0: PersistenceForecaster(n),
            'eval': PersistenceForecaster(n_buildings),
            'smooth': 0.1,
        },
        'Holt-Winters': {
            'make': lambda n, s=0: HoltWintersForecaster(n),
            'eval': HoltWintersForecaster(n_buildings),
            'smooth': 0.1,
        },
        'Weekly': {
            'make': lambda n, s=0: WeeklySeasonalityForecaster(n),
            'eval': WeeklySeasonalityForecaster(n_buildings),
            'smooth': 0.1,
        },
    }

    # Try to add LGB
    import os
    if os.path.exists('models/lgb_models.pkl'):
        from mpcgap.lgb_forecaster import LGBForecaster
        building_names_p2 = [f'Building_{i}' for i in range(1, 6)]
        forecasters['LightGBM'] = {
            'make': lambda n, s=0: LGBForecaster(n, sim_start=s,
                building_names=[f'Building_{i}' for i in range(1, n + 1)]),
            'eval': LGBForecaster(n_buildings, sim_start=start,
                                   building_names=building_names_p2),
            'smooth': 0.1,
        }

    # =====================================================
    # 1. Compute MAPE (overall + by horizon + cost-weighted)
    # =====================================================
    print("\n--- Forecast Quality Analysis (Phase 2) ---\n")

    results = {}
    for name, fc_info in forecasters.items():
        fc = fc_info['eval']

        # Overall MAPE
        lm, sm = compute_forecast_mape(fc, load_p2, solar_p2, n_buildings)

        # By horizon
        fc_eval2 = fc_info['eval']
        if name == 'LightGBM':
            fc_eval2 = LGBForecaster(n_buildings, sim_start=start,
                                      building_names=building_names_p2)
        elif name == 'Perfect':
            fc_eval2 = PerfectForecaster(
                {i: load_data[i][start:end + 24] for i in range(n_buildings)},
                {i: solar_data[i][start:end + 24] for i in range(n_buildings)},
            )
        else:
            fc_eval2 = type(fc)(n_buildings) if name != 'Holt-Winters' else HoltWintersForecaster(n_buildings)

        load_by_h, solar_by_h = compute_mape_by_hour(
            fc_eval2, load_p2, solar_p2, n_buildings
        )

        # Cost-weighted MAPE
        if name == 'LightGBM':
            fc_cw = LGBForecaster(n_buildings, sim_start=start,
                                   building_names=building_names_p2)
        elif name == 'Perfect':
            fc_cw = PerfectForecaster(
                {i: load_data[i][start:end + 24] for i in range(n_buildings)},
                {i: solar_data[i][start:end + 24] for i in range(n_buildings)},
            )
        else:
            fc_cw = type(fc)(n_buildings) if name != 'Holt-Winters' else HoltWintersForecaster(n_buildings)

        wl, ws, ul, us = compute_cost_weighted_mape(
            fc_cw, load_p2, solar_p2, n_buildings, price, carbon, start
        )

        results[name] = {
            'load_mape': lm,
            'solar_mape': sm,
            'avg_mape': (lm + sm) / 2,
            'load_by_horizon': load_by_h,
            'solar_by_horizon': solar_by_h,
            'cost_weighted_load': wl,
            'cost_weighted_solar': ws,
        }

        print(f"  {name:15s}: Load={lm:5.1f}% Solar={sm:5.1f}% "
              f"Avg={((lm+sm)/2):5.1f}% "
              f"CostW_Load={wl:5.1f}% CostW_Solar={ws:5.1f}%")

    # =====================================================
    # 2. Run MPC for each forecaster (get task loss)
    # =====================================================
    print("\n--- MPC Task Loss (3-phase weighted) ---\n")

    mpc_scores = {}
    for name, fc_info in forecasters.items():
        score, phases = run_mpc_for_forecaster(
            name, fc_info['make'], fc_info['smooth']
        )
        mpc_scores[name] = {
            'weighted': score,
            'phases': phases,
        }
        results[name]['mpc_score'] = score
        results[name]['mpc_phases'] = phases

    # =====================================================
    # 3. Correlation Analysis
    # =====================================================
    print("\n--- MAPE vs MPC Score Correlation ---\n")

    names = [n for n in results if n != 'Perfect']  # Exclude perfect (trivially 0)
    mapes = [results[n]['avg_mape'] for n in names]
    scores = [results[n]['mpc_score'] for n in names]
    cw_mapes = [(results[n]['cost_weighted_load'] + results[n]['cost_weighted_solar']) / 2
                for n in names]

    from scipy import stats
    if len(names) >= 3:
        r_mape, p_mape = stats.pearsonr(mapes, scores)
        r_cw, p_cw = stats.pearsonr(cw_mapes, scores)
        rho_mape, _ = stats.spearmanr(mapes, scores)
        rho_cw, _ = stats.spearmanr(cw_mapes, scores)

        print(f"  Pearson  (MAPE vs Score):           r={r_mape:.3f}, p={p_mape:.3f}")
        print(f"  Pearson  (CostW-MAPE vs Score):     r={r_cw:.3f}, p={p_cw:.3f}")
        print(f"  Spearman (MAPE vs Score):            ρ={rho_mape:.3f}")
        print(f"  Spearman (CostW-MAPE vs Score):      ρ={rho_cw:.3f}")

    # =====================================================
    # 4. Summary Table
    # =====================================================
    print("\n--- Summary Table ---\n")
    print(f"{'Forecaster':15s} {'Load%':>7s} {'Solar%':>7s} {'Avg%':>7s} "
          f"{'CW_L%':>7s} {'CW_S%':>7s} {'MPC':>7s}")
    print("-" * 65)
    for name in results:
        r = results[name]
        print(f"{name:15s} {r['load_mape']:7.1f} {r['solar_mape']:7.1f} "
              f"{r['avg_mape']:7.1f} {r.get('cost_weighted_load', 0):7.1f} "
              f"{r.get('cost_weighted_solar', 0):7.1f} "
              f"{r.get('mpc_score', 0):7.4f}")

    # =====================================================
    # 5. MAPE by horizon analysis
    # =====================================================
    print("\n--- MAPE by Forecast Horizon (Load, Phase 2) ---\n")
    print(f"{'Hour':>4s}", end="")
    for name in results:
        print(f"  {name:>10s}", end="")
    print()
    for h in range(24):
        print(f"{h+1:4d}", end="")
        for name in results:
            val = results[name]['load_by_horizon'][h]
            print(f"  {val:10.1f}", end="")
        print()

    # =====================================================
    # 6. Per-phase MPC breakdown
    # =====================================================
    print("\n--- Per-Phase MPC Breakdown ---\n")
    for name in results:
        if 'mpc_phases' not in results[name]:
            continue
        phases = results[name]['mpc_phases']
        print(f"\n  {name}:")
        for i, p in enumerate(phases):
            print(f"    Phase {i+1}: score={p['score']:.4f} "
                  f"cost={p['cost']:.4f} emis={p['emissions']:.4f} "
                  f"grid={p['grid']:.4f} "
                  f"(ramp={p['ramping']:.3f} lf={p['load_factor']:.3f} "
                  f"dpeak={p['daily_peak']:.3f} atp={p['all_time_peak']:.3f})")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")

    return results


if __name__ == '__main__':
    main()
