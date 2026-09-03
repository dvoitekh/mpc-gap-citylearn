"""
Forecast accuracy on Phase 2 (buildings 1-5, days 120-240): MAPE, MAE and sMAPE
for load and solar, for every forecaster.

Usage (from the repository root):
    python -m experiments.run_forecast_metrics

Output: results/forecast_metrics_results.txt
"""

import numpy as np
from mpcgap.data import load_building_data
from mpcgap.forecasters import (
    PerfectForecaster, PersistenceForecaster, HoltWintersForecaster,
    WeeklySeasonalityForecaster, EnsembleForecaster,
)
from mpcgap.lgb_forecaster import LGBForecaster


def compute_forecast_metrics(forecaster, load_truth, solar_truth, n_buildings, warmup=48):
    """
    Compute MAPE, MAE, and sMAPE for a forecaster.

    Returns dict with load_mape, solar_mape, load_mae, solar_mae, load_smape, solar_smape.
    """
    T = len(load_truth[0])
    load_abs_pct_errors = []
    solar_abs_pct_errors = []
    load_abs_errors = []
    solar_abs_errors = []
    load_sym_errors = []
    solar_sym_errors = []

    forecaster.reset()

    for t in range(T):
        hour = t % 24
        for b in range(n_buildings):
            forecaster.update(b, load_truth[b][t], solar_truth[b][t], observation=None)

        if t >= warmup and t + 24 < T and hour == 0:
            for b in range(n_buildings):
                load_pred, solar_pred = forecaster.predict(b, t, hour)
                load_actual = load_truth[b][t + 1:t + 25]
                solar_actual = solar_truth[b][t + 1:t + 25]

                # MAPE (load > 0.1 kW)
                mask_load = load_actual > 0.1
                if mask_load.any():
                    ape = np.abs(load_pred[mask_load] - load_actual[mask_load]) / load_actual[mask_load]
                    load_abs_pct_errors.append(np.mean(ape))

                # MAPE (solar > 0.1 kW)
                mask_solar = solar_actual > 0.1
                if mask_solar.any():
                    ape = np.abs(solar_pred[mask_solar] - solar_actual[mask_solar]) / solar_actual[mask_solar]
                    solar_abs_pct_errors.append(np.mean(ape))

                # MAE (all values, in kW)
                load_abs_errors.extend(np.abs(load_pred - load_actual).tolist())
                solar_abs_errors.extend(np.abs(solar_pred - solar_actual).tolist())

                # sMAPE (values where actual + pred > 0.1 to avoid 0/0)
                denom_load = np.abs(load_actual) + np.abs(load_pred)
                mask_s_load = denom_load > 0.1
                if mask_s_load.any():
                    smape = 2 * np.abs(load_pred[mask_s_load] - load_actual[mask_s_load]) / denom_load[mask_s_load]
                    load_sym_errors.extend(smape.tolist())

                denom_solar = np.abs(solar_actual) + np.abs(solar_pred)
                mask_s_solar = denom_solar > 0.1
                if mask_s_solar.any():
                    smape = 2 * np.abs(solar_pred[mask_s_solar] - solar_actual[mask_s_solar]) / denom_solar[mask_s_solar]
                    solar_sym_errors.extend(smape.tolist())

    return {
        'load_mape': np.mean(load_abs_pct_errors) * 100 if load_abs_pct_errors else float('inf'),
        'solar_mape': np.mean(solar_abs_pct_errors) * 100 if solar_abs_pct_errors else float('inf'),
        'load_mae': np.mean(load_abs_errors) if load_abs_errors else float('inf'),
        'solar_mae': np.mean(solar_abs_errors) if solar_abs_errors else float('inf'),
        'load_smape': np.mean(load_sym_errors) * 100 if load_sym_errors else float('inf'),
        'solar_smape': np.mean(solar_sym_errors) * 100 if solar_sym_errors else float('inf'),
    }


def main():
    import os
    N = 5  # Phase 2 uses buildings 0-4 (Building_1..5)
    start = 24 * 120
    end = 24 * 240

    print("=" * 80)
    print(f"FORECAST METRICS — Phase 2 (buildings 1-5, days 120-240)")
    print("=" * 80)

    load_data, solar_data = load_building_data()
    load_p2 = {i: load_data[i][start:end] for i in range(N)}
    solar_p2 = {i: solar_data[i][start:end] for i in range(N)}

    building_names = [f'Building_{i}' for i in range(1, 6)]

    forecasters_dict = {
        'Perfect': PerfectForecaster(
            {i: load_data[i][start:end + 24] for i in range(N)},
            {i: solar_data[i][start:end + 24] for i in range(N)},
        ),
        'Persistence': PersistenceForecaster(N),
        'Holt-Winters': HoltWintersForecaster(N),
        'Weekly': WeeklySeasonalityForecaster(N),
        'Ensemble': EnsembleForecaster([
            PersistenceForecaster(N),
            WeeklySeasonalityForecaster(N),
            HoltWintersForecaster(N),
        ]),
    }
    if os.path.exists('models/lgb_models.pkl'):
        forecasters_dict['LightGBM'] = LGBForecaster(
            N, sim_start=start, building_names=building_names)

    all_results = {}
    for name, fc in forecasters_dict.items():
        print(f"\n--- {name} ---")
        metrics = compute_forecast_metrics(fc, load_p2, solar_p2, N)
        all_results[name] = metrics
        print(f"  Load:  MAPE={metrics['load_mape']:.1f}%  MAE={metrics['load_mae']:.3f} kW  sMAPE={metrics['load_smape']:.1f}%")
        print(f"  Solar: MAPE={metrics['solar_mape']:.1f}%  MAE={metrics['solar_mae']:.3f} kW  sMAPE={metrics['solar_smape']:.1f}%")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Forecaster':<15} {'Load':>6} {'Solar':>6} {'Load':>6} {'Solar':>6} {'Load':>6} {'Solar':>6}"
    subhdr = f"{'':15} {'MAPE%':>6} {'MAPE%':>6} {'MAE':>6} {'MAE':>6} {'sMAPE%':>6} {'sMAPE%':>6}"
    print(header)
    print(subhdr)
    print("-" * 65)
    for name, m in all_results.items():
        print(f"{name:<15} {m['load_mape']:>6.1f} {m['solar_mape']:>6.1f} "
              f"{m['load_mae']:>6.3f} {m['solar_mae']:>6.3f} "
              f"{m['load_smape']:>6.1f} {m['solar_smape']:>6.1f}")

    with open('results/forecast_metrics_results.txt', 'w') as f:
        f.write("Forecast Metrics — Phase 2\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Forecaster':<15} {'L-MAPE%':>8} {'S-MAPE%':>8} {'L-MAE':>8} {'S-MAE':>8} {'L-sMAPE%':>9} {'S-sMAPE%':>9}\n")
        f.write("-" * 70 + "\n")
        for name, m in all_results.items():
            f.write(f"{name:<15} {m['load_mape']:>8.1f} {m['solar_mape']:>8.1f} "
                    f"{m['load_mae']:>8.3f} {m['solar_mae']:>8.3f} "
                    f"{m['load_smape']:>9.1f} {m['solar_smape']:>9.1f}\n")

    print("\nResults saved to results/forecast_metrics_results.txt")


if __name__ == '__main__':
    main()
