"""
Three-phase evaluation of one MPC configuration with a fixed seed (2022).

Runs the online stochastic MPC on Phases 1-3 with the requested forecaster and
writes the per-phase metrics together with the weighted per-metric breakdown,
so that the main results table and the per-metric table come from one run.

Usage (from the repository root):
    python -m experiments.run_verify_mpc <config>

Forecaster configs:
    perfect        exact future values (action smoothing off, S=5, sigma=0.3)
    persistence    yesterday's profile
    holtwinters    triple exponential smoothing, 24 h seasonality
    weekly         same hour one week earlier
    ensemble       equal-weight average of persistence, weekly and Holt-Winters
    lgb            LightGBM (models/lgb_models.pkl) with online OLS load correction
    lgb_avg        equal-weight average of LightGBM and persistence
    lgb_days240    LightGBM trained on days 0-238 only (temporal-overlap check)
    lgb_days240_tt same, retrained from scratch (reproducibility check)
    lgb_noweather  LightGBM without weather features

Optional suffixes (combinable, e.g. lgb_term10, perfect_s1, persistence_nosmooth):
    _smooth / _nosmooth   force action smoothing on (0.1) / off (0)
    _s1                   deterministic MPC, a single unperturbed scenario
    _term05 / _term10     terminal state-of-charge credit with weight 0.5 / 1.0

Output: results/verify_mpc_<config>.json
"""

import sys
import time
import json

from evaluate_full import PHASES
from online_mpc import run_mpc_phase
from forecasters import (
    PersistenceForecaster, HoltWintersForecaster,
    WeeklySeasonalityForecaster, EnsembleForecaster,
)


def make_forecaster(kind, n_buildings, sim_start, phase_buildings):
    if kind == 'persistence':
        return PersistenceForecaster(n_buildings)
    if kind == 'holtwinters':
        return HoltWintersForecaster(n_buildings)
    if kind == 'weekly':
        return WeeklySeasonalityForecaster(n_buildings)
    if kind == 'ensemble':
        return EnsembleForecaster([
            PersistenceForecaster(n_buildings),
            WeeklySeasonalityForecaster(n_buildings),
            HoltWintersForecaster(n_buildings),
        ])
    if kind == 'lgb':
        from lgb_forecaster import LGBForecaster
        return LGBForecaster(n_buildings, sim_start=sim_start,
                             building_names=phase_buildings)
    if kind == 'lgb_noweather':
        from lgb_forecaster import LGBForecaster
        return LGBForecaster(n_buildings, sim_start=sim_start,
                             building_names=phase_buildings,
                             model_path='models/lgb_models_noweather.pkl')
    if kind == 'lgb_days240_tt':
        from lgb_forecaster import LGBForecaster
        return LGBForecaster(n_buildings, sim_start=sim_start,
                             building_names=phase_buildings,
                             model_path='models/lgb_models_days240_tt.pkl')
    if kind == 'lgb_days240':
        from lgb_forecaster import LGBForecaster
        return LGBForecaster(n_buildings, sim_start=sim_start,
                             building_names=phase_buildings,
                             model_path='models/lgb_models_days240.pkl')
    if kind == 'lgb_avg':
        from lgb_forecaster import LGBForecaster
        return EnsembleForecaster([
            LGBForecaster(n_buildings, sim_start=sim_start,
                          building_names=phase_buildings),
            PersistenceForecaster(n_buildings),
        ])
    if kind == 'perfect':
        from run_experiments import create_perfect_forecaster_factory
        global _PERFECT_FACTORY
        if _PERFECT_FACTORY is None:
            _PERFECT_FACTORY = create_perfect_forecaster_factory()
        return _PERFECT_FACTORY(n_buildings, sim_start)
    raise ValueError(kind)


_PERFECT_FACTORY = None


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    kind = sys.argv[1]
    base_kind = kind.replace('_smooth', '').replace('_nosmooth', '')
    n_scen = 5
    if base_kind.endswith('_s1'):
        base_kind = base_kind[:-3]
        n_scen = 1  # deterministic MPC: no scenario noise
    terminal_w = 0.0
    if '_term' in base_kind:
        base_kind, suffix = base_kind.split('_term')
        terminal_w = int(suffix) / 10.0  # e.g. lgb_term10 -> 1.0
    t0 = time.time()
    weighted_score = 0.0
    phase_results = []

    for phase in PHASES:
        n = len(phase['buildings'])
        fc = make_forecaster(base_kind, n, phase['sim_start'], phase['buildings'])
        smooth = 0.0 if base_kind == 'perfect' else 0.1
        if kind.endswith('_smooth'):
            smooth = 0.1
        if kind.endswith('_nosmooth'):
            smooth = 0.0
        print(f"{phase['name']} ({n} buildings)...", flush=True)
        metrics = run_mpc_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'],
            fc, n_scenarios=n_scen, action_smooth_weight=smooth,
            reopt_interval=8, verbose=True, random_seed=2022,
            terminal_soc_weight=terminal_w,
        )
        phase_results.append(metrics)
        weighted_score += phase['weight'] * metrics['score']
        print(f"{phase['name']}: {json.dumps({k: round(float(v), 4) for k, v in metrics.items()})}",
              flush=True)

    metric_names = ['cost', 'emissions', 'ramping', 'load_factor',
                    'daily_peak', 'all_time_peak', 'grid', 'score',
                    'monthly_load_factor', 'official_grid', 'official_score']
    weighted_metrics = {
        m: sum(p['weight'] * float(phase_results[i][m]) for i, p in enumerate(PHASES))
        for m in metric_names
    }

    out = {
        'forecaster': kind,
        'seed': 2022,
        'weighted_score': weighted_score,
        'per_phase': [{k: float(v) for k, v in pr.items()} for pr in phase_results],
        'weighted_metrics': weighted_metrics,
        'elapsed_min': (time.time() - t0) / 60,
    }
    with open(f'results/verify_mpc_{kind}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
