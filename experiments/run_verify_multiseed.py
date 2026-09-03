"""
Multi-seed evaluation of one MPC configuration (seeds 2023-2026).

The seed only affects scenario sampling in the stochastic MPC; the data, the
trained forecaster and the evaluation protocol are fixed. Seed 2022 is taken
from the corresponding run_verify_mpc.py result. Use
experiments/summarize_multiseed.py to aggregate the runs into means,
confidence intervals and paired tests.

Usage (from the repository root):
    python -m experiments.run_verify_multiseed <perfect|persistence|lgb|lgb_avg|ensemble|lgb_noleak>

Output: results/verify_multiseed_<config>.json
"""

import sys
import time
import json

from mpcgap.evaluate_full import PHASES
from mpcgap.online_mpc import run_mpc_phase
from mpcgap.forecasters import PersistenceForecaster


def make(kind, n, sim_start, buildings):
    if kind == 'persistence':
        return PersistenceForecaster(n)
    if kind == 'ensemble':
        from mpcgap.forecasters import (HoltWintersForecaster,
                                 WeeklySeasonalityForecaster, EnsembleForecaster)
        return EnsembleForecaster([
            PersistenceForecaster(n),
            WeeklySeasonalityForecaster(n),
            HoltWintersForecaster(n),
        ])
    if kind == 'perfect':
        from mpcgap.data import create_perfect_forecaster_factory
        global _PF
        if _PF is None:
            _PF = create_perfect_forecaster_factory()
        return _PF(n, sim_start)
    if kind == 'lgb':
        from mpcgap.lgb_forecaster import LGBForecaster
        return LGBForecaster(n, sim_start=sim_start, building_names=buildings)
    if kind == 'lgb_avg':
        from mpcgap.lgb_forecaster import LGBForecaster
        from mpcgap.forecasters import EnsembleForecaster
        return EnsembleForecaster([
            LGBForecaster(n, sim_start=sim_start, building_names=buildings),
            PersistenceForecaster(n),
        ])
    if kind == 'lgb_noleak':
        from mpcgap.lgb_forecaster import LGBForecaster
        return LGBForecaster(n, sim_start=sim_start, building_names=buildings,
                             model_path='models/lgb_models_no_p3.pkl')
    raise ValueError(kind)


_PF = None


def main():
    kind = sys.argv[1]
    seeds = [2023, 2024, 2025, 2026] if kind != 'lgb_noleak' else [2022, 2023]
    smooth = 0.0 if kind == 'perfect' else 0.1
    out = {'config': kind, 'seeds': {}}
    for seed in seeds:
        t0 = time.time()
        weighted = 0.0
        per_phase = []
        for phase in PHASES:
            n = len(phase['buildings'])
            fc = make(kind, n, phase['sim_start'], phase['buildings'])
            metrics = run_mpc_phase(
                phase['buildings'], phase['sim_start'], phase['sim_end'],
                fc, n_scenarios=5, action_smooth_weight=smooth,
                reopt_interval=8, verbose=False, random_seed=seed,
            )
            per_phase.append({k: float(v) for k, v in metrics.items()})
            weighted += phase['weight'] * metrics['score']
        out['seeds'][seed] = {'weighted': weighted, 'per_phase': per_phase}
        print(f"{kind} seed {seed}: {weighted:.4f} ({(time.time()-t0)/60:.0f} min)",
              flush=True)
        with open(f'results/verify_multiseed_{kind}.json', 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
