"""
Load-forecast accuracy measured on the same data as the control score.

Computes MAPE and MAE on all three phases with the 0.2/0.3/0.5 phase weights of
the MPC score, plus two task-specific measures: error on peak hours (top decile
of actual district load within each phase) and error on the executed horizons
h = 1..8 (the part of each plan the controller actually applies).

Usage (from the repository root):
    python -m experiments.run_peak_metrics

Output: results/verify_peak_metrics.json
"""
import json

import numpy as np

from mpcgap.evaluate_full import PHASES
from mpcgap.data import load_building_data
from mpcgap.forecasters import (PersistenceForecaster, HoltWintersForecaster,
                         WeeklySeasonalityForecaster, EnsembleForecaster)


def make(kind, n, sim_start, buildings):
    if kind == 'persistence':
        return PersistenceForecaster(n)
    if kind == 'holtwinters':
        return HoltWintersForecaster(n)
    if kind == 'weekly':
        return WeeklySeasonalityForecaster(n)
    if kind == 'ensemble':
        return EnsembleForecaster([PersistenceForecaster(n),
                                   WeeklySeasonalityForecaster(n),
                                   HoltWintersForecaster(n)])
    from mpcgap.lgb_forecaster import LGBForecaster
    lgb = LGBForecaster(n, sim_start=sim_start, building_names=buildings)
    if kind == 'lgb':
        return lgb
    if kind == 'lgb_avg':
        return EnsembleForecaster([lgb, PersistenceForecaster(n)])
    raise ValueError(kind)


def eval_phase(kind, phase, load_data, solar_data, warmup=168):
    idx = [int(b.split('_')[1]) - 1 for b in phase['buildings']]
    n = len(idx)
    s0, s1 = phase['sim_start'], phase['sim_end']
    load = np.array([load_data[i][s0:s1 + 1] for i in idx])
    solar = np.array([solar_data[i][s0:s1 + 1] for i in idx])
    T = load.shape[1]

    # Peak threshold: top decile of district load within this phase
    district = load.sum(axis=0)
    peak_thr = np.quantile(district, 0.9)

    fc = make(kind, n, s0, phase['buildings'])
    fc.reset()

    acc = {k: [] for k in ('load_ae', 'solar_ae', 'load_ae_h8', 'load_ae_peak',
                           'load_ape', 'solar_ape')}
    for t in range(T):
        for b in range(n):
            fc.update(b, load[b][t], solar[b][t], observation=None)
        if t >= warmup and t % 24 == 0 and t + 24 < T:
            for b in range(n):
                lp, sp = fc.predict(b, t, t % 24)
                la = load[b][t + 1:t + 25]
                sa = solar[b][t + 1:t + 25]
                ae = np.abs(lp - la)
                acc['load_ae'].extend(ae.tolist())
                acc['solar_ae'].extend(np.abs(sp - sa).tolist())
                acc['load_ae_h8'].extend(ae[:8].tolist())
                # peak hours within this 24h block, by district load
                pk = district[t + 1:t + 25] >= peak_thr
                if pk.any():
                    acc['load_ae_peak'].extend(ae[pk].tolist())
                ml = la > 0.1
                if ml.any():
                    acc['load_ape'].append(float(np.mean(ae[ml] / la[ml])))
                ms = sa > 0.1
                if ms.any():
                    acc['solar_ape'].append(
                        float(np.mean(np.abs(sp[ms] - sa[ms]) / sa[ms])))
    return {
        'load_mae': float(np.mean(acc['load_ae'])),
        'solar_mae': float(np.mean(acc['solar_ae'])),
        'load_mae_h1_8': float(np.mean(acc['load_ae_h8'])),
        'load_mae_peak': float(np.mean(acc['load_ae_peak'])),
        'load_mape': float(np.mean(acc['load_ape']) * 100),
        'solar_mape': float(np.mean(acc['solar_ape']) * 100),
    }


def main():
    load_data, solar_data = load_building_data()
    kinds = ['lgb', 'lgb_avg', 'persistence', 'ensemble', 'holtwinters', 'weekly']
    out = {}
    for kind in kinds:
        per_phase = []
        for ph in PHASES:
            m = eval_phase(kind, ph, load_data, solar_data)
            per_phase.append(m)
            print(f"{kind:12s} {ph['name']}: "
                  f"MAE {m['load_mae']:.3f} | peak {m['load_mae_peak']:.3f} | "
                  f"h1-8 {m['load_mae_h1_8']:.3f} | MAPE {m['load_mape']:.1f}%",
                  flush=True)
        w = {k: sum(PHASES[i]['weight'] * per_phase[i][k] for i in range(3))
             for k in per_phase[0]}
        out[kind] = {'per_phase': per_phase, 'weighted': w}
        print(f"{kind:12s} WEIGHTED: " +
              " ".join(f"{k}={v:.3f}" for k, v in w.items()), flush=True)
    with open('results/verify_peak_metrics.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
