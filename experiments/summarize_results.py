"""
Aggregate the single-seed (2022) evaluation runs in results/ into the tables
reported in the paper: per-phase and weighted scores for every forecaster,
the per-metric breakdown, the gap decomposition, and the controller/forecaster
ablations that share the same run format.

Usage (from the repository root):
    python -m experiments.summarize_results
"""
import glob
import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

MAIN = [
    ('MPC + Perfect', 'perfect'),
    ('MPC + LGB+Persistence avg', 'lgb_avg'),
    ('MPC + Ensemble (P+W+HW)', 'ensemble'),
    ('MPC + LightGBM', 'lgb'),
    ('MPC + Persistence', 'persistence'),
    ('MPC + Holt-Winters', 'holtwinters'),
    ('MPC + Weekly', 'weekly'),
]
ABLATIONS = [
    ('Perfect, S=1 (deterministic)', 'perfect_s1', 'perfect'),
    ('Perfect, smoothing 0.1', 'perfect_smooth', 'perfect'),
    ('Perfect, terminal SOC w=1.0', 'perfect_term10', 'perfect'),
    ('LightGBM, no smoothing', 'lgb_nosmooth', 'lgb'),
    ('LightGBM, terminal SOC w=0.5', 'lgb_term05', 'lgb'),
    ('LightGBM, terminal SOC w=1.0', 'lgb_term10', 'lgb'),
    ('LightGBM, trained on days 0-238', 'lgb_days240', 'lgb'),
    ('LightGBM, days 0-238 retrained from scratch', 'lgb_days240_tt', 'lgb'),
    ('LightGBM, no weather features', 'lgb_noweather', 'lgb'),
    ('Persistence, no smoothing', 'persistence_nosmooth', 'persistence'),
]
METRICS = ['cost', 'emissions', 'ramping', 'load_factor', 'daily_peak', 'all_time_peak']


def load(kind):
    path = os.path.join(RESULTS, f'verify_mpc_{kind}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    with open(os.path.join(RESULTS, 'verify_lp.json')) as f:
        lp = json.load(f)
    lp_phases = [p['score'] for p in lp['per_phase']]

    print('Weighted score by phase (seed 2022; lower is better, 1.0 = no control)')
    print(f"{'Method':<30}{'P1':>8}{'P2':>8}{'P3':>8}{'Weighted':>10}{'Official':>10}")
    print(f"{'LP perfect foresight':<30}" + ''.join(f'{s:>8.3f}' for s in lp_phases)
          + f"{lp['weighted_score']:>10.3f}{lp['official_weighted']:>10.3f}")
    runs = {}
    for label, kind in MAIN:
        r = load(kind)
        if r is None:
            continue
        runs[kind] = r
        ph = [p['score'] for p in r['per_phase']]
        print(f"{label:<30}" + ''.join(f'{s:>8.3f}' for s in ph)
              + f"{r['weighted_score']:>10.3f}{r['weighted_metrics']['official_score']:>10.3f}")
    print(f"{'Baseline (no control)':<30}{1.0:>8.3f}{1.0:>8.3f}{1.0:>8.3f}{1.0:>10.3f}{1.0:>10.3f}")

    if 'perfect' in runs and 'lgb' in runs:
        lp_w = lp['weighted_score']
        perf = runs['perfect']['weighted_score']
        print('\nGap decomposition (LP reference -> MPC+Perfect -> practical MPC)')
        for kind in ('lgb_avg', 'lgb'):
            if kind not in runs:
                continue
            total = runs[kind]['weighted_score'] - lp_w
            rh = perf - lp_w
            print(f"  vs {kind:<8}: total {total:.3f} = receding-horizon {rh:.3f} ({100 * rh / total:.0f}%)"
                  f" + forecasting {total - rh:.3f} ({100 * (total - rh) / total:.0f}%)")

    print('\nPer-metric weighted scores')
    cols = [k for k in ('lgb_avg', 'ensemble', 'lgb', 'persistence') if k in runs]
    print(f"{'Metric':<16}" + ''.join(f'{c:>14}' for c in cols))
    for m in METRICS + ['grid', 'score']:
        print(f"{m:<16}" + ''.join(f"{runs[c]['weighted_metrics'][m]:>14.3f}" for c in cols))

    print('\nAblations (paired against the seed-2022 reference of the same forecaster)')
    print(f"{'Configuration':<46}{'Score':>8}{'Ref':>8}{'Delta':>8}")
    for label, kind, ref in ABLATIONS:
        r = load(kind)
        if r is None or ref not in runs:
            continue
        s, b = r['weighted_score'], runs[ref]['weighted_score']
        print(f"{label:<46}{s:>8.3f}{b:>8.3f}{s - b:>+8.3f}")


if __name__ == '__main__':
    main()
