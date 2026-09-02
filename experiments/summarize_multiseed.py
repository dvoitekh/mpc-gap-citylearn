"""
Aggregate the multi-seed runs (seeds 2022-2026) in results/: mean, standard
deviation, 95% confidence interval, and one-sided paired Wilcoxon tests for
the comparisons discussed in the paper.

Seed 2022 comes from results/verify_mpc_<config>.json, seeds 2023-2026 from
results/verify_multiseed_<config>.json (both produced by the run_verify_*
scripts). The seed only affects scenario sampling inside the MPC, so the
intervals measure Monte-Carlo noise of the controller, not variation across
buildings or years.

Usage (from the repository root):
    python -m experiments.summarize_multiseed
"""
import json
import os

import numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
CONFIGS = [
    ('MPC + Perfect', 'perfect'),
    ('MPC + LGB+Persistence avg', 'lgb_avg'),
    ('MPC + Ensemble', 'ensemble'),
    ('MPC + LightGBM', 'lgb'),
    ('MPC + LightGBM (no Phase-3 buildings in training)', 'lgb_noleak'),
    ('MPC + Persistence', 'persistence'),
]
# (better, worse): one-sided test that `better` scores lower than `worse`
PAIRS = [('lgb', 'persistence'), ('lgb_avg', 'lgb'), ('ensemble', 'lgb')]


def scores(kind):
    out = {}
    single = os.path.join(RESULTS, f'verify_mpc_{kind}.json')
    if os.path.exists(single):
        with open(single) as f:
            out[2022] = json.load(f)['weighted_score']
    multi = os.path.join(RESULTS, f'verify_multiseed_{kind}.json')
    if os.path.exists(multi):
        with open(multi) as f:
            for seed, r in json.load(f)['seeds'].items():
                out[int(seed)] = r['weighted']
    return dict(sorted(out.items()))


def main():
    all_scores = {}
    print(f"{'Configuration':<52}{'n':>3}{'mean':>8}{'std':>8}{'95% CI':>18}")
    for label, kind in CONFIGS:
        s = scores(kind)
        if not s:
            continue
        all_scores[kind] = s
        v = np.array(list(s.values()))
        mean, std = v.mean(), v.std(ddof=1)
        half = 1.96 * std / np.sqrt(len(v))
        print(f"{label:<52}{len(v):>3}{mean:>8.4f}{std:>8.4f}   [{mean - half:.4f}, {mean + half:.4f}]")
        print(f"{'':<52}   seeds " + ', '.join(f'{k}: {x:.4f}' for k, x in s.items()))

    print('\nPaired comparisons (same seeds), one-sided Wilcoxon signed-rank test')
    for a, b in PAIRS:
        if a not in all_scores or b not in all_scores:
            continue
        common = sorted(set(all_scores[a]) & set(all_scores[b]))
        da = np.array([all_scores[a][k] for k in common])
        db = np.array([all_scores[b][k] for k in common])
        diff = db - da
        p = stats.wilcoxon(db, da, alternative='greater').pvalue
        print(f"  {a} vs {b}: n={len(common)}, mean advantage {diff.mean():+.4f} "
              f"(min {diff.min():+.4f}, max {diff.max():+.4f}), "
              f"{int((diff > 0).sum())}/{len(common)} seeds favour {a}, p = {p:.3f}")


if __name__ == '__main__':
    main()
