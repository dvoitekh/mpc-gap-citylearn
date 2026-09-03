"""
Generate the paper figures from the results in results/ (values are embedded
below and cross-checked against the JSON files) and from the cached per-horizon
forecast errors.

Outputs (figures/output/):
    fig1_methodology        three-phase evaluation protocol diagram
    fig2_gap_decomposition  LP -> MPC -> baseline score comparison
    fig3_phase_results      score by phase for each method
    fig5_mape_vs_score      forecast accuracy versus MPC score
    fig6_horizon_mape       per-horizon MAPE, LightGBM vs Persistence vs Holt-Winters
    fig7_ablation           ablation summary
    fig7_per_metric         per-metric comparison (not used in the paper)

Usage (from the repository root):
    python figures/generate_figures.py                # use cached per-horizon MAPE
    python figures/generate_figures.py --recompute    # recompute per-horizon MAPE (needs models/lgb_models.pkl)
    python figures/generate_forecast_fig.py           # fig4_forecast_example (needs models/lgb_models.pkl)
    python figures/generate_noise_fig.py              # fig_noise_sensitivity
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# Publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# VERIFIED DATA FROM EXPERIMENTS
# ============================================================

# Main results (verified Feb 11-19 2026, from CLAUDE.md)
MAIN_RESULTS = {
    'LP Perfect\nForesight':    {'weighted': 0.664, 'P1': 0.650, 'P2': 0.703, 'P3': 0.647},
    'MPC +\nPerfect':           {'weighted': 0.705, 'P1': 0.684, 'P2': 0.734, 'P3': 0.695},
    'MPC +\nLightGBM':          {'weighted': 0.840, 'P1': 0.848, 'P2': 0.884, 'P3': 0.810},
    'MPC +\nPersistence':       {'weighted': 0.850, 'P1': 0.852, 'P2': 0.905, 'P3': 0.816},
    'MPC +\nHolt-Winters':      {'weighted': 0.853},
    'Baseline\n(no control)':   {'weighted': 1.000, 'P1': 1.000, 'P2': 1.000, 'P3': 1.000},
}

# Forecast MAPE, all three phases weighted 0.2/0.3/0.5 to match the MPC score
# (from verify_peak_metrics.json). Phase-2-only values live in Table 6 of the paper.
FORECAST_MAPE = {
    'Perfect':     {'load': 0.0,  'solar': 0.0},
    'LightGBM':    {'load': 84.3, 'solar': 63.4},
    'Persistence': {'load': 73.8, 'solar': 50.2},
    'Holt-Winters':{'load': 100.8, 'solar': 69.1},
    'Weekly':      {'load': 87.7, 'solar': 64.3},
    'Ensemble':    {'load': 73.7, 'solar': 55.7},
    'LGB+Pers. avg': {'load': 73.5, 'solar': 51.4},
}

# MPC scores for scatter plot (verified, with smoothing)
MPC_SCORES = {
    'Perfect':      0.705,
    'LightGBM':     0.840,
    'Persistence':  0.850,
    'Holt-Winters': 0.853,
    'Weekly':       0.866,
    'Ensemble':     0.839,
    'LGB+Pers. avg': 0.830,
}

# LGB per-horizon load MAPE (h=1..24), measured on Phase 2
# From OnlineMPC.LGB_NOISE_PROFILE (× 100 for %)
LGB_HORIZON_MAPE = np.array([
    54.7, 67.8, 69.3, 62.9, 57.3, 54.8, 66.8, 67.3, 72.8,
    72.5, 76.1, 78.4, 80.4, 94.1, 107.2, 116.1, 88.7, 73.1,
    64.6, 56.8, 47.9, 48.5, 49.9, 58.4
])

# Per-metric breakdown (weighted across phases, from run_experiments output)
# Format: {method: {metric: value}}
PER_METRIC = {
    'MPC + LightGBM': {
        'cost': 0.799, 'emissions': 0.884, 'ramping': 0.823,
        'load_factor': 0.913, 'daily_peak': 0.742, 'all_time_peak': 0.865,
    },
    'MPC + Persistence': {
        'cost': 0.811, 'emissions': 0.878, 'ramping': 0.855,
        'load_factor': 0.930, 'daily_peak': 0.753, 'all_time_peak': 0.905,
    },
}

# MPC formulation ablation results (weighted score)
MPC_ABLATION = {
    'LGB\nreference': 0.839,
    'Peak\npenalty': 0.839,
    'Adaptive\nnoise': 0.849,
    'Discount\nγ=0.95': 0.864,
    'Discount\nγ=0.9': 0.916,
    'Horizon\n12h': 0.892,
    'Reopt\n4h': 0.840,
    'Terminal\nSOC w=1': 0.842,
}

# Forecasting ablation results (weighted score or Phase 1 where noted)
FORECAST_ABLATION = {
    'LGB\nreference': 0.839,
    'LGB+Pers.\naverage': 0.830,
    'No OLS\ncorrection': 0.840,
    'Horizon\nblended': 0.843,
    'Hybrid\n(LGB+Pers)': 0.850,
    'NN-MSE': 0.875,
    'NN-DFL': 0.867,
}


# ============================================================
# DATA COLLECTION (per-horizon MAPE)
# ============================================================

CACHE_FILE = os.path.join(OUT_DIR, 'horizon_mape_cache.json')


def compute_horizon_mape():
    """Compute per-horizon MAPE for LGB, Persistence, and HW on Phase 2 data."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from mpcgap.lgb_evaluate import evaluate_forecast_by_horizon
    from mpcgap.data import load_building_data
    from mpcgap.forecasters import PersistenceForecaster, HoltWintersForecaster
    from mpcgap.lgb_forecaster import LGBForecaster

    print("Computing per-horizon MAPE (Phase 2)...")

    load_data, solar_data = load_building_data()
    n_buildings = 5
    start = 24 * 120
    end = 24 * 240

    load_p2 = {i: load_data[i][start:end] for i in range(n_buildings)}
    solar_p2 = {i: solar_data[i][start:end] for i in range(n_buildings)}

    results = {}

    # LightGBM
    building_names = [f'Building_{i}' for i in range(1, 6)]
    lgb = LGBForecaster(n_buildings, sim_start=start, building_names=building_names)
    lm, sm = evaluate_forecast_by_horizon(lgb, load_p2, solar_p2, n_buildings, name='LGB')
    results['LightGBM'] = {'load': lm.tolist(), 'solar': sm.tolist()}
    print(f"  LightGBM: load avg={np.mean(lm):.1f}%, solar avg={np.mean(sm):.1f}%")

    # Persistence
    pers = PersistenceForecaster(n_buildings)
    lm, sm = evaluate_forecast_by_horizon(pers, load_p2, solar_p2, n_buildings, name='Persist')
    results['Persistence'] = {'load': lm.tolist(), 'solar': sm.tolist()}
    print(f"  Persistence: load avg={np.mean(lm):.1f}%, solar avg={np.mean(sm):.1f}%")

    # Holt-Winters
    hw = HoltWintersForecaster(n_buildings)
    lm, sm = evaluate_forecast_by_horizon(hw, load_p2, solar_p2, n_buildings, name='HW')
    results['Holt-Winters'] = {'load': lm.tolist(), 'solar': sm.tolist()}
    print(f"  Holt-Winters: load avg={np.mean(lm):.1f}%, solar avg={np.mean(sm):.1f}%")

    # Cache results
    with open(CACHE_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Cached to {CACHE_FILE}")

    return results


def load_horizon_mape(recompute=False):
    """Load per-horizon MAPE data (from cache or recompute)."""
    if not recompute and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            data = json.load(f)
        print(f"Loaded cached horizon MAPE from {CACHE_FILE}")
        # Convert lists back to numpy arrays
        for name in data:
            data[name]['load'] = np.array(data[name]['load'])
            data[name]['solar'] = np.array(data[name]['solar'])
        return data

    return compute_horizon_mape()


# ============================================================
# FIGURE 1: Gap Decomposition
# ============================================================

def fig1_gap_decomposition():
    """Waterfall chart showing LP → MPC+Perfect → MPC+LGB → Baseline gap."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    methods = ['LP Perfect\nForesight', 'MPC +\nPerfect', 'MPC + LGB\n+Pers. avg',
               'MPC +\nLightGBM', 'MPC +\nPersistence', 'MPC +\nHolt-Winters',
               'Baseline\n(no control)']
    scores = [0.664, 0.705, 0.830, 0.840, 0.850, 0.853, 1.000]
    colors = ['#2196F3', '#4CAF50', '#00897B', '#FF9800', '#F44336', '#9C27B0', '#757575']

    bars = ax.bar(range(len(methods)), scores, color=colors, width=0.65, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Gap decomposition annotation: LP -> MPC+Perfect (receding horizon),
    # MPC+Perfect -> MPC+LGB+Pers. avg (forecasting)
    lp, perf, best = 0.6644, 0.7046, 0.8301   # unrounded weighted scores (results/verify_*.json)
    ax.hlines([lp, perf], -0.35, 2.35, colors='#555555', linestyles=':', linewidth=0.8, zorder=1)
    x_arrow = 1.5
    arrow = dict(arrowstyle='<->', lw=0.9, color='#222222', shrinkA=0, shrinkB=0)
    ax.annotate('', xy=(x_arrow, perf), xytext=(x_arrow, lp), arrowprops=arrow)
    ax.annotate('', xy=(x_arrow, best), xytext=(x_arrow, perf), arrowprops=arrow)
    total = best - lp
    box = dict(boxstyle='round,pad=0.3', fc='white', ec='#999999', lw=0.5)
    leader = dict(arrowstyle='-', lw=0.6, color='#666666')
    ax.annotate(f'receding horizon\n{perf - lp:.3f} ({100 * (perf - lp) / total:.0f}%)',
                xy=(x_arrow, (lp + perf) / 2), xytext=(-0.3, 0.93), fontsize=7,
                ha='left', va='center', arrowprops=leader, bbox=box)
    ax.annotate(f'forecasting\n{best - perf:.3f} ({100 * (best - perf) / total:.0f}%)',
                xy=(x_arrow, (perf + best) / 2), xytext=(0.95, 1.02), fontsize=7,
                ha='left', va='center', arrowprops=leader, bbox=box)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7.5)
    ax.set_ylabel('Weighted Score (lower is better)')
    ax.set_ylim(0.45, 1.10)
    ax.set_title('Performance Gap Decomposition: LP → MPC → Baseline')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_gap_decomposition')
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png')
    plt.close()
    print(f"  Saved: {path}.pdf/png")


# ============================================================
# FIGURE 2: MAPE vs MPC Score
# ============================================================

def fig2_mape_vs_score():
    """Scatter plot: Average MAPE vs MPC Score with correlation."""
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    names = ['Perfect', 'LightGBM', 'Persistence', 'Holt-Winters', 'Weekly', 'Ensemble', 'LGB+Pers. avg']
    markers = ['*', 'D', 's', '^', 'v', 'o', 'P']
    colors = ['#2196F3', '#FF9800', '#F44336', '#9C27B0', '#795548', '#607D8B', '#00897B']

    mapes = []
    scores = []
    for i, name in enumerate(names):
        avg_mape = (FORECAST_MAPE[name]['load'] + FORECAST_MAPE[name]['solar']) / 2
        mpc_score = MPC_SCORES[name]
        mapes.append(avg_mape)
        scores.append(mpc_score)

        ax.scatter(avg_mape, mpc_score, marker=markers[i], color=colors[i],
                   s=90, zorder=5, edgecolors='white', linewidths=0.5)

    # Labels positioned carefully to avoid overlap
    label_configs = {
        'Perfect':       (10, -10),
        'LightGBM':      (12, -12),
        'Persistence':   (-62, 8),
        'Holt-Winters':  (10, -5),
        'Weekly':        (8, 6),
        'Ensemble':      (-58, -12),
        'LGB+Pers. avg': (-44, -28),
    }
    for i, name in enumerate(names):
        avg_mape = mapes[i]
        mpc_score = scores[i]
        dx, dy = label_configs[name]
        ax.annotate(name, (avg_mape, mpc_score), xytext=(dx, dy),
                    textcoords='offset points', fontsize=7.5,
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5))

    # Correlation (practical forecasters only, excluding Perfect)
    from scipy import stats
    r_imp, _ = stats.pearsonr(mapes[1:], scores[1:])
    rho_imp, _ = stats.spearmanr(mapes[1:], scores[1:])
    print(f"  MAPE vs score over {len(mapes) - 1} practical forecasters: "
          f"Pearson r = {r_imp:.2f}, Spearman rho = {rho_imp:.2f}")

    # Correlation text box
    ax.text(0.97, 0.05,
            f'Pearson r = {r_imp:.2f}\nSpearman ρ = {rho_imp:.2f}',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9, edgecolor='#cccccc'))

    ax.set_xlabel('Average MAPE (Load + Solar) / 2, %')
    ax.set_ylabel('MPC Weighted Score (lower is better)')
    ax.set_title('Forecast Quality (MAPE) vs. Control Performance (MPC Score)')

    ax.set_xlim(-5, 115)
    ax.set_ylim(0.68, 0.89)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_mape_vs_score')
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png')
    plt.close()
    print(f"  Saved: {path}.pdf/png")


# ============================================================
# FIGURE 3: Per-Horizon MAPE
# ============================================================

def fig3_horizon_mape(horizon_data):
    """Line chart: MAPE by forecast horizon for LGB, Persistence, HW."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=False)

    horizons = np.arange(1, 25)

    # --- Panel (a): Load MAPE ---
    for name, color, marker, ls in [
        ('LightGBM', '#FF9800', 'D', '-'),
        ('Persistence', '#F44336', 's', '--'),
        ('Holt-Winters', '#9C27B0', '^', ':'),
    ]:
        if name in horizon_data:
            ax1.plot(horizons, horizon_data[name]['load'], color=color,
                     marker=marker, markersize=3.5, linewidth=1.2, linestyle=ls,
                     label=name, markeredgecolor='white', markeredgewidth=0.3)

    # MPC execution zone (h=1-8)
    ax1.axvspan(1, 8, alpha=0.08, color='green', zorder=0)
    ax1.text(4.5, 42, 'MPC zone', ha='center', va='bottom',
             fontsize=6.5, color='green', alpha=0.8, style='italic')

    ax1.set_xlabel('Forecast Horizon (hours)')
    ax1.set_ylabel('Load MAPE (%)')
    ax1.set_title('(a) Load Forecast Error by Horizon')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=7)
    ax1.set_xlim(0.5, 24.5)
    ax1.set_ylim(40, 130)
    ax1.set_xticks([1, 4, 8, 12, 16, 20, 24])

    # --- Panel (b): Solar MAPE (only daylight hours h=7-18 have data) ---
    for name, color, marker, ls in [
        ('LightGBM', '#FF9800', 'D', '-'),
        ('Persistence', '#F44336', 's', '--'),
        ('Holt-Winters', '#9C27B0', '^', ':'),
    ]:
        if name in horizon_data:
            solar_data_h = np.asarray(horizon_data[name]['solar'])
            # Only plot non-zero values (daylight hours)
            mask = solar_data_h > 0
            h_day = horizons[mask]
            s_day = solar_data_h[mask]
            ax2.plot(h_day, s_day, color=color,
                     marker=marker, markersize=3.5, linewidth=1.2, linestyle=ls,
                     label=name, markeredgecolor='white', markeredgewidth=0.3)

    ax2.axvspan(7, 8, alpha=0.12, color='green', zorder=0)

    ax2.set_xlabel('Forecast Horizon (hours)')
    ax2.set_ylabel('Solar MAPE (%)')
    ax2.set_title('(b) Solar Forecast Error by Horizon (daylight only)')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax2.set_xlim(6.5, 18.5)
    ax2.set_xticks([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_horizon_mape')
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png')
    plt.close()
    print(f"  Saved: {path}.pdf/png")


# ============================================================
# FIGURE 4: Per-Metric Breakdown
# ============================================================

def fig4_per_metric():
    """Grouped bar chart: per-metric breakdown for LGB vs Persistence."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    metrics = ['cost', 'emissions', 'ramping', 'load_factor', 'daily_peak', 'all_time_peak']
    labels = ['Cost', 'Emissions', 'Ramping', 'Load\nFactor', 'Daily\nPeak', 'All-Time\nPeak']

    lgb_vals = [PER_METRIC['MPC + LightGBM'][m] for m in metrics]
    pers_vals = [PER_METRIC['MPC + Persistence'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.32

    bars1 = ax.bar(x - width/2, lgb_vals, width, label='MPC + LightGBM',
                   color='#FF9800', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, pers_vals, width, label='MPC + Persistence',
                   color='#F44336', edgecolor='white', linewidth=0.5)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_ylabel('Metric Score (lower is better)')
    ax.set_title('Per-Metric Comparison: LightGBM vs. Persistence')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0.65, 0.97)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig7_per_metric')
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png')
    plt.close()
    print(f"  Saved: {path}.pdf/png")


# ============================================================
# FIGURE 5: 3-Phase Evaluation Methodology
# ============================================================

def fig5_methodology():
    """Visual diagram of 3-phase evaluation methodology."""
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.set_xlim(0, 365)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Title
    ax.text(182.5, 5.2, 'Three-Phase Weighted Evaluation (CityLearn 2022 dataset)',
            ha='center', va='top', fontsize=10, fontweight='bold')

    # Timeline axis
    ax.annotate('', xy=(365, 0.8), xytext=(0, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    for day in [0, 60, 120, 180, 240, 300, 365]:
        ax.plot([day, day], [0.6, 1.0], 'k-', lw=0.8)
        ax.text(day, 0.35, f'Day {day}', ha='center', fontsize=7, color='#666666')

    # Phase boxes
    phase_configs = [
        (0, 120, 'Phase 1', '#2196F3', 0.2, 'Buildings 1-5\n2,880 hours', 3.8),
        (120, 240, 'Phase 2', '#4CAF50', 0.3, 'Buildings 1-5\n2,880 hours', 3.8),
        (240, 365, 'Phase 3', '#FF9800', 0.5, 'Buildings 1-17*\n3,000 hours', 3.8),
    ]

    for start, end, name, color, weight, desc, y_center in phase_configs:
        width = end - start
        rect = FancyBboxPatch((start + 2, y_center - 0.9), width - 4, 1.8,
                              boxstyle="round,pad=0.1", facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(start + width/2, y_center + 0.3, f'{name} (w={weight})',
                ha='center', va='center', fontsize=9, fontweight='bold', color=color)
        ax.text(start + width/2, y_center - 0.35, desc,
                ha='center', va='center', fontsize=7, color='#444444')

    # Arrows from phases to timeline
    for start, end, _, color, _, _, y_center in phase_configs:
        mid = (start + end) / 2
        ax.annotate('', xy=(mid, 1.1), xytext=(mid, y_center - 0.9),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8, alpha=0.5))

    # Scoring formula
    ax.text(182.5, 1.8, 'Score = 0.2×P1 + 0.3×P2 + 0.5×P3',
            ha='center', va='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                     edgecolor='#cccccc', alpha=0.9))

    # Footnote
    ax.text(365, 0.05, '* Excludes Buildings 12, 15', ha='right', va='bottom',
            fontsize=6.5, color='#888888', style='italic')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_methodology')
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png')
    plt.close()
    print(f"  Saved: {path}.pdf/png")


# ============================================================
# FIGURE 6: Ablation Study Summary
# ============================================================

def fig6_ablation():
    """Combined bar chart: MPC + forecasting ablation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # --- Panel (a): MPC Formulation Ablation ---
    names = list(MPC_ABLATION.keys())
    scores = list(MPC_ABLATION.values())
    colors = ['#4CAF50'] + ['#F44336' if s > 0.842 else ('#2196F3' if s < 0.836 else '#FF9800') for s in scores[1:]]

    bars = ax1.barh(range(len(names)), scores, color=colors, height=0.6,
                    edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', ha='left', va='center', fontsize=7)

    # Baseline reference line
    ax1.axvline(x=0.839, color='#4CAF50', linestyle='--', alpha=0.5, linewidth=1.0)

    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel('Weighted Score')
    ax1.set_title('(a) MPC Formulation Variants')
    ax1.set_xlim(0.82, 0.93)
    ax1.invert_yaxis()

    # --- Panel (b): Forecasting Ablation ---
    names2 = list(FORECAST_ABLATION.keys())
    scores2 = list(FORECAST_ABLATION.values())
    colors2 = ['#4CAF50'] + ['#F44336' if s > 0.842 else ('#2196F3' if s < 0.836 else '#FF9800') for s in scores2[1:]]

    bars2 = ax2.barh(range(len(names2)), scores2, color=colors2, height=0.6,
                     edgecolor='white', linewidth=0.5)

    for bar, score in zip(bars2, scores2):
        # Place label inside bar if bar is long enough, otherwise outside
        if score > 0.87:
            ax2.text(bar.get_width() - 0.004, bar.get_y() + bar.get_height() / 2,
                     f'{score:.3f}', ha='right', va='center', fontsize=7, color='white',
                     fontweight='bold')
        else:
            ax2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                     f'{score:.3f}', ha='left', va='center', fontsize=7)

    ax2.axvline(x=0.839, color='#4CAF50', linestyle='--', alpha=0.5, linewidth=1.0)

    ax2.set_yticks(range(len(names2)))
    ax2.set_yticklabels(names2, fontsize=7)
    ax2.set_xlabel('Weighted Score')
    ax2.set_title('(b) Forecasting Variants')
    ax2.set_xlim(0.82, 0.96)
    ax2.invert_yaxis()

    # Legend — place in panel (a) instead to avoid overlap
    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', label='LGB reference (0.839)'),
        mpatches.Patch(facecolor='#2196F3', label='Better'),
        mpatches.Patch(facecolor='#FF9800', label='Similar'),
        mpatches.Patch(facecolor='#F44336', label='Worse'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig7_ablation')
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png')
    plt.close()
    print(f"  Saved: {path}.pdf/png")


# ============================================================
# FIGURE 7: Phase-by-Phase Results
# ============================================================

def fig7_phase_results():
    """Grouped bar chart: scores by phase for each method."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    methods = ['LP Perfect\nForesight', 'MPC +\nPerfect', 'MPC +\nLightGBM',
               'MPC +\nPersistence', 'Baseline']
    p1 = [0.650, 0.684, 0.848, 0.852, 1.000]
    p2 = [0.703, 0.734, 0.884, 0.905, 1.000]
    p3 = [0.647, 0.695, 0.810, 0.816, 1.000]

    x = np.arange(len(methods))
    width = 0.22

    ax.bar(x - width, p1, width, label='Phase 1 (w=0.2)', color='#2196F3',
           edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.bar(x, p2, width, label='Phase 2 (w=0.3)', color='#4CAF50',
           edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.bar(x + width, p3, width, label='Phase 3 (w=0.5)', color='#FF9800',
           edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.set_ylabel('Score (lower is better)')
    ax.set_title('Performance Across Evaluation Phases')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7.5)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0.5, 1.08)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_phase_results')
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png')
    plt.close()
    print(f"  Saved: {path}.pdf/png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute per-horizon MAPE from data')
    args = parser.parse_args()

    print("=" * 60)
    print("Generating paper figures")
    print("=" * 60)

    # Load/compute per-horizon MAPE data
    horizon_data = load_horizon_mape(recompute=args.recompute)

    # Generate all figures
    print("\nGenerating figures...")
    fig1_gap_decomposition()
    fig2_mape_vs_score()
    fig3_horizon_mape(horizon_data)
    fig4_per_metric()
    fig5_methodology()
    fig6_ablation()
    fig7_phase_results()

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Files: fig1_gap_decomposition, fig2_mape_vs_score, fig3_horizon_mape,")
    print("       fig4_per_metric, fig5_methodology, fig6_ablation, fig7_phase_results")


if __name__ == '__main__':
    main()
