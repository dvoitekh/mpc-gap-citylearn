"""
Generate forecast example figure (Fig 8) for the paper.

Shows actual vs LGB vs Persistence predictions for representative days.
Picks a window that includes both a typical day and a high-peak day.

Usage:
    python paper/generate_forecast_fig.py
"""

import os
import sys
import numpy as np

# Add parent dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

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

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    from mpcgap.data import load_building_data
    from mpcgap.lgb_forecaster import LGBForecaster
    from mpcgap.forecasters import PersistenceForecaster

    print("Loading building data...")
    load_data, solar_data = load_building_data()

    # Phase 2: Buildings 1-5, days 120-240
    n_buildings = 5
    phase_start = 24 * 120  # hour 2880
    phase_end = 24 * 240    # hour 5760
    building_names = [f'Building_{i}' for i in range(1, 6)]

    # Use Building 1 (index 0) for the example
    building_idx = 0

    # Create forecasters
    print("Creating forecasters...")
    lgb = LGBForecaster(n_buildings, sim_start=phase_start,
                         building_names=building_names)
    pers = PersistenceForecaster(n_buildings)

    # Run through Phase 2, collecting predictions at every hour 0
    print("Running forecasters through Phase 2...")
    T = phase_end - phase_start

    # Storage: at each prediction point (hour 0), store 24h predictions
    predictions = []  # list of dicts

    for t in range(T):
        hour = t % 24
        global_t = phase_start + t

        # Update both forecasters with actual data
        for b in range(n_buildings):
            actual_load = load_data[b][global_t]
            actual_solar = solar_data[b][global_t]
            lgb.update(b, actual_load, actual_solar)
            pers.update(b, actual_load, actual_solar)

        # At hour 0, make predictions for the next 24 hours
        if hour == 0 and t >= 48 and t + 24 < T:
            lgb_load, lgb_solar = lgb.predict(building_idx, t, hour)
            pers_load, pers_solar = pers.predict(building_idx, t, hour)

            actual_load_24 = [load_data[building_idx][global_t + h + 1]
                              for h in range(24)]
            actual_solar_24 = [solar_data[building_idx][global_t + h + 1]
                               for h in range(24)]

            predictions.append({
                'day': t // 24,
                'global_start': global_t,
                'lgb_load': lgb_load,
                'lgb_solar': lgb_solar,
                'pers_load': pers_load,
                'pers_solar': pers_solar,
                'actual_load': np.array(actual_load_24),
                'actual_solar': np.array(actual_solar_24),
            })

    print(f"  Collected {len(predictions)} daily predictions")

    # Find interesting windows:
    # 1) Day with highest actual peak load (shows peak detection)
    # 2) A typical day (median peak)
    peak_loads = [p['actual_load'].max() for p in predictions]
    peak_idx = np.argmax(peak_loads)
    median_peak = np.median(peak_loads)
    typical_idx = np.argmin(np.abs(np.array(peak_loads) - median_peak))

    # Make sure we pick days that aren't adjacent
    if abs(peak_idx - typical_idx) < 3:
        # Find another typical day further away
        for i in range(len(predictions)):
            if abs(i - peak_idx) > 5 and abs(peak_loads[i] - median_peak) < 0.3:
                typical_idx = i
                break

    print(f"  Peak day: prediction #{peak_idx} (day {predictions[peak_idx]['day']}, "
          f"max load = {peak_loads[peak_idx]:.2f} kW)")
    print(f"  Typical day: prediction #{typical_idx} (day {predictions[typical_idx]['day']}, "
          f"max load = {peak_loads[typical_idx]:.2f} kW)")

    # --- Generate Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5))

    hours = np.arange(1, 25)  # Forecast horizons h=1..24
    hour_labels = [(h) % 24 for h in hours]  # Actual hours of day

    for col, (idx, title_suffix) in enumerate([
        (typical_idx, 'Typical Day'),
        (peak_idx, 'Peak Day')
    ]):
        p = predictions[idx]
        day_num = p['day'] + 120  # Convert back to absolute day
        x_hours = np.arange(24)  # 0-23 for the predicted hours 1-24

        # Hour labels for x-axis (actual hours of day: 1, 2, ..., 24 → 0)
        x_labels = [(h + 1) % 24 for h in range(24)]

        # --- Load (top row) ---
        ax = axes[0, col]
        ax.plot(x_hours, p['actual_load'], 'k-', linewidth=2.0,
                label='Actual', zorder=5)
        ax.plot(x_hours, p['lgb_load'], '-', color='#FF9800', linewidth=1.5,
                label='LightGBM', alpha=0.9, zorder=4)
        ax.plot(x_hours, p['pers_load'], '--', color='#F44336', linewidth=1.3,
                label='Persistence', alpha=0.8, zorder=3)

        # Shade the MPC execution zone (h=1-8 → first 8 hours)
        ax.axvspan(0, 7, alpha=0.06, color='green', zorder=0)

        ax.set_title(f'{title_suffix} (Day {day_num}, Building 1)', fontsize=9)
        ax.set_ylabel('Load (kW)')
        ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
        ax.set_xticklabels([f'{x_labels[i]:02d}:00' for i in [0, 4, 8, 12, 16, 20, 23]],
                           rotation=30, fontsize=7)
        # Add MAPE annotation
        mask_load = p['actual_load'] > 0.1
        if mask_load.any():
            lgb_mape = np.mean(np.abs(p['lgb_load'][mask_load] - p['actual_load'][mask_load])
                               / p['actual_load'][mask_load]) * 100
            pers_mape = np.mean(np.abs(p['pers_load'][mask_load] - p['actual_load'][mask_load])
                                / p['actual_load'][mask_load]) * 100
            ax.text(0.97, 0.95,
                    f'MAPE: LGB {lgb_mape:.0f}%, Pers {pers_mape:.0f}%',
                    transform=ax.transAxes, fontsize=6.5, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                             alpha=0.9, edgecolor='#cccccc'))

        # Increase y-axis headroom
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)

        # --- Solar (bottom row) ---
        ax = axes[1, col]
        ax.plot(x_hours, p['actual_solar'], 'k-', linewidth=2.0,
                label='Actual', zorder=5)
        ax.plot(x_hours, p['lgb_solar'], '-', color='#FF9800', linewidth=1.5,
                label='LightGBM', alpha=0.9, zorder=4)
        ax.plot(x_hours, p['pers_solar'], '--', color='#F44336', linewidth=1.3,
                label='Persistence', alpha=0.8, zorder=3)

        ax.axvspan(0, 7, alpha=0.06, color='green', zorder=0)

        ax.set_ylabel('Solar (kW)')
        ax.set_xlabel('Hour of Day')
        ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
        ax.set_xticklabels([f'{x_labels[i]:02d}:00' for i in [0, 4, 8, 12, 16, 20, 23]],
                           rotation=30, fontsize=7)

        # Solar MAPE (daylight only)
        daylight = np.array([(x_labels[h] >= 7) and (x_labels[h] < 20) for h in range(24)])
        mask_solar = (p['actual_solar'] > 0.1) & daylight
        if mask_solar.any():
            lgb_s_mape = np.mean(np.abs(p['lgb_solar'][mask_solar] - p['actual_solar'][mask_solar])
                                  / p['actual_solar'][mask_solar]) * 100
            pers_s_mape = np.mean(np.abs(p['pers_solar'][mask_solar] - p['actual_solar'][mask_solar])
                                   / p['actual_solar'][mask_solar]) * 100
            ax.text(0.97, 0.95,
                    f'MAPE: LGB {lgb_s_mape:.0f}%, Pers {pers_s_mape:.0f}%',
                    transform=ax.transAxes, fontsize=6.5, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                             alpha=0.9, edgecolor='#cccccc'))

    # Add "MPC zone" label to top-left panel only (top of green area)
    axes[0, 0].text(3.5, axes[0, 0].get_ylim()[1] * 0.92,
                    'MPC zone',
                    ha='center', va='top', fontsize=6.5, color='green',
                    alpha=0.8, style='italic')

    # Shared legend at top of figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               framealpha=0.9, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(h_pad=1.0, w_pad=0.5, rect=[0, 0, 1, 0.96])

    path = os.path.join(OUT_DIR, 'fig4_forecast_example')
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png')
    plt.close()
    print(f"\nSaved: {path}.pdf/png")


if __name__ == '__main__':
    main()
