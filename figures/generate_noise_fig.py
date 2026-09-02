"""Scenario-noise sensitivity figure (values from results/noise_sensitivity_results.txt)."""
import os
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)
import matplotlib
matplotlib.use('Agg')

sigma = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
weighted = [0.8381, 0.8373, 0.8376, 0.8385, 0.8397, 0.8403, 0.8417, 0.8449]

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))

ax.plot(sigma, weighted, 'o-', color='#2563eb', linewidth=1.5, markersize=5, zorder=3)
ax.axhline(y=0.849, color='#dc2626', linestyle='--', linewidth=1, alpha=0.7,
           label='MAPE-calibrated ($\\sigma$=0.48–1.16)')
ax.axhspan(0.8373 - 0.002, 0.8373 + 0.002, alpha=0.1, color='#2563eb',
           label='Inter-seed variation (±0.002)')

ax.set_xlabel('Scenario noise $\\sigma$', fontsize=9)
ax.set_ylabel('Weighted MPC score (lower is better)', fontsize=9)
ax.set_xlim(0.05, 0.55)
ax.set_ylim(0.835, 0.852)
ax.legend(fontsize=7, loc='upper left')
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_noise_sensitivity.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'fig_noise_sensitivity.png'), dpi=150, bbox_inches='tight')
print("Saved fig_noise_sensitivity.pdf/png")
