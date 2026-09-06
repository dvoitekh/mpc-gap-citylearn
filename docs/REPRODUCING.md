# Reproducing the paper

All commands run from the repository root with the pinned environment
(`requirements.txt`, Python 3.12.8). Runtimes are for an Apple M1 Pro, single process.
The CityLearn package downloads the `citylearn_challenge_2022_phase_all` dataset on first use.

Every MPC run is seeded (2022 unless stated). Scores are reproducible to the fourth
decimal on the same platform; across platforms the CLARABEL solve path can shift a
score by about 0.001.

## 0. Models

Either download the released models into `models/` (see `models/README.md`) or train them:

```bash
python -m mpcgap.lgb_train --quick             # models/lgb_models.pkl, ~13 min
python -m experiments.run_train_days240        # models/lgb_models_days240.pkl
python -m experiments.run_train_noweather      # models/lgb_models_noweather.pkl
python -m experiments.run_train_sparse         # models/lgb_models_sparse.pkl
python -m mpcgap.lgb_train --quick --exclude-phase3 --output models/lgb_models_no_p3.pkl
```

## 1. Baseline and perfect-foresight reference

| Paper item | Command | Output | Time |
|---|---|---|---|
| No-control baseline (all metrics 1.0) | `python -m mpcgap.evaluate_full` | stdout | 2 min |
| LP perfect foresight by phase and metric | `python -m mpcgap.perfect_foresight_lp` | `results/verify_lp.json` | 6 min |

## 2. Online MPC with every forecaster (main results table, per-metric table, gap decomposition)

One run per configuration, about 60-90 minutes each:

```bash
for c in perfect lgb lgb_avg ensemble persistence holtwinters weekly; do
  python -m experiments.run_verify_mpc $c
done
python -m experiments.summarize_results        # prints the tables and the decomposition
```

`summarize_results` reproduces the main table (per-phase and weighted scores), the
official-metric column of the winner comparison, the per-metric breakdown, and the
receding-horizon / forecasting split.

## 3. Controller-side ablations

| Paper item | Command |
|---|---|
| Action smoothing (perfect / LightGBM / Persistence) | `run_verify_mpc perfect_smooth`, `lgb_nosmooth`, `persistence_nosmooth` |
| Perfect-forecast controller configurations (S = 1, smoothing) | `run_verify_mpc perfect_s1`, `perfect_smooth` |
| Terminal SOC credit | `run_verify_mpc lgb_term05`, `lgb_term10`, `perfect_term10` |
| Scenario-noise sweep (sigma = 0.1 ... 0.5) | `python -m experiments.run_noise_sensitivity` (8 runs, ~10 h) |
| Scenario count (S = 1, 5, 10, 15) | `python -m experiments.run_scenario_sensitivity` (~6 h) |

The remaining controller variants are keyword arguments of `run_mpc_phase` in
`mpcgap/online_mpc.py` and were evaluated with the same three-phase loop as
`experiments/run_verify_mpc.py`:

```python
from mpcgap.evaluate_full import PHASES
from mpcgap.online_mpc import run_mpc_phase
from mpcgap.lgb_forecaster import LGBForecaster

phase = PHASES[0]
fc = LGBForecaster(len(phase['buildings']), sim_start=phase['sim_start'],
                   building_names=phase['buildings'])
run_mpc_phase(phase['buildings'], phase['sim_start'], phase['sim_end'], fc,
              peak_penalty_weight=1.0)        # peak-aware penalty, w in {0.25, 0.5, 1.0}
#             horizon_discount=0.95)          # exponential discount, gamma in {0.95, 0.9}
#             horizon=12)                     # shorter planning horizon (Phase 1 only)
#             reopt_interval=4)               # re-optimize every 4 h instead of 8 h
#             noise_profile=OnlineMPC.LGB_NOISE_PROFILE)   # MAPE-calibrated scenario noise
```

## 4. Forecasting-side ablations

| Paper item | Command / class |
|---|---|
| LightGBM + Persistence average | `run_verify_mpc lgb_avg` |
| Weather-feature ablation | `run_train_noweather`, then `run_verify_mpc lgb_noweather` |
| Sparse-lag feature ablation | `run_train_sparse`, then `run_verify_mpc lgb_sparse` |
| Temporal overlap (train on days 0-238) | `run_train_days240`, then `run_verify_mpc lgb_days240` |
| Building overlap (no Phase-3 buildings in training) | `mpcgap.lgb_train --quick --exclude-phase3 ...`, then `run_verify_multiseed lgb_noleak` |
| Hybrid (LightGBM load, Persistence solar) | `mpcgap.forecasters.HybridForecaster` |
| Horizon-blended (LightGBM h <= 12, Persistence beyond) | `mpcgap.forecasters.HorizonBlendedForecaster` |
| No online correction / solar correction / 30-day window | `LGBForecaster(online_correction=False)`, `correct_solar=True`, `correction_window=30` |
| Sparse-lag feature set | earlier feature-builder revision with four lags (h-1, h-24, h-48, h-168) instead of `N_LAGS = 168` in `mpcgap/lgb_feature_engineering.py`; retrain with `python -m mpcgap.lgb_train --quick` |
| Feature importance | `mpcgap.lgb_train.get_feature_importance` on `models/lgb_models.pkl` |
| Decision-focused learning | `python -m experiments.dfl_surrogate` (data generation + training, ~2 h), then `python -m experiments.dfl_surrogate --eval-only` (~2 h) |

## 5. Forecast accuracy tables

```bash
python -m experiments.run_forecast_metrics     # Phase 2: MAPE, MAE, sMAPE (load and solar)
python -m experiments.run_peak_metrics         # all phases, peak hours, executed horizons
python -m mpcgap.lgb_evaluate --forecast       # per-horizon MAPE (also cached for figures)
```

## 6. Multi-seed reproducibility and paired tests

```bash
for c in perfect lgb lgb_avg ensemble persistence; do
  python -m experiments.run_verify_multiseed $c      # seeds 2023-2026, ~5 h each
done
python -m experiments.summarize_multiseed
```

## 7. Edge benchmark

```bash
python -m experiments.run_bench_edge --prepare   # extracts one 24 h window into experiments/bench_data.npz
OMP_NUM_THREADS=1 python -m experiments.run_bench_edge
```

Copy `experiments/run_bench_edge.py`, `experiments/bench_data.npz`, the `mpcgap/`
package and `models/lgb_models.pkl` to the target device with the same directory layout;
the benchmark needs only numpy, pandas, cvxpy, clarabel, lightgbm, scikit-learn and joblib.

## 8. Figures

```bash
python figures/generate_figures.py
python figures/generate_forecast_fig.py
python figures/generate_noise_fig.py
```
