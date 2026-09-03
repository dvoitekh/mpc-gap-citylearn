# Results

Raw outputs of the experiment scripts. Every number in the paper's tables can be
traced to one of these files; `python -m experiments.summarize_results` and
`python -m experiments.summarize_multiseed` print the tables from them.

## Single-seed evaluation runs (seed 2022)

`verify_mpc_<config>.json`, one per controller/forecaster configuration, written by
`experiments/run_verify_mpc.py`. Each file holds the per-phase metrics
(`cost`, `emissions`, `ramping`, `load_factor`, `daily_peak`, `all_time_peak`, `grid`,
`score`) and the phase-weighted metrics, under both the CityLearn v2 score used in the
paper and the original 2022 competition score (`official_*`).

| File | Configuration |
|---|---|
| `verify_lp.json` | full-horizon LP with perfect foresight (`mpcgap/perfect_foresight_lp.py`) |
| `verify_mpc_perfect.json` | MPC + perfect forecast, S = 5, sigma = 0.3, no action smoothing |
| `verify_mpc_perfect_s1.json` | MPC + perfect forecast, deterministic (S = 1) |
| `verify_mpc_perfect_smooth.json` | MPC + perfect forecast with action smoothing 0.1 |
| `verify_mpc_perfect_term10.json` | MPC + perfect forecast with terminal SOC credit, w = 1.0 |
| `verify_mpc_lgb.json` | MPC + LightGBM (headline) |
| `verify_mpc_lgb_avg.json` | MPC + equal-weight average of LightGBM and Persistence |
| `verify_mpc_lgb_nosmooth.json` | MPC + LightGBM without action smoothing |
| `verify_mpc_lgb_term05.json`, `verify_mpc_lgb_term10.json` | MPC + LightGBM with terminal SOC credit, w = 0.5 / 1.0 |
| `verify_mpc_lgb_days240.json` | MPC + LightGBM trained on days 0-238 only |
| `verify_mpc_lgb_days240_tt.json` | same model retrained from scratch (reproducibility check) |
| `verify_mpc_lgb_noweather.json` | MPC + LightGBM without weather features |
| `verify_mpc_persistence.json`, `verify_mpc_persistence_nosmooth.json` | MPC + Persistence, with / without action smoothing |
| `verify_mpc_holtwinters.json`, `verify_mpc_weekly.json`, `verify_mpc_ensemble.json` | MPC + Holt-Winters / Weekly / statistical Ensemble |

## Multi-seed runs

`verify_multiseed_<config>.json` (seeds 2023-2026; seed 2022 is the single-seed file above),
written by `experiments/run_verify_multiseed.py`. `lgb_noleak` is the LightGBM model
trained without the ten buildings that appear only in Phase 3.

## Forecast accuracy

* `verify_peak_metrics.json`: load MAPE/MAE over all three phases (phase-weighted),
  on peak hours, and on the executed horizons h = 1..8 (`experiments/run_peak_metrics.py`).
* `forecast_metrics_results.txt`: MAPE, MAE, sMAPE on Phase 2 (`experiments/run_forecast_metrics.py`).

## Sensitivity sweeps and other experiments

* `noise_sensitivity_results.txt`: scenario-noise sweep, MPC + LightGBM (`experiments/run_noise_sensitivity.py`).
* `scenario_sensitivity_results.txt`: scenario-count sweep (`experiments/run_scenario_sensitivity.py`).
* `experiment_results.txt`: an earlier full-suite run (`experiments/run_all.py`) that also contains
  the Hybrid (LightGBM load / Persistence solar) and 4-hour re-optimization variants. Its
  reference scores differ from the `verify_*` runs by up to 0.002 because each batch draws
  its own MPC scenarios; ablation deltas are always paired within a batch.
* `dfl_eval_results.txt`: three-phase evaluation of the neural MSE and decision-focused
  forecasters (`experiments/dfl_surrogate.py --eval-only`).
* `verify_edge_orangepi.json`: single-core timing benchmark on an Orange Pi 5 Pro
  (RK3588S) and an Apple M1 Pro (`experiments/run_bench_edge.py`).
