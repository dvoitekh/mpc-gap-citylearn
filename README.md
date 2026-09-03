# mpc-gap-citylearn

Code, trained models and raw results for the paper
**"Decomposing the Model Predictive Control Performance Gap in Edge-Deployed Building Energy Management"**
(D. Voitekh, A. Tymoshenko; manuscript under review).

When an online model predictive controller (MPC) for residential battery storage falls
short of what a perfect-foresight optimizer achieves, how much of the shortfall is due to
the receding-horizon controller and how much to forecast error? This repository measures
that on the CityLearn 2022 Challenge dataset (17 California homes with rooftop PV and a
6.4 kWh battery) with a fully open-source stack: CVXPY + CLARABEL for the optimization,
LightGBM for forecasting.

## Main result

Weighted score over the three evaluation phases (lower is better; 1.0 = no control).

| Controller | Phase 1 | Phase 2 | Phase 3 | Weighted |
|---|---|---|---|---|
| Full-horizon LP, perfect foresight | 0.650 | 0.703 | 0.647 | **0.664** |
| MPC + perfect forecast | 0.684 | 0.734 | 0.695 | 0.705 |
| MPC + LightGBM + Persistence average | 0.834 | 0.879 | 0.799 | **0.830** |
| MPC + statistical ensemble (Persistence + Weekly + Holt-Winters) | 0.841 | 0.892 | 0.807 | 0.839 |
| MPC + LightGBM | 0.848 | 0.884 | 0.810 | 0.840 |
| MPC + Persistence | 0.852 | 0.905 | 0.815 | 0.850 |
| MPC + Holt-Winters | 0.860 | 0.898 | 0.822 | 0.853 |
| MPC + Weekly seasonality | 0.870 | 0.927 | 0.829 | 0.866 |
| No control | 1.000 | 1.000 | 1.000 | 1.000 |

The gap between the perfect-foresight reference (0.664) and the best practical controller
(0.830) splits into a receding-horizon component of 0.040 (about one quarter) and a
forecasting component of 0.125 (about three quarters).

Further findings, each backed by a script and a result file in this repository:

* Beyond two regularizers that are part of the tuned baseline (action smoothing and
  scenario noise), none of twenty further modifications of the MPC formulation or the
  forecasting pipeline improves the score. Averaging the LightGBM and Persistence
  forecasts does (0.840 to 0.830, consistent over five seeds).
* Measured on the same data as the score, Persistence is more accurate than LightGBM on
  every accuracy metric tried (MAPE, MAE, peak-hour error, executed-horizon error) and still
  controls worse. Forecasters should be selected in closed loop, not by validation error.
* The 24 h-ahead weather channels are worth 0.014 of score; without them LightGBM is no
  better than Persistence.
* The whole pipeline runs on an Orange Pi 5 Pro (RK3588S): 3.5 s per five-building MPC
  solve, 60 ms per LightGBM forecast, under 0.3 GB resident memory.

## Repository layout

```
mpcgap/                         library code
  environment.py                CityLearn wrapper: environment creation and scoring (v2 and 2022 metrics)
  evaluate_full.py              three-phase evaluation protocol and no-control baseline
  perfect_foresight_lp.py       full-horizon LP with perfect foresight (CVXPY / CLARABEL)
  online_mpc.py                 rolling-horizon stochastic MPC with pluggable forecasters
  forecasters.py                Perfect, Persistence, Holt-Winters, Weekly, Ensemble, Hybrid, HorizonBlended
  lgb_feature_engineering.py    449-feature builder (offline and online variants)
  lgb_train.py                  LightGBM training (published hyperparameters or optional Optuna search)
  lgb_forecaster.py             LightGBM forecaster with online OLS bias correction
  lgb_evaluate.py               forecast accuracy by horizon, MPC evaluation, feature ablation
  data.py                       raw dataset access and forecaster factories
experiments/                    one script per experiment in the paper (see docs/REPRODUCING.md)
  run_verify_mpc.py             three-phase MPC evaluation of one configuration
  run_verify_multiseed.py       the same over seeds 2023-2026
  run_*_sensitivity.py          scenario-noise and scenario-count sweeps
  run_peak_metrics.py, ...      forecast accuracy on all phases, peak hours, executed horizons
  run_train_*.py                retraining for the temporal-overlap and weather ablations
  run_bench_edge.py             single-core timing benchmark
  dfl_*.py                      decision-focused learning experiments
  summarize_*.py                tables and seed statistics from results/
  run_all.py                    full suite in one go
results/                        raw JSON/text outputs behind every reported number
figures/                        figure scripts and generated figures
models/                         trained models (downloaded from the release, see models/README.md)
docs/REPRODUCING.md             table-by-table reproduction guide with runtimes
```

## Installation

```bash
git clone https://github.com/dvoitekh/mpc-gap-citylearn.git
cd mpc-gap-citylearn
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add requirements-dfl.txt for the DFL experiments
gh release download v1.0.0 --dir models  # trained models (or retrain, see models/README.md)
```

All reported results were produced with Python 3.12.8, CityLearn 2.2b0, CVXPY 1.8.1,
CLARABEL 0.11.1, LightGBM 4.6.0, scikit-learn 1.7.1 and NumPy 2.3.2 on an Apple M1 Pro.
The dataset (`citylearn_challenge_2022_phase_all`) is fetched by the CityLearn package on
first use.

## Quick start

```bash
python -m mpcgap.evaluate_full                       # no-control baseline, ~2 min
python -m mpcgap.perfect_foresight_lp                # LP reference, ~6 min
python -m experiments.run_verify_mpc lgb             # MPC + LightGBM on all phases, ~75 min
python -m experiments.run_verify_mpc lgb_avg         # MPC + LightGBM/Persistence average
python -m experiments.summarize_results              # tables from results/*.json
python -m experiments.summarize_multiseed            # seed statistics and paired tests
```

`docs/REPRODUCING.md` maps every table and figure in the paper to a command, an output
file and a runtime.

## Evaluation protocol

The official 2022 competition scored hidden building districts. This repository emulates
that structure with time-based splits of the public full-year dataset:

| Phase | Buildings | Days | Weight |
|---|---|---|---|
| 1 | 1-5 | 0-120 | 0.2 |
| 2 | 1-5 | 120-240 | 0.3 |
| 3 | 1-17 except 12 and 15 | 240-365 | 0.5 |

Buildings 12 and 15 are excluded for data quality (a frozen load meter and a near-dead
PV series). Scores use the CityLearn v2 evaluation, `score = (cost + emissions + grid) / 3`
with `grid` the mean of ramping, daily load factor, daily peak and all-time peak; the
original 2022 competition metric (ramping and monthly load factor only) is recorded
alongside as `official_score` in every result file. Scores under the two definitions are
not interchangeable.

## Acknowledgements

The LP formulation and the LightGBM hyperparameters follow the CityLearn 2022 winning
solution by Team Together (https://github.com/Tobi-Tob/CityLearn2022), re-implemented
here on an open-source solver stack. The simulation environment and dataset are from
CityLearn (https://github.com/intelligent-environments-lab/CityLearn).

## License and citation

MIT License. If you use this code or the results, please cite the paper (see
`CITATION.cff`).
