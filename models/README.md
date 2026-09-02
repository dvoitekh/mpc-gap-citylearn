# Trained models

The LightGBM forecasters are not tracked in git (about 70 MB each). Download them
from the GitHub release and place them in this directory, or retrain them.

| File | Training data | Used for |
|---|---|---|
| `lgb_models.pkl` | all 17 buildings, all 365 days | every headline LightGBM result |
| `lgb_models_days240.pkl` | all 17 buildings, days 0-238 | temporal-overlap check |
| `lgb_models_no_p3.pkl` | buildings 1-5, 12, 15 (Phase 3-only buildings removed), all days | building-overlap check |
| `lgb_models_noweather.pkl` | as `lgb_models.pkl`, all 100 weather columns removed | weather-feature ablation |
| `nn_forecaster_mse.pt`, `nn_forecaster_dfl.pt`, `surrogate.pt`, `surrogate_data.npz` | see `dfl_surrogate.py` | decision-focused learning experiment |

Download:

```bash
gh release download v1.0.0 --repo dvoitekh/mpc-gap-citylearn --dir models
```

Retrain (about 13 minutes on an Apple M1 Pro; deterministic, seed 2022):

```bash
python lgb_train.py --quick                                   # -> models/lgb_models.pkl
python lgb_train.py --quick --exclude-phase3 --output models/lgb_models_no_p3.pkl
python -m experiments.run_train_days240                       # -> models/lgb_models_days240.pkl
python -m experiments.run_train_noweather                     # -> models/lgb_models_noweather.pkl
```

All released models use the hyperparameters published with the CityLearn 2022
winning solution (500 estimators, learning rate 0.02 load / 0.05 solar, 31 leaves,
min 20 samples per leaf, row/column subsampling 0.8, L1/L2 = 0.1). No
hyperparameter search was run for the released models; `lgb_train.py` still
contains an optional Optuna search for anyone who wants one.
