"""
Reproducibility check for the days-0-238 model: train from scratch on days
0-238 with the hyperparameters published by the CityLearn 2022 winning
solution, without reading any existing model file. The result should match
models/lgb_models_days240.pkl (and its MPC score) exactly, which confirms that
the released models carry those published hyperparameters and nothing else.

Usage (from the repository root):
    python -m experiments.run_train_days240_tt

Output: models/lgb_models_days240_tt.pkl
"""
import time

import joblib

from lgb_train import (build_training_data, train_single_model,
                       get_feature_importance, TT_LOAD_PARAMS, TT_SOLAR_PARAMS)
from lgb_feature_engineering import get_feature_columns

MAX_DAY = 238


def main():
    t0 = time.time()
    print(f"TT published params (no HPO on our data):\n  load={TT_LOAD_PARAMS}\n  solar={TT_SOLAR_PARAMS}")

    X, y_load, y_solar, days = build_training_data()
    mask = days <= MAX_DAY
    X, y_load, y_solar = X[mask], y_load[mask], y_solar[mask]
    print(f"Training on days <= {MAX_DAY}: {len(X)} samples")

    print("Training load model...")
    load_model = train_single_model(X, y_load, TT_LOAD_PARAMS)
    print("Training solar model...")
    solar_model = train_single_model(X, y_solar, TT_SOLAR_PARAMS)

    names = get_feature_columns()
    joblib.dump({
        'load_model': load_model, 'solar_model': solar_model,
        'load_params': TT_LOAD_PARAMS, 'solar_params': TT_SOLAR_PARAMS,
        'load_cv_mape': -1.0, 'solar_cv_mape': -1.0,
        'feature_columns': names,
        'load_importance': get_feature_importance(load_model, names),
        'solar_importance': get_feature_importance(solar_model, names),
        'note': f'days 0-{MAX_DAY} only, Team Together hyperparameters, no HPO on our data',
    }, 'models/lgb_models_days240_tt.pkl')
    print(f"Saved models/lgb_models_days240_tt.pkl ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
