"""
Sparse-lag ablation: retrain LightGBM keeping only four lags per signal
(h-1, h-24, h-48, h-168, i.e. *_past_0/23/47/167) plus the rolling statistics,
calendar, weather and building features; everything else identical to the
headline model (published Team Together hyperparameters).

Because LGBForecaster builds its input from the stored 'feature_columns'
list, the trained model is drop-in usable with no forecaster changes.

Usage (from the repository root):
    python -m experiments.run_train_sparse

Output: models/lgb_models_sparse.pkl
"""
import time

import joblib

from mpcgap.lgb_train import build_training_data, train_single_model, get_feature_importance
from mpcgap.lgb_feature_engineering import get_feature_columns, N_LAGS

SPARSE_LAGS = {0, 23, 47, 167}  # past_i = value at t - i - 1  ->  h-1, h-24, h-48, h-168


def main():
    t0 = time.time()
    ref = joblib.load('models/lgb_models.pkl')
    load_params, solar_params = ref['load_params'], ref['solar_params']

    drop = {f'{p}_past_{i}' for p in ('load', 'solar')
            for i in range(N_LAGS) if i not in SPARSE_LAGS}
    keep = [c for c in get_feature_columns() if c not in drop]
    print(f"Dropping {len(drop)} lag features; {len(keep)} features remain")

    X, y_load, y_solar, days = build_training_data()
    X = X[keep]

    print("Training load model...")
    load_model = train_single_model(X, y_load, load_params)
    print("Training solar model...")
    solar_model = train_single_model(X, y_solar, solar_params)

    joblib.dump({
        'load_model': load_model, 'solar_model': solar_model,
        'load_params': load_params, 'solar_params': solar_params,
        'load_cv_mape': -1.0, 'solar_cv_mape': -1.0,
        'feature_columns': keep,
        'load_importance': get_feature_importance(load_model, keep),
        'solar_importance': get_feature_importance(solar_model, keep),
        'note': 'four sparse lags per signal (h-1, h-24, h-48, h-168) instead of 168',
    }, 'models/lgb_models_sparse.pkl')
    print(f"Saved models/lgb_models_sparse.pkl ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
