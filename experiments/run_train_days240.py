"""
Temporal-overlap check: retrain LightGBM on days 0-238 only (all 17 buildings),
so that no training target falls inside the Phase 3 evaluation window
(days 240-365). Hyperparameters are read from the full model
(models/lgb_models.pkl) so that only the training data changes.

A sample at hour index h has targets h+1..h+24, so requiring day <= 238
(h <= 5735) keeps every target hour strictly before day 240 (hour 5760).

Usage (from the repository root):
    python -m experiments.run_train_days240

Output: models/lgb_models_days240.pkl
"""
import time

import joblib
import numpy as np

from lgb_train import build_training_data, train_single_model, get_feature_importance
from lgb_feature_engineering import get_feature_columns

MAX_DAY = 238


def main():
    t0 = time.time()
    ref = joblib.load('models/lgb_models.pkl')
    load_params, solar_params = ref['load_params'], ref['solar_params']
    print(f"Reusing tuned params: load={load_params}\n solar={solar_params}")

    X, y_load, y_solar, days = build_training_data()
    mask = days <= MAX_DAY
    X, y_load, y_solar = X[mask], y_load[mask], y_solar[mask]
    print(f"Filtered to days <= {MAX_DAY}: {len(X)} samples "
          f"(of {len(mask)}; {100 * mask.mean():.0f}%)")

    print("Training load model...")
    load_model = train_single_model(X, y_load, load_params)
    print("Training solar model...")
    solar_model = train_single_model(X, y_solar, solar_params)

    feature_names = get_feature_columns()
    joblib.dump({
        'load_model': load_model,
        'solar_model': solar_model,
        'load_params': load_params,
        'solar_params': solar_params,
        'load_cv_mape': -1.0,
        'solar_cv_mape': -1.0,
        'feature_columns': feature_names,
        'load_importance': get_feature_importance(load_model, feature_names),
        'solar_importance': get_feature_importance(solar_model, feature_names),
        'note': f'trained on days 0-{MAX_DAY} only (temporal-leakage ablation)',
    }, 'models/lgb_models_days240.pkl')
    print(f"Saved models/lgb_models_days240.pkl ({(time.time() - t0) / 60:.1f} min)")


if __name__ == '__main__':
    main()
