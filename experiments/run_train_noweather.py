"""
Weather-feature ablation: retrain LightGBM with every weather column removed
(current weather and the 24 h-ahead weather channels, 100 columns in total),
keeping all other settings identical to the full model.

Because LGBForecaster builds its input from the stored 'feature_columns'
list, the resulting model is drop-in usable with no forecaster changes.

Usage (from the repository root):
    python -m experiments.run_train_noweather

Output: models/lgb_models_noweather.pkl
"""
import time

import joblib

from mpcgap.lgb_train import build_training_data, train_single_model, get_feature_importance
from mpcgap.lgb_feature_engineering import (get_feature_columns, WEATHER_ACTUAL,
                                     _weather_future_cols)


def main():
    t0 = time.time()
    ref = joblib.load('models/lgb_models.pkl')
    load_params, solar_params = ref['load_params'], ref['solar_params']

    weather = set(WEATHER_ACTUAL) | set(_weather_future_cols())
    keep = [c for c in get_feature_columns() if c not in weather]
    print(f"Dropping {len(weather)} weather features; {len(keep)} features remain")

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
        'note': 'all weather features removed (ablation for Recommendation 3)',
    }, 'models/lgb_models_noweather.pkl')
    print(f"Saved models/lgb_models_noweather.pkl ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
