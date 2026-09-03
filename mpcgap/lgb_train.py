"""
LightGBM training pipeline.

Training strategy:
- 449 engineered features (calendar, weather, 168-hour load/solar lags,
  rolling statistics, building metadata), see lgb_feature_engineering.py
- Direct multi-output regression: 24 estimators per target (load, solar)
- Optional Optuna hyperparameter search with expanding-window CV (2 folds)
- Final models saved to the models/ directory

All released models were trained with --quick, i.e. with the hyperparameters
published by the CityLearn 2022 winning solution and no search.

Usage:
    python -m mpcgap.lgb_train --quick          # published hyperparameters, no search (~13 min)
    python -m mpcgap.lgb_train                  # Optuna search (100 trials) + training
    python -m mpcgap.lgb_train --n-trials 20    # fewer Optuna trials
    python -m mpcgap.lgb_train --quick --exclude-phase3 --output models/lgb_models_no_p3.pkl
"""

import os
import sys
import time
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error

from mpcgap.lgb_feature_engineering import FeatureBuilder, get_feature_columns, get_target_columns

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Time-series expanding window CV folds (2 folds for speed)
CV_FOLDS = [
    {'train_end': 240, 'val_start': 241, 'val_end': 300},
    {'train_end': 300, 'val_start': 301, 'val_end': 365},
]

# Team Together's known good hyperparameters (warm start)
TT_LOAD_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'max_depth': -1,
}

TT_SOLAR_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'max_depth': -1,
}


def build_training_data(exclude_phase3=False):
    """Build full training dataset with day tracking.

    Args:
        exclude_phase3: if True, exclude Phase 3 buildings (6-17 except 12,15)
                        to test data leakage impact.
    """
    print("Building training dataset...")
    fb = FeatureBuilder()

    # Phase 3 buildings to exclude if requested
    p3_buildings = {f'Building_{i}' for i in range(1, 18) if i not in [12, 15] and i > 5}

    feature_cols = get_feature_columns()
    load_target_cols = get_target_columns('load')
    solar_target_cols = get_target_columns('solar')

    all_X = []
    all_y_load = []
    all_y_solar = []
    all_days = []

    for b_idx, b_name in enumerate(fb.all_buildings):
        if exclude_phase3 and b_name in p3_buildings:
            print(f"  Skipping {b_name} (Phase 3)")
            continue

        df = fb._build_one_building(b_idx, b_name)
        if df is None:
            continue

        all_X.append(df[feature_cols])
        all_y_load.append(df[load_target_cols])
        all_y_solar.append(df[solar_target_cols])

        # After dropna, the DataFrame index holds the original row indices.
        # Each original row index = hour within the year, so day = idx // 24.
        days = df.index.values // 24
        all_days.append(days)

    X = pd.concat(all_X, ignore_index=True)
    y_load = pd.concat(all_y_load, ignore_index=True)
    y_solar = pd.concat(all_y_solar, ignore_index=True)
    days = np.concatenate(all_days)

    print(f"  Total samples: {len(X)}, features: {X.shape[1]}")
    print(f"  Day range: {days.min()} - {days.max()}")

    return X, y_load, y_solar, days


def cv_split(X, y, days, fold):
    """Split data for one CV fold."""
    train_mask = days <= fold['train_end']
    val_mask = (days >= fold['val_start']) & (days <= fold['val_end'])

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    return X_train, y_train, X_val, y_val


def compute_mape(y_true, y_pred, threshold=0.1):
    """Compute MAPE, ignoring values below threshold."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100


def train_single_model(X_train, y_train, params):
    """Train a MultiOutputRegressor with LightGBM."""
    model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            objective='regression',
            random_state=2022,
            verbose=-1,
            n_jobs=1,
            **params
        ),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_cv(X, y, days, params, target_type='load'):
    """Evaluate model across all CV folds, return mean MAPE."""
    mapes = []
    threshold = 0.1 if target_type == 'load' else 0.01

    for fold in CV_FOLDS:
        X_train, y_train, X_val, y_val = cv_split(X, y, days, fold)

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = train_single_model(X_train, y_train, params)
        y_pred = model.predict(X_val)
        mape = compute_mape(y_val.values, y_pred, threshold=threshold)
        mapes.append(mape)

    return np.mean(mapes) if mapes else float('inf')


def create_optuna_objective(X, y, days, target_type, warm_start_params):
    """Create Optuna objective function."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
        }

        # Use pruning: evaluate on first 2 folds, then all 4
        mapes = []
        threshold = 0.1 if target_type == 'load' else 0.01

        for fold_idx, fold in enumerate(CV_FOLDS):
            X_train, y_train, X_val, y_val = cv_split(X, y, days, fold)
            if len(X_train) == 0 or len(X_val) == 0:
                continue

            model = train_single_model(X_train, y_train, params)
            y_pred = model.predict(X_val)
            mape = compute_mape(y_val.values, y_pred, threshold=threshold)
            mapes.append(mape)

            # Report intermediate value for pruning
            trial.report(np.mean(mapes), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(mapes) if mapes else float('inf')

    return objective


def run_optuna_hpo(X, y, days, target_type, n_trials=100):
    """Run Optuna HPO for one model type."""
    warm_start = TT_LOAD_PARAMS if target_type == 'load' else TT_SOLAR_PARAMS

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=2022),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    # Enqueue warm-start trial
    study.enqueue_trial(warm_start)

    objective = create_optuna_objective(X, y, days, target_type, warm_start)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best {target_type} MAPE: {study.best_value:.2f}%")
    print(f"  Best params: {study.best_params}")

    return study.best_params, study.best_value


def get_feature_importance(model, feature_names, top_k=20):
    """Extract feature importance from MultiOutputRegressor."""
    importances = np.zeros(len(feature_names))
    for estimator in model.estimators_:
        importances += estimator.feature_importances_

    importances /= len(model.estimators_)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False)

    return importance_df.head(top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Skip Optuna, use TT defaults')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--exclude-phase3', action='store_true',
                        help='Exclude Phase 3 buildings from training (data leakage test)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model path (default: models/lgb_models.pkl)')
    args = parser.parse_args()

    start_time = time.time()

    # Build data
    X, y_load, y_solar, days = build_training_data(exclude_phase3=args.exclude_phase3)

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    if args.quick:
        print("\n--- Quick mode: using Team Together defaults ---")
        load_params = TT_LOAD_PARAMS
        solar_params = TT_SOLAR_PARAMS
        load_cv_mape = -1.0  # skip CV in quick mode
        solar_cv_mape = -1.0
    else:
        # Run Optuna HPO
        print(f"\n--- Optuna HPO: {args.n_trials} trials per model ---")

        print("\nOptimizing LOAD model...")
        load_params, load_mape = run_optuna_hpo(X, y_load, days, 'load', args.n_trials)

        print("\nOptimizing SOLAR model...")
        solar_params, solar_mape = run_optuna_hpo(X, y_solar, days, 'solar', args.n_trials)

        # Evaluate best params with full CV
        print("\n--- Evaluating best parameters ---")
        load_cv_mape = evaluate_cv(X, y_load, days, load_params, 'load')
        solar_cv_mape = evaluate_cv(X, y_solar, days, solar_params, 'solar')
        print(f"  Load CV MAPE:  {load_cv_mape:.2f}%")
        print(f"  Solar CV MAPE: {solar_cv_mape:.2f}%")

    # Train final models on ALL data
    print("\n--- Training final models on all data ---")

    print("  Training load model...")
    load_model = train_single_model(X, y_load, load_params)

    print("  Training solar model...")
    solar_model = train_single_model(X, y_solar, solar_params)

    # Feature importance
    print("\n--- Feature Importance (Load) ---")
    feature_names = get_feature_columns()
    load_importance = get_feature_importance(load_model, feature_names)
    print(load_importance.to_string(index=False))

    print("\n--- Feature Importance (Solar) ---")
    solar_importance = get_feature_importance(solar_model, feature_names)
    print(solar_importance.to_string(index=False))

    # Save models and metadata
    model_data = {
        'load_model': load_model,
        'solar_model': solar_model,
        'load_params': load_params,
        'solar_params': solar_params,
        'load_cv_mape': load_cv_mape,
        'solar_cv_mape': solar_cv_mape,
        'feature_columns': feature_names,
        'load_importance': load_importance,
        'solar_importance': solar_importance,
    }

    model_path = args.output if args.output else 'models/lgb_models.pkl'
    joblib.dump(model_data, model_path)
    print(f"\nModels saved to {model_path}")

    # Also save params separately for reference
    params_path = 'models/lgb_params.txt'
    with open(params_path, 'w') as f:
        f.write("LightGBM Best Hyperparameters\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Load model (CV MAPE: {load_cv_mape:.2f}%):\n")
        for k, v in load_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nSolar model (CV MAPE: {solar_cv_mape:.2f}%):\n")
        for k, v in solar_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTop-20 Load features:\n")
        f.write(load_importance.to_string(index=False))
        f.write(f"\n\nTop-20 Solar features:\n")
        f.write(solar_importance.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
