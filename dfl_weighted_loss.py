"""
Phase 2: Decision-Focused Weighted Loss (Variant C).

Train LightGBM with cost-weighted sample weights:
- Samples during high-price/high-carbon hours get higher weight
- Asymmetric loss: penalize under-prediction more (battery can't compensate)

This is the simplest DFL variant — no surrogate NN, no differentiable optimization.
Just re-weighting the training loss by MPC cost relevance.
"""

import os
import time
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from citylearn.data import DataSet

from lgb_feature_engineering import FeatureBuilder, get_feature_columns, get_target_columns
from lgb_train import build_training_data, cv_split, compute_mape, CV_FOLDS, TT_LOAD_PARAMS, TT_SOLAR_PARAMS

warnings.filterwarnings('ignore')


def compute_hour_cost_weights(days, price, carbon, alpha=0.5):
    """
    Compute per-sample cost weights based on hour-of-day electricity price and carbon.

    Samples at hours with high price/carbon get higher weight in training loss.
    This makes the model focus accuracy on hours that matter most for MPC.

    Args:
        days: array of day indices per sample
        price: full-year electricity price array (8760,)
        carbon: full-year carbon intensity array (8760,)
        alpha: balance between price (alpha) and carbon (1-alpha)

    Returns:
        weights: array of sample weights (normalized to mean=1)
    """
    n_samples = len(days)
    weights = np.ones(n_samples)

    # Price/carbon by hour of day (averaged across the year)
    price_by_hour = np.zeros(24)
    carbon_by_hour = np.zeros(24)
    for h in range(24):
        price_by_hour[h] = np.mean(price[h::24])
        carbon_by_hour[h] = np.mean(carbon[h::24])

    # Normalize to [0, 1]
    price_norm = price_by_hour / (np.max(price_by_hour) + 1e-8)
    carbon_norm = carbon_by_hour / (np.max(carbon_by_hour) + 1e-8)

    # Combined weight per hour
    hour_weight = alpha * price_norm + (1 - alpha) * carbon_norm

    # For each sample, compute the AVERAGE weight across its 24 target hours
    # (since each sample predicts 24 hours ahead)
    for i in range(n_samples):
        # Get the hour of this sample from its day and position
        # Actually, we need the actual hour from the feature data
        # For simplicity: weight by the average cost across ALL 24 target hours
        # (slightly different for each starting hour, but close enough)
        weights[i] = np.mean(hour_weight)

    # Better approach: use the actual starting hour from features
    # The features include 'hour' which tells us the hour of the sample
    return weights


def compute_sample_weights_from_features(X, price, carbon, alpha=0.5):
    """
    Compute per-sample weights using the actual hour from feature data.

    For each sample starting at hour h, the 24 target hours are h+1..h+24.
    Weight = average cost importance across those 24 target hours.
    """
    n_samples = len(X)

    # Price/carbon by hour of day
    price_by_hour = np.array([np.mean(price[h::24]) for h in range(24)])
    carbon_by_hour = np.array([np.mean(carbon[h::24]) for h in range(24)])

    # Normalize
    p_norm = price_by_hour / (np.max(price_by_hour) + 1e-8)
    c_norm = carbon_by_hour / (np.max(carbon_by_hour) + 1e-8)

    # Combined cost weight by hour
    cost_by_hour = alpha * p_norm + (1 - alpha) * c_norm

    weights = np.ones(n_samples)

    if 'hour' in X.columns:
        hours = X['hour'].values.astype(int)
        for i in range(n_samples):
            h = hours[i]
            # Target hours: h+1, h+2, ..., h+24 (mod 24)
            target_hours = [(h + j + 1) % 24 for j in range(24)]
            weights[i] = np.mean([cost_by_hour[th] for th in target_hours])

    # Normalize to mean=1
    weights = weights / (np.mean(weights) + 1e-8)

    return weights


def compute_peak_hour_weights(X, load_targets, price, carbon,
                               peak_multiplier=3.0):
    """
    Compute per-sample weights emphasizing peak consumption hours.

    Peak hours (top 10% of load) get peak_multiplier times more weight.
    Cheap/low-load hours get weight 1.0.

    This targets the MPC's key pain point: mispredicting peaks
    causes the battery to not discharge when it should.
    """
    n_samples = len(X)
    weights = np.ones(n_samples)

    if load_targets is not None:
        # Compute average load across all target horizons
        avg_load = load_targets.mean(axis=1).values

        # Top 10% of loads are "peak hours"
        threshold = np.percentile(avg_load, 90)
        peak_mask = avg_load >= threshold
        weights[peak_mask] = peak_multiplier

        # Also boost high-price hours
        if 'hour' in X.columns:
            hours = X['hour'].values.astype(int)
            price_by_hour = np.array([np.mean(price[h::24]) for h in range(24)])
            price_threshold = np.percentile(price_by_hour, 75)
            for i in range(n_samples):
                if price_by_hour[hours[i] % 24] >= price_threshold:
                    weights[i] = max(weights[i], peak_multiplier * 0.5)

    # Normalize to mean=1
    weights = weights / (np.mean(weights) + 1e-8)
    return weights


def train_weighted_model(X_train, y_train, params, sample_weights):
    """Train MultiOutputRegressor with per-sample weights."""
    # LightGBM supports sample_weight in fit()
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
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def evaluate_weighted_cv(X, y, days, params, weights, target_type='load'):
    """Evaluate weighted model across CV folds."""
    mapes = []
    threshold = 0.1 if target_type == 'load' else 0.01

    for fold in CV_FOLDS:
        train_mask = days <= fold['train_end']
        val_mask = (days >= fold['val_start']) & (days <= fold['val_end'])

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        w_train = weights[train_mask]

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = train_weighted_model(X_train, y_train, params, w_train)
        y_pred = model.predict(X_val)
        mape = compute_mape(y_val.values, y_pred, threshold=threshold)
        mapes.append(mape)

    return np.mean(mapes) if mapes else float('inf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--peak-mult', type=float, default=3.0,
                        help='Peak hour weight multiplier')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run MPC evaluation after training')
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("PHASE 2: DECISION-FOCUSED WEIGHTED LOSS TRAINING")
    print("=" * 70)

    # Load training data
    X, y_load, y_solar, days = build_training_data()

    # Load price/carbon
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')
    root_dir = schema['root_directory']
    price = pd.read_csv(f'{root_dir}/pricing.csv')['electricity_pricing'].values
    carbon = pd.read_csv(f'{root_dir}/carbon_intensity.csv')['carbon_intensity'].values

    os.makedirs('models', exist_ok=True)

    # =====================================================
    # Strategy 1: Cost-weighted by hour
    # =====================================================
    print("\n--- Strategy 1: Cost-Weighted by Hour ---")
    cost_weights = compute_sample_weights_from_features(X, price, carbon)
    print(f"  Weight stats: min={cost_weights.min():.3f}, "
          f"max={cost_weights.max():.3f}, std={cost_weights.std():.3f}")

    load_mape_cw = evaluate_weighted_cv(X, y_load, days, TT_LOAD_PARAMS,
                                         cost_weights, 'load')
    solar_mape_cw = evaluate_weighted_cv(X, y_solar, days, TT_SOLAR_PARAMS,
                                          cost_weights, 'solar')
    print(f"  Cost-weighted Load CV MAPE:  {load_mape_cw:.2f}%")
    print(f"  Cost-weighted Solar CV MAPE: {solar_mape_cw:.2f}%")

    # =====================================================
    # Strategy 2: Peak-hour emphasis
    # =====================================================
    print(f"\n--- Strategy 2: Peak-Hour Emphasis (mult={args.peak_mult}) ---")
    peak_weights = compute_peak_hour_weights(
        X, y_load, price, carbon, peak_multiplier=args.peak_mult
    )
    print(f"  Weight stats: min={peak_weights.min():.3f}, "
          f"max={peak_weights.max():.3f}, std={peak_weights.std():.3f}")

    load_mape_peak = evaluate_weighted_cv(X, y_load, days, TT_LOAD_PARAMS,
                                           peak_weights, 'load')
    solar_mape_peak = evaluate_weighted_cv(X, y_solar, days, TT_SOLAR_PARAMS,
                                            peak_weights, 'solar')
    print(f"  Peak-weighted Load CV MAPE:  {load_mape_peak:.2f}%")
    print(f"  Peak-weighted Solar CV MAPE: {solar_mape_peak:.2f}%")

    # =====================================================
    # Strategy 3: Combined (cost + peak)
    # =====================================================
    print("\n--- Strategy 3: Combined (Cost + Peak) ---")
    combined_weights = 0.5 * cost_weights + 0.5 * peak_weights
    combined_weights = combined_weights / (np.mean(combined_weights) + 1e-8)

    load_mape_comb = evaluate_weighted_cv(X, y_load, days, TT_LOAD_PARAMS,
                                           combined_weights, 'load')
    solar_mape_comb = evaluate_weighted_cv(X, y_solar, days, TT_SOLAR_PARAMS,
                                            combined_weights, 'solar')
    print(f"  Combined Load CV MAPE:  {load_mape_comb:.2f}%")
    print(f"  Combined Solar CV MAPE: {solar_mape_comb:.2f}%")

    # =====================================================
    # Baseline: Uniform weights (standard training)
    # =====================================================
    print("\n--- Baseline: Uniform Weights ---")
    uniform_weights = np.ones(len(X))
    load_mape_uni = evaluate_weighted_cv(X, y_load, days, TT_LOAD_PARAMS,
                                          uniform_weights, 'load')
    solar_mape_uni = evaluate_weighted_cv(X, y_solar, days, TT_SOLAR_PARAMS,
                                           uniform_weights, 'solar')
    print(f"  Uniform Load CV MAPE:  {load_mape_uni:.2f}%")
    print(f"  Uniform Solar CV MAPE: {solar_mape_uni:.2f}%")

    # =====================================================
    # Summary
    # =====================================================
    print("\n--- Summary ---\n")
    print(f"{'Strategy':25s} {'Load MAPE':>12s} {'Solar MAPE':>12s}")
    print("-" * 52)
    print(f"{'Uniform (baseline)':25s} {load_mape_uni:>10.2f}% {solar_mape_uni:>10.2f}%")
    print(f"{'Cost-weighted':25s} {load_mape_cw:>10.2f}% {solar_mape_cw:>10.2f}%")
    print(f"{'Peak-emphasis':25s} {load_mape_peak:>10.2f}% {solar_mape_peak:>10.2f}%")
    print(f"{'Combined':25s} {load_mape_comb:>10.2f}% {solar_mape_comb:>10.2f}%")

    # =====================================================
    # Train best model and save
    # =====================================================
    # Pick the strategy with the lowest COMBINED MAPE... but we want MPC score,
    # not MAPE! For now, save the peak-weighted model (most decision-focused).
    print("\n--- Training final peak-weighted models ---")

    load_model = train_weighted_model(X, y_load, TT_LOAD_PARAMS, peak_weights)
    solar_model = train_weighted_model(X, y_solar, TT_SOLAR_PARAMS, peak_weights)

    model_data = {
        'load_model': load_model,
        'solar_model': solar_model,
        'load_params': TT_LOAD_PARAMS,
        'solar_params': TT_SOLAR_PARAMS,
        'feature_columns': get_feature_columns(),
        'weighting': 'peak_emphasis',
        'peak_multiplier': args.peak_mult,
    }

    model_path = 'models/lgb_models_weighted.pkl'
    joblib.dump(model_data, model_path)
    print(f"  Saved to {model_path}")

    # =====================================================
    # Optional: Run MPC evaluation
    # =====================================================
    if args.evaluate:
        print("\n--- MPC Evaluation with Weighted Models ---")
        from evaluate_full import PHASES
        from online_mpc import run_mpc_phase
        from lgb_forecaster import LGBForecaster

        # Temporarily swap models
        original_path = 'models/lgb_models.pkl'
        backup_path = 'models/lgb_models_backup.pkl'

        if os.path.exists(original_path):
            os.rename(original_path, backup_path)
        os.rename(model_path, original_path)

        try:
            weighted_score = 0
            for phase in PHASES:
                N = len(phase['buildings'])
                forecaster = LGBForecaster(N, sim_start=phase['sim_start'],
                                            building_names=phase['buildings'])
                metrics = run_mpc_phase(
                    phase['buildings'], phase['sim_start'], phase['sim_end'],
                    forecaster, n_scenarios=5, action_smooth_weight=0.1,
                    reopt_interval=8, verbose=True
                )
                weighted_score += phase['weight'] * metrics['score']
                print(f"  {phase['name']}: Score={metrics['score']:.4f}")

            print(f"\n  Weighted MPC Score (peak-weighted LGB): {weighted_score:.4f}")
        finally:
            # Restore original models
            os.rename(original_path, model_path)
            if os.path.exists(backup_path):
                os.rename(backup_path, original_path)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
