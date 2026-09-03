"""
LightGBM-based forecaster for Online MPC.

Implements the BaseForecaster interface using pre-trained LightGBM models.
Features are built online via OnlineFeatureBuilder.
Includes Team Together-style OLS online correction.

Usage:
    forecaster = LGBForecaster(n_buildings=5, sim_start=0,
                                building_names=['Building_1', ..., 'Building_5'])
    # During simulation:
    forecaster.update(b, load, solar)
    load_pred, solar_pred = forecaster.predict(b, step, hour)
"""

import numpy as np
import pandas as pd
import joblib

from mpcgap.forecasters import BaseForecaster
from mpcgap.lgb_feature_engineering import OnlineFeatureBuilder, get_feature_columns


class LGBForecaster(BaseForecaster):
    """
    LightGBM forecaster with online correction.

    Uses pre-trained MultiOutputRegressor models (24 outputs each) for
    load and solar prediction. Features built from accumulated history
    and pre-loaded weather data via OnlineFeatureBuilder.

    Online correction (Team Together style):
    - Per-building, per-hour OLS correction factor
    - Rolling 90-day window
    - β = Σ(real * pred) / Σ(pred²) — optimal least-squares multiplier
    """

    def __init__(self, n_buildings, sim_start=0, building_names=None,
                 model_path='models/lgb_models.pkl',
                 online_correction=True, correction_window=90,
                 correct_solar=False):
        """
        Args:
            n_buildings: int
            sim_start: int, global start timestep for this phase
            building_names: list of building name strings
            model_path: path to saved model data
            online_correction: bool, enable OLS correction
            correction_window: int, days for rolling OLS window (default 90)
            correct_solar: bool, also apply OLS correction to solar (default False)
        """
        self.n_buildings = n_buildings
        self.sim_start = sim_start
        self.online_correction = online_correction
        self.correction_window = correction_window
        self.correct_solar = correct_solar

        # Load models
        model_data = joblib.load(model_path)
        self.load_model = model_data['load_model']
        self.solar_model = model_data['solar_model']
        self.feature_columns = model_data['feature_columns']

        # Feature builder
        self.feature_builder = OnlineFeatureBuilder(
            n_buildings, sim_start, building_names
        )

        # Online correction state
        self._init_correction()

        # Track predictions for correction
        self.last_load_pred = {i: None for i in range(n_buildings)}
        self.last_solar_pred = {i: None for i in range(n_buildings)}
        self.step_of_last_pred = {i: -1 for i in range(n_buildings)}

    def _init_correction(self):
        """Initialize online correction state."""
        N = self.n_buildings
        self.pred_load_correction = {i: np.ones(24) for i in range(N)}
        self.pred_solar_correction = {i: np.ones(24) for i in range(N)}

        # Accumulators for OLS: β = Σ(real * pred) / Σ(pred²)
        self.load_pred_real_sum = {i: np.zeros(24) for i in range(N)}
        self.load_pred_sq_sum = {i: np.zeros(24) for i in range(N)}
        self.solar_pred_real_sum = {i: np.zeros(24) for i in range(N)}
        self.solar_pred_sq_sum = {i: np.zeros(24) for i in range(N)}

        # Rolling window (90 days) FIFO lists
        self.load_pred_real_list = {i: [] for i in range(N)}
        self.load_pred_sq_list = {i: [] for i in range(N)}
        self.solar_pred_real_list = {i: [] for i in range(N)}
        self.solar_pred_sq_list = {i: [] for i in range(N)}

        self.correction_count = {i: 0 for i in range(N)}

    def predict(self, building_idx, current_step, current_hour):
        """
        Predict load and solar for next 24 hours.

        Returns:
            load_pred: array (24,) in kW
            solar_pred: array (24,) in kW
        """
        # Build feature dict and convert to numpy array (faster than DataFrame)
        feat_dict = self.feature_builder.build_features(
            building_idx, current_step, current_hour
        )
        feat_array = np.array(
            [[feat_dict[c] for c in self.feature_columns]]
        )

        # Predict using individual estimators directly (avoids DataFrame overhead)
        load_pred = np.array([
            est.predict(feat_array)[0] for est in self.load_model.estimators_
        ])
        solar_pred = np.array([
            est.predict(feat_array)[0] for est in self.solar_model.estimators_
        ])

        # Clip to non-negative
        load_pred = np.maximum(load_pred, 0.0)
        solar_pred = np.maximum(solar_pred, 0.0)

        # Night solar hard constraint: zero solar for night hours
        for h in range(24):
            target_hour = (current_hour + h + 1) % 24
            if target_hour < 7 or target_hour >= 20:
                solar_pred[h] = 0.0

        # Store raw prediction for correction computation
        self.last_load_pred[building_idx] = load_pred.copy()
        self.last_solar_pred[building_idx] = solar_pred.copy()
        self.step_of_last_pred[building_idx] = current_step

        # Apply online correction
        if self.online_correction and self.correction_count[building_idx] > self.correction_window:
            corr_load = self.pred_load_correction[building_idx]

            # Shift correction to align with prediction hour
            shift = current_hour
            if shift > 0:
                corr_load = np.concatenate([corr_load[shift:], corr_load[:shift]])
            load_pred = load_pred * corr_load

            if self.correct_solar:
                corr_solar = self.pred_solar_correction[building_idx]
                if shift > 0:
                    corr_solar = np.concatenate([corr_solar[shift:], corr_solar[:shift]])
                solar_pred = solar_pred * corr_solar

        return np.maximum(load_pred, 0.0), np.maximum(solar_pred, 0.0)

    def update(self, building_idx, load, solar, observation=None):
        """
        Update forecaster with new observation.

        Args:
            building_idx: int
            load: float, observed load in kW
            solar: float, observed solar in kW
            observation: array, full CityLearn observation (unused here)
        """
        self.feature_builder.update(building_idx, load, solar)

        # Update online correction every 24 hours (when we have a full day of actuals)
        if not self.online_correction:
            return

        hist = self.feature_builder.load_hist[building_idx]
        n = len(hist)
        W = self.correction_window

        # Every 24 hours, update correction factors
        if n >= 24 and n % 24 == 0 and self.last_load_pred[building_idx] is not None:
            # Get last 24 hours of actuals
            real_load = np.array(hist[-24:])
            real_solar = np.array(self.feature_builder.solar_hist[building_idx][-24:])

            pred_load = self.last_load_pred[building_idx]
            pred_solar = self.last_solar_pred[building_idx] if self.last_solar_pred[building_idx] is not None else np.zeros(24)

            # Compute OLS terms for load and solar
            load_pr = real_load * pred_load
            load_p2 = pred_load ** 2
            solar_pr = real_solar * pred_solar
            solar_p2 = pred_solar ** 2

            b = building_idx
            self.correction_count[b] += 1

            if self.correction_count[b] <= W:
                # Accumulate
                self.load_pred_real_list[b].append(load_pr)
                self.load_pred_sq_list[b].append(load_p2)
                self.load_pred_real_sum[b] += load_pr
                self.load_pred_sq_sum[b] += load_p2
                self.solar_pred_real_list[b].append(solar_pr)
                self.solar_pred_sq_list[b].append(solar_p2)
                self.solar_pred_real_sum[b] += solar_pr
                self.solar_pred_sq_sum[b] += solar_p2
            else:
                # Rolling window: add new, remove oldest
                self.load_pred_real_list[b].append(load_pr)
                self.load_pred_sq_list[b].append(load_p2)
                self.load_pred_real_sum[b] += load_pr - self.load_pred_real_list[b].pop(0)
                self.load_pred_sq_sum[b] += load_p2 - self.load_pred_sq_list[b].pop(0)

                self.solar_pred_real_list[b].append(solar_pr)
                self.solar_pred_sq_list[b].append(solar_p2)
                self.solar_pred_real_sum[b] += solar_pr - self.solar_pred_real_list[b].pop(0)
                self.solar_pred_sq_sum[b] += solar_p2 - self.solar_pred_sq_list[b].pop(0)

                # Compute correction: β = Σ(real*pred) / Σ(pred²)
                load_denom = np.where(self.load_pred_sq_sum[b] > 1e-6, self.load_pred_sq_sum[b], 1.0)
                self.pred_load_correction[b] = np.clip(
                    self.load_pred_real_sum[b] / load_denom, 0.5, 2.0)

                solar_denom = np.where(self.solar_pred_sq_sum[b] > 1e-6, self.solar_pred_sq_sum[b], 1.0)
                self.pred_solar_correction[b] = np.clip(
                    self.solar_pred_real_sum[b] / solar_denom, 0.5, 2.0)

    def reset(self):
        """Reset all state."""
        self.feature_builder.reset()
        self._init_correction()
        self.last_load_pred = {i: None for i in range(self.n_buildings)}
        self.last_solar_pred = {i: None for i in range(self.n_buildings)}
        self.step_of_last_pred = {i: -1 for i in range(self.n_buildings)}
