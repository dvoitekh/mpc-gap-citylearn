"""
Forecasting modules for Online MPC.

Three forecasters with different information levels:
- PerfectForecaster: uses pre-loaded ground truth data (upper bound)
- PersistenceForecaster: uses yesterday's pattern (lag-24)
- HoltWintersForecaster: exponential smoothing on historical data
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseForecaster(ABC):
    """Base class for 24-hour ahead forecasters."""

    @abstractmethod
    def predict(self, building_idx, current_step, current_hour):
        """
        Predict load and solar for next 24 hours.

        Args:
            building_idx: int, building index
            current_step: int, global simulation step
            current_hour: int, hour of day (0-23)

        Returns:
            load_pred: array (24,) predicted load in kW
            solar_pred: array (24,) predicted solar in kW
        """
        pass

    @abstractmethod
    def update(self, building_idx, load, solar, observation=None):
        """
        Update forecaster with new observation.

        Args:
            building_idx: int
            load: float, observed load in kW
            solar: float, observed solar in kW
            observation: array, full CityLearn observation (optional)
        """
        pass

    def reset(self):
        """Reset forecaster state."""
        pass


class PerfectForecaster(BaseForecaster):
    """
    Perfect foresight forecaster — loads all data before simulation.

    This is clearly an UPPER BOUND on forecasting quality.
    Used to measure the gap between perfect and imperfect forecasting.

    Note: load_data/solar_data contain FULL YEAR data.
    The sim_start offset maps local step 0 -> global position in the arrays.
    """

    def __init__(self, load_data, solar_data, sim_start=0):
        """
        Args:
            load_data: dict {building_idx: array of all load values (full year)}
            solar_data: dict {building_idx: array of all solar values (full year)}
            sim_start: int, offset into the data arrays for this phase
        """
        self.load_data = load_data
        self.solar_data = solar_data
        self.sim_start = sim_start

    def predict(self, building_idx, current_step, current_hour):
        load_all = self.load_data[building_idx]
        solar_all = self.solar_data[building_idx]

        load_pred = np.zeros(24)
        solar_pred = np.zeros(24)
        for h in range(24):
            idx = self.sim_start + current_step + 1 + h
            if idx < len(load_all):
                load_pred[h] = load_all[idx]
                solar_pred[h] = solar_all[idx]
            else:
                load_pred[h] = load_all[-1]
                solar_pred[h] = solar_all[-1]

        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        pass  # No update needed — data is pre-loaded

    def reset(self):
        pass


class PersistenceForecaster(BaseForecaster):
    """
    Persistence forecaster — predicts next 24h = last 24h observed.

    This is the simplest reasonable forecasting baseline.
    """

    def __init__(self, n_buildings):
        self.n_buildings = n_buildings
        self.load_hist = {i: [] for i in range(n_buildings)}
        self.solar_hist = {i: [] for i in range(n_buildings)}

    def predict(self, building_idx, current_step, current_hour):
        load_h = self.load_hist[building_idx]
        solar_h = self.solar_hist[building_idx]

        if len(load_h) >= 24:
            load_pred = np.array(load_h[-24:])
            solar_pred = np.array(solar_h[-24:])
        else:
            load_pred = np.ones(24) * (np.mean(load_h) if load_h else 1.0)
            solar_pred = np.zeros(24)

        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        self.load_hist[building_idx].append(load)
        self.solar_hist[building_idx].append(solar)
        # Keep last 168 hours (1 week)
        if len(self.load_hist[building_idx]) > 168:
            self.load_hist[building_idx] = self.load_hist[building_idx][-168:]
            self.solar_hist[building_idx] = self.solar_hist[building_idx][-168:]

    def reset(self):
        for i in range(self.n_buildings):
            self.load_hist[i] = []
            self.solar_hist[i] = []


class HoltWintersForecaster(BaseForecaster):
    """
    Holt-Winters (triple exponential smoothing) with 24h seasonality.

    Uses additive seasonality with period=24 (hourly pattern).
    Better than persistence because it adapts level and trend.
    """

    def __init__(self, n_buildings, alpha=0.3, beta=0.05, gamma=0.3):
        self.n_buildings = n_buildings
        self.alpha = alpha  # level smoothing
        self.beta = beta    # trend smoothing
        self.gamma = gamma  # seasonal smoothing
        self.period = 24

        self.load_hist = {i: [] for i in range(n_buildings)}
        self.solar_hist = {i: [] for i in range(n_buildings)}

    def _hw_predict(self, history, steps=24):
        """Apply Holt-Winters to a single time series."""
        n = len(history)
        if n < 2 * self.period:
            # Not enough data — fall back to persistence
            if n >= 24:
                return np.array(history[-24:])
            return np.ones(steps) * (np.mean(history) if history else 1.0)

        y = np.array(history, dtype=float)
        p = self.period

        # Initialize seasonal components from first two periods
        season = np.zeros(p)
        for j in range(p):
            season[j] = np.mean(y[j::p][:2]) - np.mean(y[:2 * p])

        level = np.mean(y[:p])
        trend = (np.mean(y[p:2 * p]) - np.mean(y[:p])) / p

        # Fit on historical data
        seasonal = list(season)
        for t in range(n):
            s_idx = t % p
            old_season = seasonal[s_idx]

            new_level = self.alpha * (y[t] - old_season) + (1 - self.alpha) * (level + trend)
            new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend
            new_season = self.gamma * (y[t] - new_level) + (1 - self.gamma) * old_season

            level = new_level
            trend = new_trend
            seasonal[s_idx] = new_season

        # Forecast
        forecast = np.zeros(steps)
        for h in range(steps):
            s_idx = (n + h) % p
            forecast[h] = level + (h + 1) * trend + seasonal[s_idx]

        return np.maximum(forecast, 0)

    def predict(self, building_idx, current_step, current_hour):
        load_pred = self._hw_predict(self.load_hist[building_idx])
        solar_pred = self._hw_predict(self.solar_hist[building_idx])
        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        self.load_hist[building_idx].append(load)
        self.solar_hist[building_idx].append(solar)
        # Keep last 2 weeks
        if len(self.load_hist[building_idx]) > 336:
            self.load_hist[building_idx] = self.load_hist[building_idx][-336:]
            self.solar_hist[building_idx] = self.solar_hist[building_idx][-336:]

    def reset(self):
        for i in range(self.n_buildings):
            self.load_hist[i] = []
            self.solar_hist[i] = []


class WeeklySeasonalityForecaster(BaseForecaster):
    """
    Weekly seasonality forecaster — predicts using data from 168 hours ago.

    Uses the same hour on the same weekday from last week.
    Captures weekly patterns (weekday vs weekend) that persistence misses.
    Falls back to persistence (lag-24) when less than 1 week of data is available.
    """

    def __init__(self, n_buildings):
        self.n_buildings = n_buildings
        self.load_hist = {i: [] for i in range(n_buildings)}
        self.solar_hist = {i: [] for i in range(n_buildings)}

    def predict(self, building_idx, current_step, current_hour):
        load_h = self.load_hist[building_idx]
        solar_h = self.solar_hist[building_idx]

        load_pred = np.zeros(24)
        solar_pred = np.zeros(24)

        for h in range(24):
            # Try weekly lag (168 hours ago + h offset)
            weekly_idx = len(load_h) - 168 + h
            daily_idx = len(load_h) - 24 + h

            if weekly_idx >= 0:
                load_pred[h] = load_h[weekly_idx]
                solar_pred[h] = solar_h[weekly_idx]
            elif daily_idx >= 0:
                # Fall back to persistence (lag-24)
                load_pred[h] = load_h[daily_idx]
                solar_pred[h] = solar_h[daily_idx]
            else:
                load_pred[h] = np.mean(load_h) if load_h else 1.0
                solar_pred[h] = 0.0

        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        self.load_hist[building_idx].append(load)
        self.solar_hist[building_idx].append(solar)
        # Keep last 2 weeks
        if len(self.load_hist[building_idx]) > 336:
            self.load_hist[building_idx] = self.load_hist[building_idx][-336:]
            self.solar_hist[building_idx] = self.solar_hist[building_idx][-336:]

    def reset(self):
        for i in range(self.n_buildings):
            self.load_hist[i] = []
            self.solar_hist[i] = []


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble of multiple forecasters with weighted averaging.

    Combines predictions from multiple forecasters to reduce variance.
    Weights can be uniform or custom.
    """

    def __init__(self, forecasters, weights=None):
        """
        Args:
            forecasters: list of BaseForecaster instances
            weights: list of floats (must sum to 1), or None for uniform
        """
        self.forecasters = forecasters
        if weights is None:
            self.weights = [1.0 / len(forecasters)] * len(forecasters)
        else:
            self.weights = weights

    def predict(self, building_idx, current_step, current_hour):
        load_pred = np.zeros(24)
        solar_pred = np.zeros(24)
        for f, w in zip(self.forecasters, self.weights):
            lp, sp = f.predict(building_idx, current_step, current_hour)
            load_pred += w * lp
            solar_pred += w * sp
        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        for f in self.forecasters:
            f.update(building_idx, load, solar, observation=observation)

    def reset(self):
        for f in self.forecasters:
            f.reset()


class HybridForecaster(BaseForecaster):
    """
    Hybrid forecaster: uses one forecaster for load, another for solar.

    Motivated by finding that persistence is better for solar (73.8% MAPE)
    while LGB is better for load at longer horizons.
    Also applies night solar hard constraint (solar=0 for hours 0-6, 20-23).
    """

    def __init__(self, load_forecaster, solar_forecaster):
        self.load_forecaster = load_forecaster
        self.solar_forecaster = solar_forecaster

    def predict(self, building_idx, current_step, current_hour):
        load_pred, _ = self.load_forecaster.predict(building_idx, current_step, current_hour)
        _, solar_pred = self.solar_forecaster.predict(building_idx, current_step, current_hour)

        # Night solar hard constraint
        for h in range(24):
            target_hour = (current_hour + h + 1) % 24
            if target_hour < 7 or target_hour >= 20:
                solar_pred[h] = 0.0

        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        self.load_forecaster.update(building_idx, load, solar, observation=observation)
        self.solar_forecaster.update(building_idx, load, solar, observation=observation)

    def reset(self):
        self.load_forecaster.reset()
        self.solar_forecaster.reset()


class HorizonBlendedForecaster(BaseForecaster):
    """
    Blends two forecasters by horizon: primary for h=0..blend_h-1,
    secondary for h=blend_h..23.

    Use case: LGB is more accurate at short horizons (h=1-8),
    Persistence is more stable at long horizons (h=9-24).
    """

    def __init__(self, primary_forecaster, secondary_forecaster, blend_h=8):
        self.primary = primary_forecaster
        self.secondary = secondary_forecaster
        self.blend_h = blend_h

    def predict(self, building_idx, current_step, current_hour):
        load_p, solar_p = self.primary.predict(building_idx, current_step, current_hour)
        load_s, solar_s = self.secondary.predict(building_idx, current_step, current_hour)

        load_pred = np.concatenate([load_p[:self.blend_h], load_s[self.blend_h:]])
        solar_pred = np.concatenate([solar_p[:self.blend_h], solar_s[self.blend_h:]])
        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        self.primary.update(building_idx, load, solar, observation=observation)
        self.secondary.update(building_idx, load, solar, observation=observation)

    def reset(self):
        self.primary.reset()
        self.secondary.reset()


def compute_forecast_mape(forecaster, load_truth, solar_truth, n_buildings, warmup=48):
    """
    Compute MAPE of a forecaster against ground truth data.

    Runs the forecaster through the data, collecting 24h-ahead predictions
    and comparing to actuals.

    Args:
        forecaster: BaseForecaster instance
        load_truth: dict {building_idx: array of true load values}
        solar_truth: dict {building_idx: array of true solar values}
        n_buildings: int
        warmup: int, steps to skip before measuring

    Returns:
        load_mape: float, mean absolute percentage error for load
        solar_mape: float, mean absolute percentage error for solar
    """
    T = len(load_truth[0])
    load_errors = []
    solar_errors = []

    forecaster.reset()

    for t in range(T):
        hour = t % 24

        # Update with current observation
        for b in range(n_buildings):
            forecaster.update(b, load_truth[b][t], solar_truth[b][t], observation=None)

        # Predict every 24 steps after warmup
        if t >= warmup and t + 24 < T and hour == 0:
            for b in range(n_buildings):
                load_pred, solar_pred = forecaster.predict(b, t, hour)
                load_actual = load_truth[b][t + 1:t + 25]
                solar_actual = solar_truth[b][t + 1:t + 25]

                # MAPE (avoid division by zero)
                mask = load_actual > 0.1
                if mask.any():
                    load_errors.append(np.mean(np.abs(load_pred[mask] - load_actual[mask]) / load_actual[mask]))

                mask = solar_actual > 0.1
                if mask.any():
                    solar_errors.append(np.mean(np.abs(solar_pred[mask] - solar_actual[mask]) / solar_actual[mask]))

    load_mape = np.mean(load_errors) * 100 if load_errors else float('inf')
    solar_mape = np.mean(solar_errors) * 100 if solar_errors else float('inf')

    return load_mape, solar_mape
