"""
Feature engineering for LightGBM forecasters.

Two classes:
- FeatureBuilder: builds training DataFrames from CityLearn CSV files (offline)
- OnlineFeatureBuilder: constructs feature vectors during MPC simulation (online)

Feature design follows Team Together's approach:
- Calendar features (month, hour, day_type) + cyclical encoding
- Weather actuals + 24h future weather (shifted from CSV)
- Full 168-hour (7 day) lag history for load and solar (like TT)
- Rolling statistics (24h mean/std)
- Building metadata (building_id, pv_capacity)
"""

import numpy as np
import pandas as pd
from citylearn.data import DataSet


# Feature column names (shared between offline and online builders)
CALENDAR_FEATURES = ['month', 'hour', 'day_type']
CYCLICAL_FEATURES = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
WEATHER_ACTUAL = [
    'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity',
    'diffuse_solar_irradiance', 'direct_solar_irradiance',
]
WEATHER_FUTURE_PREFIXES = [
    'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity',
    'diffuse_solar_irradiance', 'direct_solar_irradiance',
]

N_LAGS = 168  # 7 days of hourly lags
N_FUTURE_WEATHER = 24  # 24h ahead


def _weather_future_cols():
    """Column names for future weather features (h1..h24 for each variable)."""
    cols = []
    for prefix in WEATHER_FUTURE_PREFIXES:
        for h in range(N_FUTURE_WEATHER):
            cols.append(f'{prefix}_h{h+1}')
    return cols


def _lag_cols(prefix):
    """Column names for lag features: 168 individual lags + rolling stats."""
    lags = [f'{prefix}_past_{i}' for i in range(N_LAGS)]
    stats = [f'{prefix}_rolling_mean_24', f'{prefix}_rolling_std_24']
    return lags + stats


def get_feature_columns():
    """Return ordered list of all feature column names."""
    cols = list(CALENDAR_FEATURES)
    cols += CYCLICAL_FEATURES
    cols += WEATHER_ACTUAL
    cols += _weather_future_cols()
    cols += _lag_cols('load')
    cols += _lag_cols('solar')
    cols += ['building_id', 'pv_capacity']
    return cols


def get_target_columns(prefix):
    """Return target column names for 24h ahead prediction."""
    return [f'{prefix}_h{h+1}' for h in range(24)]


class FeatureBuilder:
    """
    Build training DataFrames from CityLearn CSV files.

    Loads all 17 buildings, creates features + targets, returns
    a single DataFrame ready for LightGBM training.
    """

    def __init__(self, dataset_name='citylearn_challenge_2022_phase_all'):
        self.schema = DataSet.get_schema(dataset_name)
        self.root_dir = self.schema['root_directory']

        # Load weather data once
        self.weather_df = pd.read_csv(f'{self.root_dir}/weather.csv')

        # Get PV capacities
        self.pv_caps = {}
        self.all_buildings = [f'Building_{i}' for i in range(1, 18)]
        for b in self.all_buildings:
            if b in self.schema['buildings']:
                self.pv_caps[b] = self.schema['buildings'][b].get(
                    'pv', {}
                ).get('nominal_power', 4.0)

    def build(self, buildings=None):
        """
        Build feature DataFrame for specified buildings.

        Args:
            buildings: list of building names, or None for all 17

        Returns:
            X: DataFrame with feature columns
            y_load: DataFrame with 24 load target columns
            y_solar: DataFrame with 24 solar target columns
        """
        if buildings is None:
            buildings = self.all_buildings

        all_X = []
        all_y_load = []
        all_y_solar = []

        for b_idx, b_name in enumerate(buildings):
            df = self._build_one_building(b_idx, b_name)
            if df is not None:
                feature_cols = get_feature_columns()
                load_target_cols = get_target_columns('load')
                solar_target_cols = get_target_columns('solar')
                all_X.append(df[feature_cols])
                all_y_load.append(df[load_target_cols])
                all_y_solar.append(df[solar_target_cols])

        X = pd.concat(all_X, ignore_index=True)
        y_load = pd.concat(all_y_load, ignore_index=True)
        y_solar = pd.concat(all_y_solar, ignore_index=True)

        return X, y_load, y_solar

    def _build_one_building(self, b_idx, b_name):
        """Build features for one building."""
        bdf = pd.read_csv(f'{self.root_dir}/{b_name}.csv')
        pv_cap = self.pv_caps.get(b_name, 4.0)
        T = len(bdf)

        load = bdf['non_shiftable_load'].values
        solar = bdf['solar_generation'].values * pv_cap / 1000.0

        # Calendar
        month = bdf['month'].values
        hour = bdf['hour'].values % 24
        day_type = bdf['day_type'].values

        # Cyclical
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # Weather actual
        w = self.weather_df
        temp = w['outdoor_dry_bulb_temperature'].values
        humid = w['outdoor_relative_humidity'].values
        diffuse = w['diffuse_solar_irradiance'].values
        direct = w['direct_solar_irradiance'].values

        # Build data dict
        data = {
            'month': month, 'hour': hour, 'day_type': day_type,
            'hour_sin': hour_sin, 'hour_cos': hour_cos,
            'month_sin': month_sin, 'month_cos': month_cos,
            'outdoor_dry_bulb_temperature': temp[:T],
            'outdoor_relative_humidity': humid[:T],
            'diffuse_solar_irradiance': diffuse[:T],
            'direct_solar_irradiance': direct[:T],
        }

        # Future weather (shifted from CSV)
        weather_arrays = {
            'outdoor_dry_bulb_temperature': temp,
            'outdoor_relative_humidity': humid,
            'diffuse_solar_irradiance': diffuse,
            'direct_solar_irradiance': direct,
        }
        for prefix, arr in weather_arrays.items():
            for h in range(N_FUTURE_WEATHER):
                col = f'{prefix}_h{h+1}'
                shifted = np.full(T, np.nan)
                if h + 1 < len(arr):
                    end = min(T, len(arr) - h - 1)
                    shifted[:end] = arr[h + 1:h + 1 + end]
                data[col] = shifted

        # Full 168-hour load lags (like Team Together: Load_Past_0..167)
        for i in range(N_LAGS):
            col = f'load_past_{i}'
            shifted = np.full(T, np.nan)
            if i + 1 < T:
                shifted[i + 1:] = load[:T - i - 1]
            data[col] = shifted

        # Load rolling stats
        load_series = pd.Series(load)
        rm = load_series.rolling(24, min_periods=1).mean().values
        rs = load_series.rolling(24, min_periods=1).std().fillna(0).values
        data['load_rolling_mean_24'] = np.concatenate([[np.nan], rm[:T - 1]])
        data['load_rolling_std_24'] = np.concatenate([[np.nan], rs[:T - 1]])

        # Full 168-hour solar lags (like Team Together: Solar_Past_0..167)
        for i in range(N_LAGS):
            col = f'solar_past_{i}'
            shifted = np.full(T, np.nan)
            if i + 1 < T:
                shifted[i + 1:] = solar[:T - i - 1]
            data[col] = shifted

        # Solar rolling stats
        solar_series = pd.Series(solar)
        rm_s = solar_series.rolling(24, min_periods=1).mean().values
        rs_s = solar_series.rolling(24, min_periods=1).std().fillna(0).values
        data['solar_rolling_mean_24'] = np.concatenate([[np.nan], rm_s[:T - 1]])
        data['solar_rolling_std_24'] = np.concatenate([[np.nan], rs_s[:T - 1]])

        # Building metadata
        data['building_id'] = np.full(T, b_idx, dtype=np.int32)
        data['pv_capacity'] = np.full(T, pv_cap)

        # Targets: 24h ahead load and solar
        for h in range(24):
            load_col = f'load_h{h+1}'
            solar_col = f'solar_h{h+1}'
            data[load_col] = np.full(T, np.nan)
            data[solar_col] = np.full(T, np.nan)
            if h + 1 < T:
                data[load_col][:T - h - 1] = load[h + 1:T]
                data[solar_col][:T - h - 1] = solar[h + 1:T]

        df = pd.DataFrame(data)

        # Drop rows with NaN in essential features (lags require warmup)
        df = df.dropna()

        return df


class OnlineFeatureBuilder:
    """
    Build feature vectors during MPC simulation.

    Pre-loads weather CSV at init. Accumulates load/solar history from
    update() calls. Constructs feature vectors for predict().
    """

    def __init__(self, n_buildings, sim_start=0, building_names=None,
                 dataset_name='citylearn_challenge_2022_phase_all'):
        self.n_buildings = n_buildings
        self.sim_start = sim_start

        schema = DataSet.get_schema(dataset_name)
        root_dir = schema['root_directory']

        # Pre-load weather data (full year)
        w = pd.read_csv(f'{root_dir}/weather.csv')
        self.temp = w['outdoor_dry_bulb_temperature'].values
        self.humid = w['outdoor_relative_humidity'].values
        self.diffuse = w['diffuse_solar_irradiance'].values
        self.direct = w['direct_solar_irradiance'].values

        # Pre-load building CSV data for initial lags (before sim starts)
        self.preloaded_load = {}
        self.preloaded_solar = {}
        if building_names is None:
            building_names = [f'Building_{i}' for i in range(1, n_buildings + 1)]
        self.building_names = building_names

        # Get PV capacities
        self.pv_caps = []
        for b in building_names:
            if b in schema['buildings']:
                self.pv_caps.append(
                    schema['buildings'][b].get('pv', {}).get('nominal_power', 4.0)
                )
            else:
                self.pv_caps.append(4.0)

        # Pre-load historical data for lag warmup
        for i, b in enumerate(building_names):
            bdf = pd.read_csv(f'{root_dir}/{b}.csv')
            self.preloaded_load[i] = bdf['non_shiftable_load'].values
            self.preloaded_solar[i] = bdf['solar_generation'].values * self.pv_caps[i] / 1000.0

        # Building ID mapping
        all_buildings = [f'Building_{i}' for i in range(1, 18)]
        self.building_ids = []
        for b in building_names:
            if b in all_buildings:
                self.building_ids.append(all_buildings.index(b))
            else:
                self.building_ids.append(0)

        # History buffers (per building)
        self.load_hist = {i: [] for i in range(n_buildings)}
        self.solar_hist = {i: [] for i in range(n_buildings)}

        self.feature_columns = get_feature_columns()

    def update(self, building_idx, load, solar):
        """Append new observation to history."""
        self.load_hist[building_idx].append(load)
        self.solar_hist[building_idx].append(solar)
        # Keep enough for 168 lags + buffer
        max_hist = N_LAGS + 48
        if len(self.load_hist[building_idx]) > max_hist:
            self.load_hist[building_idx] = self.load_hist[building_idx][-max_hist:]
            self.solar_hist[building_idx] = self.solar_hist[building_idx][-max_hist:]

    def _get_lag_value(self, building_idx, lag, variable='load'):
        """
        Get lag value, using pre-loaded CSV data for warmup period.

        lag=0 means most recent observation (shift by 1),
        lag=167 means 168 steps ago.
        """
        if variable == 'load':
            hist = self.load_hist[building_idx]
            preloaded = self.preloaded_load[building_idx]
        else:
            hist = self.solar_hist[building_idx]
            preloaded = self.preloaded_solar[building_idx]

        n = len(hist)
        actual_lag = lag + 1  # lag=0 -> shift(1)

        if n >= actual_lag:
            return hist[-actual_lag]
        else:
            # Use pre-loaded CSV data
            global_step = self.sim_start + n - actual_lag
            if 0 <= global_step < len(preloaded):
                return preloaded[global_step]
            else:
                return 0.0

    def build_features(self, building_idx, current_step, current_hour):
        """
        Build feature vector for one building at current timestep.

        Returns:
            features: dict with feature column names as keys
        """
        g = self.sim_start + current_step  # global index
        month = self._get_month(g)

        feat = {}

        # Calendar
        feat['month'] = month
        feat['hour'] = current_hour
        feat['day_type'] = self._get_day_type(g)

        # Cyclical
        feat['hour_sin'] = np.sin(2 * np.pi * current_hour / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * current_hour / 24)
        feat['month_sin'] = np.sin(2 * np.pi * month / 12)
        feat['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Weather actual
        if g < len(self.temp):
            feat['outdoor_dry_bulb_temperature'] = self.temp[g]
            feat['outdoor_relative_humidity'] = self.humid[g]
            feat['diffuse_solar_irradiance'] = self.diffuse[g]
            feat['direct_solar_irradiance'] = self.direct[g]
        else:
            feat['outdoor_dry_bulb_temperature'] = self.temp[-1]
            feat['outdoor_relative_humidity'] = self.humid[-1]
            feat['diffuse_solar_irradiance'] = self.diffuse[-1]
            feat['direct_solar_irradiance'] = self.direct[-1]

        # Future weather
        for prefix, arr in [
            ('outdoor_dry_bulb_temperature', self.temp),
            ('outdoor_relative_humidity', self.humid),
            ('diffuse_solar_irradiance', self.diffuse),
            ('direct_solar_irradiance', self.direct),
        ]:
            for h in range(N_FUTURE_WEATHER):
                idx = g + h + 1
                if idx < len(arr):
                    feat[f'{prefix}_h{h+1}'] = arr[idx]
                else:
                    feat[f'{prefix}_h{h+1}'] = arr[-1]

        # Full 168 load lags
        for i in range(N_LAGS):
            feat[f'load_past_{i}'] = self._get_lag_value(building_idx, i, 'load')

        # Load rolling stats (from recent history)
        load_h = self.load_hist[building_idx]
        n = len(load_h)
        if n >= 24:
            last24 = load_h[-24:]
            feat['load_rolling_mean_24'] = np.mean(last24)
            feat['load_rolling_std_24'] = np.std(last24)
        elif load_h:
            feat['load_rolling_mean_24'] = np.mean(load_h)
            feat['load_rolling_std_24'] = np.std(load_h) if n > 1 else 0.0
        else:
            feat['load_rolling_mean_24'] = 1.0
            feat['load_rolling_std_24'] = 0.0

        # Full 168 solar lags
        for i in range(N_LAGS):
            feat[f'solar_past_{i}'] = self._get_lag_value(building_idx, i, 'solar')

        # Solar rolling stats
        solar_h = self.solar_hist[building_idx]
        n_s = len(solar_h)
        if n_s >= 24:
            last24_s = solar_h[-24:]
            feat['solar_rolling_mean_24'] = np.mean(last24_s)
            feat['solar_rolling_std_24'] = np.std(last24_s)
        elif solar_h:
            feat['solar_rolling_mean_24'] = np.mean(solar_h)
            feat['solar_rolling_std_24'] = np.std(solar_h) if n_s > 1 else 0.0
        else:
            feat['solar_rolling_mean_24'] = 0.0
            feat['solar_rolling_std_24'] = 0.0

        # Building metadata
        feat['building_id'] = self.building_ids[building_idx]
        feat['pv_capacity'] = self.pv_caps[building_idx]

        return feat

    def reset(self):
        """Clear all history."""
        for i in range(self.n_buildings):
            self.load_hist[i] = []
            self.solar_hist[i] = []

    def _get_month(self, global_step):
        """Approximate month from global step (hour index)."""
        day = global_step // 24
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        cumulative = 0
        for m, d in enumerate(month_days, 1):
            cumulative += d
            if day < cumulative:
                return m
        return 12

    def _get_day_type(self, global_step):
        """Approximate day_type (1-8) from global step."""
        day = global_step // 24
        weekday = (day + 5) % 7  # 0=Mon, 6=Sun
        if weekday < 5:
            return 1  # workday
        return 8  # weekend


if __name__ == '__main__':
    print("Building training features...")
    fb = FeatureBuilder()
    X, y_load, y_solar = fb.build()
    print(f"X shape: {X.shape}")
    print(f"y_load shape: {y_load.shape}")
    print(f"y_solar shape: {y_solar.shape}")
    print(f"\nFeature columns ({len(X.columns)})")
    print(f"First 20: {list(X.columns[:20])}")
    print(f"Last 20: {list(X.columns[-20:])}")
