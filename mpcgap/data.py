"""
Raw dataset access and forecaster factories shared by the experiment scripts.
"""

import pandas as pd
from citylearn.data import DataSet

from mpcgap.forecasters import PerfectForecaster
from mpcgap.lgb_forecaster import LGBForecaster


def load_building_data():
    """Load raw load/solar data for all buildings (for forecaster evaluation)."""
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')
    root_dir = schema['root_directory']

    all_buildings = [f'Building_{i}' for i in range(1, 18)]
    pv_caps = {}
    for b in all_buildings:
        if b in schema['buildings']:
            pv_caps[b] = schema['buildings'][b].get('pv', {}).get('nominal_power', 4.0)

    load_data = {}
    solar_data = {}
    for i, b in enumerate(all_buildings):
        df = pd.read_csv(f'{root_dir}/{b}.csv')
        load_data[i] = df['non_shiftable_load'].values
        solar_data[i] = df['solar_generation'].values * pv_caps.get(b, 4.0) / 1000.0

    return load_data, solar_data


def create_perfect_forecaster_factory():
    """Create factory that returns PerfectForecaster with pre-loaded data."""
    load_data, solar_data = load_building_data()

    def make_perfect(n_buildings, sim_start=0):
        return PerfectForecaster(
            {i: load_data[i] for i in range(n_buildings)},
            {i: solar_data[i] for i in range(n_buildings)},
            sim_start=sim_start,
        )
    return make_perfect


def make_lgb_forecaster(n_buildings, sim_start=0, phase_buildings=None):
    """Create LGBForecaster with correct building names."""
    return LGBForecaster(
        n_buildings, sim_start=sim_start,
        building_names=phase_buildings,
    )
