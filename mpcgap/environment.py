"""
CityLearn Environment Wrapper for MPC experiments.

Provides:
- create_env(): Create CityLearn environment from config dict
- evaluate_env(): Evaluate using CityLearn 2022 Challenge scoring
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet


def create_env(config):
    """
    Create CityLearn environment from config dict.

    Args:
        config: dict with keys:
            - dataset_name: str (default 'citylearn_challenge_2022_phase_all')
            - buildings: list of building names
            - sim_start: int, start timestep
            - sim_end: int, end timestep
            - central_agent: bool (default True)
    """
    env = CityLearnEnv(
        schema=config.get('dataset_name', 'citylearn_challenge_2022_phase_all'),
        central_agent=config.get('central_agent', True),
        buildings=config['buildings'],
        simulation_start_time_step=config['sim_start'],
        simulation_end_time_step=config['sim_end'],
    )
    return env


def evaluate_env(env):
    """
    Evaluate environment using CityLearn 2022 Challenge scoring.

    Score = (Cost + Emissions + Grid) / 3
    Grid = (ramping + 1-load_factor + daily_peak + all_time_peak) / 4

    All metrics normalized to baseline (1.0 = no control).
    """
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    kpis = unwrapped.evaluate()

    district_kpis = kpis[kpis['level'] == 'district'] if 'level' in kpis.columns else kpis

    def get_metric(name, default=1.0):
        row = district_kpis[district_kpis['cost_function'] == name]
        if not row.empty:
            val = row['value'].iloc[0]
            return float(val) if not np.isnan(val) else default
        return default

    metrics = {}
    metrics['cost'] = get_metric('cost_total')
    metrics['emissions'] = get_metric('carbon_emissions_total')

    ramping = get_metric('ramping_average')
    one_minus_lf = get_metric('daily_one_minus_load_factor_average')
    daily_peak = get_metric('daily_peak_average')
    all_time_peak = get_metric('all_time_peak_average')

    metrics['grid'] = (ramping + one_minus_lf + daily_peak + all_time_peak) / 4
    metrics['score'] = (metrics['cost'] + metrics['emissions'] + metrics['grid']) / 3

    metrics['ramping'] = ramping
    metrics['load_factor'] = one_minus_lf
    metrics['daily_peak'] = daily_peak
    metrics['all_time_peak'] = all_time_peak

    # Official CityLearn 2022 challenge scoring (citylearn 1.3.6 evaluate()):
    # grid KPI D = (ramping + monthly one-minus-load-factor) / 2,
    # score = (cost + emissions + D) / 3
    monthly_lf = get_metric('monthly_one_minus_load_factor_average')
    metrics['monthly_load_factor'] = monthly_lf
    metrics['official_grid'] = (ramping + monthly_lf) / 2
    metrics['official_score'] = (metrics['cost'] + metrics['emissions']
                                 + metrics['official_grid']) / 3

    return metrics
