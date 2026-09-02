"""
Perfect Foresight LP Agent for CityLearn 2022 Challenge.

Solves a single LP for the entire episode with known future data.
This is the perfect-foresight reference used in the gap decomposition. Note
that the objective is a linear surrogate of the CityLearn score (cost,
emissions, ramping, monthly load factor), evaluated afterwards under the full
score, so the result is an achievable-in-hindsight operating point rather than
a proven optimum.

Objective function (from Team Together's Method.md):
  min: price_term / price_baseline
     + emission_term / emission_baseline
     + 0.5 * ramping_term / ramping_baseline
     + 0.5 * load_factor_term / load_factor_baseline

where:
  price_term     = Σ_t max(district_consumption[t], 0) * price[t]
  emission_term  = Σ_t Σ_b max(building_consumption[b,t], 0) * carbon[t]
  ramping_term   = Σ_t |Δdistrict_consumption[t]|
  load_factor_term = Σ_m (1 - avg_m/max_m)  (CityLearn normalized form)

Grid metric in CityLearn 2022 evaluation:
  Grid = (ramping + daily_peak + 1-load_factor + all_time_peak) / 4

Subject to:
  SOC[b,t+1] = SOC[b,t] + η*charge[b,t] - discharge[b,t]/η
  0 <= SOC[b,t] <= 1
  0 <= charge <= P_max, 0 <= discharge <= P_max
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import time
from citylearn.data import DataSet


class PerfectForesightLP:
    """LP optimization with perfect foresight for CityLearn."""

    def __init__(self, capacity=6.4, efficiency=0.912):
        self.capacity = capacity
        self.efficiency = efficiency
        self.inv_efficiency = 1 / efficiency
        self.max_power = 5.0 / capacity  # Normalized to capacity

        self.actions = None
        self.step = 0

    def solve(self, L_bd, prices, carbon, initial_soc=None):
        """
        Solve LP for entire episode with perfect foresight.

        Args:
            L_bd: array (n_buildings, T) - net load = (load - solar) / capacity
            prices: array (T,) - electricity prices
            carbon: array (T,) - carbon intensity
            initial_soc: array (n_buildings,) - initial SOC [0,1]

        Returns:
            actions: array (n_buildings, T) - optimal battery actions
        """
        N, T = L_bd.shape

        if initial_soc is None:
            initial_soc = [0.0] * N

        # Calculate baselines (no-battery values)
        district_no_bat = L_bd.sum(axis=0)  # sum across buildings
        emission_base = np.sum(L_bd.clip(0).sum(axis=0) * carbon) + 0.01
        price_base = np.sum(district_no_bat.clip(0) * prices) + 0.01
        ramping_base = np.sum(np.abs(np.diff(district_no_bat))) + 0.01

        # Monthly load factor baseline: CityLearn uses 1 - avg/max per month
        hours_per_month = 730
        n_months = max(1, T // hours_per_month)
        lf_base = 0.0
        for m in range(n_months):
            t_start = m * hours_per_month
            t_end = min((m + 1) * hours_per_month, T)
            chunk = district_no_bat[t_start:t_end]
            chunk_max = np.max(chunk)
            if chunk_max > 0:
                lf_base += 1 - np.mean(chunk) / chunk_max
            else:
                lf_base += 0.0
        lf_base = max(lf_base, 0.01)

        print(f"LP: T={T}, N={N}")
        print(f"  Baselines: price={price_base:.2f}, emission={emission_base:.2f}, "
              f"ramping={ramping_base:.2f}, load_factor={lf_base:.2f}")

        c1, c2 = self.efficiency, self.inv_efficiency

        # Decision variables
        x = cp.Variable((N, T), nonneg=True)   # charge (x^+)
        y = cp.Variable((N, T), nonpos=True)    # discharge (x^-)
        soc = cp.Variable((N, T))
        vu = cp.Variable((N, T), nonneg=True)   # building consumption (clipped >=0)
        mu = cp.Variable(T, nonneg=True)         # district consumption (clipped >=0)
        w = cp.Variable(T - 1, nonneg=True)      # ramping |delta|

        # Monthly peak variables for load factor
        e_max = cp.Variable(n_months)  # monthly peak
        e_avg = cp.Variable(n_months)  # monthly average

        constraints = [x <= self.max_power, y >= -self.max_power]

        # SOC dynamics
        for i in range(N):
            constraints.append(soc[i, 0] == initial_soc[i] + x[i, 0] * c1 + y[i, 0] * c2)
            for t in range(1, T):
                constraints.append(soc[i, t] == soc[i, t - 1] + x[i, t] * c1 + y[i, t] * c2)
            constraints += [soc[i, :] >= 0, soc[i, :] <= 1]

        # Building consumption >= net load + battery action (per building)
        for i in range(N):
            for t in range(T):
                constraints.append(vu[i, t] >= L_bd[i, t] + x[i, t] + y[i, t])

        # District consumption >= sum of (net load + battery) across buildings
        for t in range(T):
            constraints.append(mu[t] >= cp.sum(L_bd[:, t] + x[:, t] + y[:, t]))

        # Ramping: |district[t+1] - district[t]|
        for t in range(T - 1):
            delta = cp.sum(L_bd[:, t + 1] + x[:, t + 1] + y[:, t + 1]) - cp.sum(L_bd[:, t] + x[:, t] + y[:, t])
            constraints += [w[t] >= delta, w[t] >= -delta]

        # Monthly load factor constraints
        for m in range(n_months):
            t_start = m * hours_per_month
            t_end = min((m + 1) * hours_per_month, T)
            chunk_len = t_end - t_start

            # Average district consumption in month m
            month_sum = cp.sum(L_bd[:, t_start:t_end].sum(axis=0) + cp.sum(x[:, t_start:t_end] + y[:, t_start:t_end], axis=0))
            constraints.append(e_avg[m] == month_sum / chunk_len)

            # Peak district consumption in month m: e_max[m] >= district[t] for all t in month
            for t in range(t_start, t_end):
                constraints.append(e_max[m] >= cp.sum(L_bd[:, t] + x[:, t] + y[:, t]))

        # Objective: price + emission + 0.5*ramping + 0.5*load_factor (TT formulation)
        obj = (cp.sum(cp.multiply(mu, prices)) / price_base +
               cp.sum(cp.multiply(vu, carbon)) / emission_base +
               0.5 * cp.sum(w) / ramping_base +
               0.5 * cp.sum(e_max - e_avg) / lf_base)

        problem = cp.Problem(cp.Minimize(obj), constraints)

        start_time = time.time()
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            solve_time = time.time() - start_time
            print(f"  Solved in {solve_time:.1f}s, status={problem.status}, obj={problem.value:.4f}")

            if problem.status in ['optimal', 'optimal_inaccurate']:
                self.actions = x.value + y.value
                self.step = 0
                return self.actions
        except Exception as e:
            print(f"  LP solver failed: {e}")

        print("  LP failed, returning zeros")
        self.actions = np.zeros((N, T))
        self.step = 0
        return self.actions

    def get_action(self, step=None):
        """Get pre-computed action for current step."""
        if self.actions is None:
            return None
        if step is None:
            step = self.step
            self.step += 1
        if step < self.actions.shape[1]:
            return self.actions[:, step]
        return np.zeros(self.actions.shape[0])

    def reset(self):
        self.actions = None
        self.step = 0


def load_citylearn_data(dataset_name='citylearn_challenge_2022_phase_all'):
    """
    Load all CityLearn data for perfect foresight.

    Returns:
        net_load: dict {building_name: (load - solar) / capacity array}
        price: array of electricity prices
        carbon: array of carbon intensity
        pv_capacities: dict {building_name: PV capacity kW}
    """
    schema = DataSet.get_schema(dataset_name)
    root_dir = schema['root_directory']

    all_buildings = [f'Building_{i}' for i in range(1, 18)]
    pv_capacities = {}
    for b in all_buildings:
        if b in schema['buildings']:
            pv_capacities[b] = schema['buildings'][b].get('pv', {}).get('nominal_power', 4.0)

    capacity = 6.4

    carbon_df = pd.read_csv(f'{root_dir}/carbon_intensity.csv')
    pricing_df = pd.read_csv(f'{root_dir}/pricing.csv')
    price = pricing_df['electricity_pricing'].values
    carbon = carbon_df['carbon_intensity'].values

    # Pad with last value
    price = np.append(price, price[-1])
    carbon = np.append(carbon, carbon[-1])

    net_load = {}
    for b in all_buildings:
        df = pd.read_csv(f'{root_dir}/{b}.csv')
        solar_gen = df['solar_generation'].values * pv_capacities.get(b, 4.0) / 1000.0
        load = df['non_shiftable_load'].values
        data = (load - solar_gen) / capacity
        net_load[b] = np.append(data, data[-1])

    return net_load, price, carbon, pv_capacities


def solve_lp_phase(buildings, sim_start, sim_end, net_load_all, price_full, carbon_full):
    """
    Solve LP for a specific phase.

    Uses 1-indexed offset (like Team Together's code).
    """
    T = sim_end - sim_start + 1
    N = len(buildings)

    L_bd = np.zeros((N, T))
    for i, b in enumerate(buildings):
        L_bd[i, :] = net_load_all[b][sim_start + 1:sim_start + 1 + T]

    P = price_full[sim_start + 1:sim_start + 1 + T]
    E = carbon_full[sim_start + 1:sim_start + 1 + T]

    agent = PerfectForesightLP()
    actions = agent.solve(L_bd, P, E)
    return actions


def evaluate_lp_phase(buildings, sim_start, sim_end, actions):
    """Run simulation with LP actions and evaluate."""
    from environment import create_env, evaluate_env

    config = {
        'dataset_name': 'citylearn_challenge_2022_phase_all',
        'buildings': buildings,
        'sim_start': sim_start,
        'sim_end': sim_end,
        'central_agent': True,
    }

    env = create_env(config)
    obs, _ = env.reset()

    step = 0
    n_buildings = len(buildings)
    while not env.terminated:
        if step < actions.shape[1]:
            action = [[float(actions[b, step]) for b in range(n_buildings)]]
        else:
            action = [[0.0] * n_buildings]
        obs, _, _, _, _ = env.step(action)
        step += 1

    metrics = evaluate_env(env)
    return metrics


def run_full_evaluation():
    """Run full Phase 1-3 weighted evaluation."""
    print("Loading CityLearn data...")
    net_load, price, carbon, pv_caps = load_citylearn_data()

    phases = [
        {
            'name': 'Phase 1',
            'buildings': [f'Building_{i}' for i in range(1, 6)],
            'sim_start': 0,
            'sim_end': 24 * 120 - 1,
            'weight': 0.2,
        },
        {
            'name': 'Phase 2',
            'buildings': [f'Building_{i}' for i in range(1, 6)],
            'sim_start': 24 * 120,
            'sim_end': 24 * 240 - 1,
            'weight': 0.3,
        },
        {
            'name': 'Phase 3',
            'buildings': [f'Building_{i}' for i in range(1, 18) if i not in [12, 15]],
            'sim_start': 24 * 240,
            'sim_end': 24 * 365 - 1,
            'weight': 0.5,
        },
    ]

    print('=' * 70)
    print('LP PERFECT FORESIGHT - FULL PHASE 1-3 EVALUATION')
    print('=' * 70)

    weighted_score = 0
    phase_results = []

    for phase in phases:
        print(f"\nSolving {phase['name']}...")
        actions = solve_lp_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'],
            net_load, price, carbon
        )
        metrics = evaluate_lp_phase(
            phase['buildings'], phase['sim_start'], phase['sim_end'], actions
        )

        print(f"{phase['name']}: Score={metrics['score']:.4f}, "
              f"Cost={metrics['cost']:.4f}, Emis={metrics['emissions']:.4f}, "
              f"Grid={metrics['grid']:.4f}")
        print(f"  Grid breakdown: ramping={metrics['ramping']:.4f}, "
              f"peak={metrics['daily_peak']:.4f}, "
              f"load_factor={metrics['load_factor']:.4f}, "
              f"all_time_peak={metrics['all_time_peak']:.4f}")
        if 'official_score' in metrics:
            print(f"  Official-2022 metric: score={metrics['official_score']:.4f}, "
                  f"grid={metrics['official_grid']:.4f}, "
                  f"monthly_lf={metrics['monthly_load_factor']:.4f}")

        phase_results.append(metrics)
        weighted_score += phase['weight'] * metrics['score']

    print('\n' + '=' * 70)
    print('FINAL RESULTS')
    print('=' * 70)
    official_weighted = 0.0
    for i, (phase, metrics) in enumerate(zip(phases, phase_results)):
        print(f"{phase['name']}: {metrics['score']:.4f} (weight {phase['weight']})")
        official_weighted += phase['weight'] * metrics.get('official_score', float('nan'))
    print(f'\nLP WEIGHTED SCORE: {weighted_score:.4f}')
    print(f'LP WEIGHTED SCORE (official 2022 metric): {official_weighted:.4f}')
    import json as _json
    with open('results/verify_lp.json', 'w') as f:
        _json.dump({'weighted_score': weighted_score,
                    'official_weighted': official_weighted,
                    'per_phase': [{k: float(v) for k, v in m.items()}
                                  for m in phase_results]}, f, indent=2)

    return weighted_score, phase_results


if __name__ == "__main__":
    run_full_evaluation()
