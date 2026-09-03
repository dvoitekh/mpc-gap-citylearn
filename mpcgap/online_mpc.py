"""
Online Rolling-Horizon MPC Agent for CityLearn 2022 Challenge.

Key design:
- Takes forecasts from a pluggable forecaster (no energy_simulation access)
- Solves 24h rolling horizon LP with CVXPY/CLARABEL
- Same objective as LP perfect foresight (price + emission + ramping + load_factor)
- Re-optimizes every 8 hours (at hours 0, 8, 16)

Objective (normalized):
  min: price_term + emission_term + 0.5 * ramping_term + 0.5 * load_factor_approx
"""

import numpy as np
import cvxpy as cp
import pandas as pd
from citylearn.data import DataSet
from mpcgap.forecasters import BaseForecaster


class OnlineMPC:
    """Rolling-horizon MPC with pluggable forecaster."""

    # LGB per-horizon load MAPE (h=1..24), measured on Phase 2
    LGB_NOISE_PROFILE = np.array([
        0.547, 0.678, 0.693, 0.629, 0.573, 0.548, 0.668, 0.673, 0.728,
        0.725, 0.761, 0.784, 0.804, 0.941, 1.072, 1.161, 0.887, 0.731,
        0.646, 0.568, 0.479, 0.485, 0.499, 0.584])

    def __init__(self, forecaster, n_buildings, n_scenarios=5, sim_start=0,
                 action_smooth_weight=0.1, reopt_interval=8, peak_penalty_weight=0.0,
                 horizon_discount=1.0, noise_profile=None, random_seed=2022,
                 terminal_soc_weight=0.0):
        self.forecaster = forecaster
        self.n_buildings = n_buildings
        self.n_scenarios = n_scenarios
        self.sim_start = sim_start  # offset into price/carbon arrays
        self.action_smooth_weight = action_smooth_weight
        self.reopt_interval = reopt_interval  # re-optimize every N hours
        self.peak_penalty_weight = peak_penalty_weight
        self.horizon_discount = horizon_discount  # gamma for exp decay on objective
        self.noise_profile = noise_profile  # per-horizon noise std (T,), None = flat 0.3
        # Credit for energy left in the battery at the end of the planning
        # horizon, valued at the horizon-average price/carbon rate (x eta for
        # discharge losses). 0.0 = no terminal value (myopic horizon end).
        self.terminal_soc_weight = terminal_soc_weight
        self.horizon = 24  # planning horizon (can be reduced)
        self.historical_peak = 0.0  # running max district consumption (normalized)
        self.rng = np.random.RandomState(random_seed)  # reproducible scenario generation

        # Battery parameters
        self.capacity = 6.4
        self.max_power = 5.0
        self.eta = 0.912
        self.max_act = self.max_power / self.capacity  # ~0.78

        # Price/carbon data (loaded from dataset)
        self.price = None
        self.carbon = None

        # Current plan and last executed action (for continuity)
        self.plan = np.zeros((n_buildings, 24))  # always 24, padded with 0
        self.last_action = np.zeros(n_buildings)  # last executed action per building
        self.global_step = 0

    def reset(self, dataset_name='citylearn_challenge_2022_phase_all'):
        """Load price/carbon data and reset state."""
        schema = DataSet.get_schema(dataset_name)
        root_dir = schema['root_directory']

        self.price = pd.read_csv(f'{root_dir}/pricing.csv')['electricity_pricing'].values
        self.carbon = pd.read_csv(f'{root_dir}/carbon_intensity.csv')['carbon_intensity'].values

        self.plan = np.zeros((self.n_buildings, 24))
        self.last_action = np.zeros(self.n_buildings)
        self.global_step = 0
        self.historical_peak = 0.0
        self.forecaster.reset()

    def _solve_mpc(self, init_soc, load_pred, solar_pred, price_24, carbon_24,
                    prev_action=None, historical_peak=0.0):
        """
        Solve 24h MPC with stochastic scenarios.

        Args:
            init_soc: array (N,) current SOC
            load_pred: array (N, 24) predicted load in kW
            solar_pred: array (N, 24) predicted solar in kW
            price_24: array (24,) electricity prices
            carbon_24: array (24,) carbon intensity
            prev_action: array (N,) last executed action per building (for smoothness)

        Returns:
            actions: array (N, 24) or None if solver fails
        """
        N = self.n_buildings
        T = self.horizon
        S = self.n_scenarios
        eta = self.eta

        # Slice inputs to planning horizon
        load_pred = load_pred[:, :T]
        solar_pred = solar_pred[:, :T]
        price_24 = price_24[:T]
        carbon_24 = carbon_24[:T]

        # Convert predictions to normalized net load: (load - solar) / capacity
        net_load_base = (load_pred - solar_pred) / self.capacity  # (N, T)

        # Per-horizon noise std: adaptive profile or flat 0.3
        if self.noise_profile is not None:
            noise_std_h = self.noise_profile  # (24,)
        else:
            noise_std_h = np.full(T, 0.3)

        # Generate scenarios with horizon-dependent Gaussian noise
        net_load_scenarios = np.zeros((S, N, T))
        for s in range(S):
            noise = self.rng.normal(0, 1, (N, T)) * noise_std_h[np.newaxis, :]
            net_load_scenarios[s] = net_load_base * (1 + noise)

        # Horizon discount weights: gamma^t
        if self.horizon_discount < 1.0:
            tw = np.array([self.horizon_discount ** t for t in range(T)])
        else:
            tw = np.ones(T)

        # Compute baselines from mean scenario
        mean_scenario = net_load_scenarios.mean(axis=0)
        district_mean = mean_scenario.sum(axis=0)
        price_base = np.sum(np.maximum(district_mean, 0) * price_24) + 0.01
        emission_base = np.sum(np.maximum(mean_scenario, 0).sum(axis=0) * carbon_24) + 0.01
        ramping_base = np.sum(np.abs(np.diff(district_mean))) + 0.01

        # Decision variables (shared across scenarios)
        x = cp.Variable((N, T), nonneg=True)   # charge
        y = cp.Variable((N, T), nonpos=True)    # discharge

        constraints = [x <= self.max_act, y >= -self.max_act]

        obj = 0

        for s in range(S):
            L = net_load_scenarios[s]

            # SOC
            soc = cp.Variable((N, T))
            constraints += [soc >= 0, soc <= 1]
            for i in range(N):
                constraints.append(soc[i, 0] == init_soc[i] + x[i, 0] * eta + y[i, 0] / eta)
                for t in range(1, T):
                    constraints.append(soc[i, t] == soc[i, t - 1] + x[i, t] * eta + y[i, t] / eta)

            # District positive consumption (price term, horizon-discounted)
            for t in range(T):
                district_t = cp.sum(L[:, t] + x[:, t] + y[:, t])
                obj += tw[t] * price_24[t] * cp.pos(district_t) / price_base

            # Per-building positive consumption (emission term, horizon-discounted)
            for t in range(T):
                for b in range(N):
                    net_b = L[b, t] + x[b, t] + y[b, t]
                    obj += tw[t] * carbon_24[t] * cp.pos(net_b) / emission_base

            # Ramping (horizon-discounted)
            for t in range(T - 1):
                d_t = cp.sum(L[:, t] + x[:, t] + y[:, t])
                d_t1 = cp.sum(L[:, t + 1] + x[:, t + 1] + y[:, t + 1])
                obj += 0.5 * tw[t] * cp.abs(d_t1 - d_t) / ramping_base

            # Load factor approximation: penalize peak - average
            district_vars = [cp.sum(L[:, t] + x[:, t] + y[:, t]) for t in range(T)]
            peak = cp.Variable()
            for t in range(T):
                constraints.append(peak >= district_vars[t])
            avg = cp.sum(cp.hstack(district_vars)) / T
            obj += 0.5 * (peak - avg) / (np.max(district_mean) - np.mean(district_mean) + 0.01)

            # Peak-aware penalty: penalize exceeding historical district peak
            if self.peak_penalty_weight > 0 and historical_peak > 0:
                peak_excess = cp.pos(peak - historical_peak)
                normalize_base = np.max(district_mean) + 0.01
                obj += self.peak_penalty_weight * peak_excess / normalize_base

            # Terminal SOC value: credit stored energy at the horizon end at the
            # horizon-average price/carbon rate (scaled by eta for discharge loss)
            if self.terminal_soc_weight > 0:
                credit_rate = eta * (np.mean(price_24) / price_base
                                     + np.mean(carbon_24) / emission_base)
                obj -= self.terminal_soc_weight * credit_rate * cp.sum(soc[:, T - 1])

        obj = obj / S

        # Per-building action smoothness penalty (scenario-independent)
        if self.action_smooth_weight > 0:
            for b in range(N):
                # Penalize action changes between consecutive timesteps
                for t in range(T - 1):
                    obj += self.action_smooth_weight * cp.abs(
                        (x[b, t + 1] + y[b, t + 1]) - (x[b, t] + y[b, t]))
                # Penalize jump from previous plan's last action to new plan's first action
                if prev_action is not None:
                    obj += self.action_smooth_weight * cp.abs(
                        (x[b, 0] + y[b, 0]) - prev_action[b])

        problem = cp.Problem(cp.Minimize(obj), constraints)

        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                result = x.value + y.value  # (N, T)
                if T < 24:
                    # Pad to 24 columns with zeros for consistent plan indexing
                    result = np.pad(result, ((0, 0), (0, 24 - T)), constant_values=0)
                return result
        except Exception as e:
            pass  # Solver failure — return None

        return None

    def compute_action(self, observations):
        """
        Compute battery actions for all buildings.

        Args:
            observations: list of observation arrays (one per building)
                Expected indices: [2]=hour, [20]=load, [21]=solar, [22]=soc

        Returns:
            actions: list of arrays, one per building
        """
        N = len(observations)
        hour = int(observations[0][2]) % 24

        # Update forecaster with current observations and track district peak
        district_now = 0.0
        for b in range(N):
            load = observations[b][20]  # raw kW
            solar = observations[b][21]  # raw kW
            self.forecaster.update(b, load, solar, observation=observations[b])
            district_now += (load - solar) / self.capacity
        self.historical_peak = max(self.historical_peak, district_now)

        # Re-optimize at configured interval (after warmup)
        should_reopt = (self.global_step % self.reopt_interval == 0)

        if should_reopt and self.global_step >= 24:
            # Get forecasts for all buildings
            load_pred = np.zeros((N, 24))
            solar_pred = np.zeros((N, 24))
            for b in range(N):
                load_pred[b], solar_pred[b] = self.forecaster.predict(b, self.global_step, hour)

            # Get price/carbon for next 24 hours (using global index)
            g = self.sim_start + self.global_step
            price_24 = self.price[g + 1:g + 25] if g + 25 <= len(self.price) else np.pad(
                self.price[g + 1:], (0, max(0, 24 - len(self.price) + g + 1)), constant_values=0.25)
            carbon_24 = self.carbon[g + 1:g + 25] if g + 25 <= len(self.carbon) else np.pad(
                self.carbon[g + 1:], (0, max(0, 24 - len(self.carbon) + g + 1)), constant_values=0.4)

            if len(price_24) < 24:
                price_24 = np.pad(price_24, (0, 24 - len(price_24)), constant_values=0.25)
            if len(carbon_24) < 24:
                carbon_24 = np.pad(carbon_24, (0, 24 - len(carbon_24)), constant_values=0.4)

            init_soc = [observations[b][22] for b in range(N)]

            result = self._solve_mpc(init_soc, load_pred, solar_pred, price_24, carbon_24,
                                      prev_action=self.last_action,
                                      historical_peak=self.historical_peak)
            if result is not None:
                self.plan = result

        # Extract action from current plan
        step_in_plan = self.global_step % self.reopt_interval
        actions = []
        for b in range(N):
            if step_in_plan < 24:
                action = self.plan[b, step_in_plan]
            else:
                action = 0.0
            action = np.clip(action, -self.max_act, self.max_act)
            self.last_action[b] = action
            actions.append(np.array([action], dtype=np.float32))

        self.global_step += 1
        return actions


def run_mpc_phase(buildings, sim_start, sim_end, forecaster, n_scenarios=5,
                   action_smooth_weight=0.1, reopt_interval=8, peak_penalty_weight=0.0,
                   horizon_discount=1.0, noise_profile=None, horizon=24, verbose=True,
                   random_seed=2022, terminal_soc_weight=0.0):
    """
    Run Online MPC on a specific phase.

    Args:
        buildings: list of building names
        sim_start, sim_end: timestep range
        forecaster: BaseForecaster instance
        n_scenarios: number of stochastic scenarios
        action_smooth_weight: weight for per-building action smoothness penalty
        reopt_interval: re-optimize every N hours (default 8)
        verbose: print progress

    Returns:
        metrics: dict with score, cost, emissions, grid, etc.
    """
    import time
    from mpcgap.environment import create_env, evaluate_env

    N = len(buildings)
    config = {
        'dataset_name': 'citylearn_challenge_2022_phase_all',
        'buildings': buildings,
        'sim_start': sim_start,
        'sim_end': sim_end,
        'central_agent': False,
    }

    env = create_env(config)
    agent = OnlineMPC(forecaster, N, n_scenarios=n_scenarios, sim_start=sim_start,
                      action_smooth_weight=action_smooth_weight,
                      reopt_interval=reopt_interval,
                      peak_penalty_weight=peak_penalty_weight,
                      horizon_discount=horizon_discount,
                      noise_profile=noise_profile,
                      random_seed=random_seed,
                      terminal_soc_weight=terminal_soc_weight)
    agent.horizon = horizon
    agent.reset()

    obs, _ = env.reset()
    step = 0
    total = sim_end - sim_start + 1
    start_time = time.time()

    while not env.terminated:
        actions = agent.compute_action(obs)
        obs, _, _, _, _ = env.step(actions)
        step += 1
        if verbose and step % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step}/{total} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    metrics = evaluate_env(env)

    if verbose:
        print(f"  Done in {elapsed:.1f}s. Score={metrics['score']:.4f}")

    return metrics
