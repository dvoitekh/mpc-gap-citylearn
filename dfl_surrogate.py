"""
Decision-focused learning via a cost surrogate.

1. Generate (forecast, MPC cost) pairs by running the MPC on perturbed
   LightGBM forecasts (fast parametric LP).
2. Train a surrogate network: forecast -> predicted MPC cost.
3. Pre-train a neural forecaster with MSE, then fine-tune it against the
   surrogate loss (with an MSE anchor).

The experiment tests whether training a forecaster on decision cost rather
than accuracy improves MPC performance. On this benchmark it does not close
the gap to LightGBM; see results/dfl_eval_results.txt.
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp
from citylearn.data import DataSet

from lgb_feature_engineering import FeatureBuilder, OnlineFeatureBuilder, get_feature_columns, get_target_columns
from lgb_train import build_training_data
from evaluate_full import PHASES
from online_mpc import run_mpc_phase
from forecasters import BaseForecaster

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================
# Part 1: Fast LP Solver + Cost Evaluator
# =============================================================

class FastBatteryLP:
    """
    Fast single-building LP for generating (forecast, cost) pairs.

    Uses CVXPY parameters to compile the problem once, then only
    swap data values on each solve. Solves per-building (N=1) for speed.
    """

    def __init__(self, capacity=6.4, efficiency=0.912, max_power=5.0):
        self.capacity = capacity
        self.eta = efficiency
        self.max_act = max_power / capacity
        self._problem = None
        self._build_parametric_problem()

    def _build_parametric_problem(self):
        """Build CVXPY problem once with Parameters."""
        T = 24

        # Parameters (updated per solve)
        self.p_net = cp.Parameter(T)          # net load for 1 building
        self.p_price = cp.Parameter(T, nonneg=True)
        self.p_carbon = cp.Parameter(T, nonneg=True)
        self.p_soc0 = cp.Parameter(nonneg=True)
        # Baseline normalizers (positive scalars)
        self.p_price_base = cp.Parameter(pos=True)
        self.p_emission_base = cp.Parameter(pos=True)
        self.p_ramping_base = cp.Parameter(pos=True)
        self.p_lf_base = cp.Parameter(pos=True)

        # Variables
        x = cp.Variable(T, nonneg=True)   # charge
        y = cp.Variable(T, nonpos=True)    # discharge
        peak = cp.Variable()
        self._x = x
        self._y = y

        constraints = [x <= self.max_act, y >= -self.max_act]

        # SOC constraints (cumulative)
        for t in range(T):
            # SOC after step t
            charge_sum = cp.sum(x[:t+1]) * self.eta
            discharge_sum = cp.sum(y[:t+1]) / self.eta
            soc_t = self.p_soc0 + charge_sum + discharge_sum
            constraints += [soc_t >= 0, soc_t <= 1]

        # Net consumption with battery: d_t = net_t + x_t + y_t
        d = self.p_net + x + y  # (T,)

        # Objective terms
        # 1. Price cost
        price_cost = cp.sum(cp.multiply(self.p_price, cp.pos(d))) / self.p_price_base

        # 2. Emission cost (same as price for single building)
        emission_cost = cp.sum(cp.multiply(self.p_carbon, cp.pos(d))) / self.p_emission_base

        # 3. Ramping
        ramping = cp.sum(cp.abs(cp.diff(d))) / self.p_ramping_base

        # 4. Peak - avg (load factor proxy)
        for t in range(T):
            constraints.append(peak >= d[t])
        avg = cp.sum(d) / T
        lf_cost = (peak - avg) / self.p_lf_base

        obj = price_cost + emission_cost + 0.5 * ramping + 0.5 * lf_cost

        self._problem = cp.Problem(cp.Minimize(obj), constraints)

    def solve_single(self, net_load_1, price_24, carbon_24, init_soc=0.5):
        """
        Solve 24h LP for a single building.

        Args:
            net_load_1: (24,) net load for one building
            price_24: (24,) prices
            carbon_24: (24,) carbon intensity
            init_soc: scalar SOC

        Returns:
            actions: (24,) or None if fails
        """
        # Compute baselines
        price_base = np.sum(np.maximum(net_load_1, 0) * price_24) + 0.01
        emission_base = np.sum(np.maximum(net_load_1, 0) * carbon_24) + 0.01
        ramping_base = np.sum(np.abs(np.diff(net_load_1))) + 0.01
        lf_base = np.max(net_load_1) - np.mean(net_load_1) + 0.01

        # Set parameters
        self.p_net.value = net_load_1
        self.p_price.value = price_24
        self.p_carbon.value = carbon_24
        self.p_soc0.value = init_soc
        self.p_price_base.value = price_base
        self.p_emission_base.value = emission_base
        self.p_ramping_base.value = ramping_base
        self.p_lf_base.value = lf_base

        try:
            self._problem.solve(solver=cp.CLARABEL, verbose=False, warm_start=True)
            if self._problem.status in ['optimal', 'optimal_inaccurate']:
                return self._x.value + self._y.value
        except Exception:
            pass
        return None

    def solve(self, net_load, price_24, carbon_24, init_soc=None):
        """
        Solve for N buildings (calls solve_single per building).

        Args:
            net_load: (N, 24) predicted net load
            price_24: (24,) prices
            carbon_24: (24,) carbon intensity
            init_soc: (N,) initial SOC, default 0.5

        Returns:
            actions: (N, 24) or None if any building fails
        """
        N = net_load.shape[0]
        if init_soc is None:
            init_soc = np.full(N, 0.5)

        actions = np.zeros_like(net_load)
        for i in range(N):
            a = self.solve_single(net_load[i], price_24, carbon_24, init_soc[i])
            if a is None:
                return None
            actions[i] = a
        return actions


def compute_actual_cost(actions, true_net_load, price_24, carbon_24):
    """
    Compute actual MPC cost when applying actions to true data.

    This is the "task loss" — the cost that actually matters.

    Args:
        actions: (N, 24) battery actions
        true_net_load: (N, 24) true (load-solar)/capacity
        price_24: (24,) prices
        carbon_24: (24,) carbon intensity

    Returns:
        cost: scalar, lower is better
    """
    N, T = true_net_load.shape

    # District consumption with battery actions
    net_with_battery = true_net_load + actions  # (N, T)
    district = net_with_battery.sum(axis=0)  # (T,)

    # Baselines (without battery)
    district_base = true_net_load.sum(axis=0)
    price_base = np.sum(np.maximum(district_base, 0) * price_24) + 0.01
    emission_base = np.sum(np.maximum(true_net_load, 0).sum(axis=0) * carbon_24) + 0.01
    ramping_base = np.sum(np.abs(np.diff(district_base))) + 0.01

    # Price cost
    price_cost = np.sum(np.maximum(district, 0) * price_24) / price_base

    # Emission cost
    emission_cost = np.sum(np.maximum(net_with_battery, 0).sum(axis=0) * carbon_24) / emission_base

    # Ramping
    ramping = np.sum(np.abs(np.diff(district))) / ramping_base

    # Peak - avg (proxy for load factor)
    peak = np.max(district)
    avg = np.mean(district)
    lf_base = np.max(district_base) - np.mean(district_base) + 0.01
    load_factor = (peak - avg) / lf_base

    # Combined (same weights as MPC objective)
    total = price_cost + emission_cost + 0.5 * ramping + 0.5 * load_factor
    return total


# =============================================================
# Part 2: Data Generation
# =============================================================

def generate_surrogate_data(n_windows=200, n_perturbations=30, n_buildings=5,
                             phase_start=0, phase_end=24*240, seed=42):
    """
    Generate (forecast, cost) pairs for surrogate training.

    For each time window:
    1. Get true load/solar for next 24h
    2. Generate perturbed forecasts
    3. Solve LP with each forecast
    4. Evaluate actions against true data
    5. Record (forecast, context, cost)

    Returns:
        forecasts: (M, 48) — 24h load + 24h solar predictions (normalized)
        contexts: (M, 49) — price(24) + carbon(24) + soc(1)
        costs: (M,) — actual MPC cost
        true_data: (M, 48) — true load + solar (for reference)
    """
    np.random.seed(seed)

    # Load data
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')
    root_dir = schema['root_directory']

    buildings = [f'Building_{i}' for i in range(1, n_buildings + 1)]
    pv_caps = {}
    for b in buildings:
        if b in schema['buildings']:
            pv_caps[b] = schema['buildings'][b].get('pv', {}).get('nominal_power', 4.0)

    capacity = 6.4
    load_all = {}
    solar_all = {}
    for i, b in enumerate(buildings):
        df = pd.read_csv(f'{root_dir}/{b}.csv')
        load_all[i] = df['non_shiftable_load'].values
        solar_all[i] = df['solar_generation'].values * pv_caps.get(b, 4.0) / 1000.0

    price = pd.read_csv(f'{root_dir}/pricing.csv')['electricity_pricing'].values
    carbon = pd.read_csv(f'{root_dir}/carbon_intensity.csv')['carbon_intensity'].values

    solver = FastBatteryLP()

    # Sample time windows evenly
    valid_range = range(phase_start + 48, phase_end - 24)  # skip warmup, leave room for 24h
    window_indices = np.linspace(0, len(valid_range) - 1, n_windows, dtype=int)
    time_windows = [valid_range[i] for i in window_indices]

    all_forecasts = []
    all_contexts = []
    all_costs = []
    all_true = []

    n_failed = 0
    n_success = 0

    print(f"Generating surrogate data: {n_windows} windows × {n_perturbations} perturbations")

    for wi, t in enumerate(time_windows):
        if wi % 20 == 0:
            print(f"  Window {wi}/{n_windows} (t={t})...")

        # True data for next 24h (all buildings, aggregated as mean for features)
        true_load = np.zeros((n_buildings, 24))
        true_solar = np.zeros((n_buildings, 24))
        for b in range(n_buildings):
            true_load[b] = load_all[b][t+1:t+25]
            true_solar[b] = solar_all[b][t+1:t+25]

        true_net = (true_load - true_solar) / capacity  # (N, 24)

        # Context
        p24 = price[t+1:t+25]
        c24 = carbon[t+1:t+25]
        if len(p24) < 24:
            continue

        # Mean load/solar across buildings (for 48-dim feature)
        true_load_mean = true_load.mean(axis=0)  # (24,)
        true_solar_mean = true_solar.mean(axis=0)  # (24,)
        true_48 = np.concatenate([true_load_mean, true_solar_mean])

        # Also solve LP with perfect forecast (for regret computation)
        init_soc = np.full(n_buildings, 0.5)
        perfect_actions = solver.solve(true_net, p24, c24, init_soc)
        if perfect_actions is None:
            n_failed += 1
            continue
        perfect_cost = compute_actual_cost(perfect_actions, true_net, p24, c24)

        # Generate perturbations
        for _ in range(n_perturbations):
            # Random perturbation: multiplicative noise + additive bias
            noise_scale = np.random.uniform(0.1, 0.5)
            bias = np.random.uniform(-0.3, 0.3)

            pred_load = true_load * (1 + np.random.normal(bias, noise_scale, true_load.shape))
            pred_solar = true_solar * (1 + np.random.normal(bias * 0.5, noise_scale, true_solar.shape))
            pred_load = np.maximum(pred_load, 0)
            pred_solar = np.maximum(pred_solar, 0)

            pred_net = (pred_load - pred_solar) / capacity

            # Solve LP with perturbed forecast
            actions = solver.solve(pred_net, p24, c24, init_soc)
            if actions is None:
                n_failed += 1
                continue

            # Evaluate actions on TRUE data
            actual_cost = compute_actual_cost(actions, true_net, p24, c24)

            # Store
            pred_load_mean = pred_load.mean(axis=0)
            pred_solar_mean = pred_solar.mean(axis=0)
            forecast_48 = np.concatenate([pred_load_mean, pred_solar_mean])

            context = np.concatenate([p24, c24, [0.5]])  # price + carbon + soc

            all_forecasts.append(forecast_48)
            all_contexts.append(context)
            all_costs.append(actual_cost)
            all_true.append(true_48)
            n_success += 1

    print(f"  Generated {n_success} samples, {n_failed} solver failures")

    return (np.array(all_forecasts), np.array(all_contexts),
            np.array(all_costs), np.array(all_true))


# =============================================================
# Part 3: Surrogate Neural Network
# =============================================================

class SurrogateNet(nn.Module):
    """
    Surrogate model: (forecast, context) → predicted MPC cost.

    Input: forecast (48) + context (49) = 97 dims
    Output: scalar cost prediction
    """

    def __init__(self, input_dim=97, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_surrogate(forecasts, contexts, costs, epochs=200, lr=1e-3,
                     val_frac=0.2, batch_size=256):
    """Train surrogate NN on (forecast, context) → cost data."""
    # Normalize inputs
    f_mean, f_std = forecasts.mean(0), forecasts.std(0) + 1e-8
    c_mean, c_std = contexts.mean(0), contexts.std(0) + 1e-8
    cost_mean, cost_std = costs.mean(), costs.std() + 1e-8

    forecasts_n = (forecasts - f_mean) / f_std
    contexts_n = (contexts - c_mean) / c_std
    costs_n = (costs - cost_mean) / cost_std

    X = np.concatenate([forecasts_n, contexts_n], axis=1)

    # Train/val split
    n = len(X)
    n_val = int(n * val_frac)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    X_train = torch.FloatTensor(X[train_idx]).to(DEVICE)
    y_train = torch.FloatTensor(costs_n[train_idx]).to(DEVICE)
    X_val = torch.FloatTensor(X[val_idx]).to(DEVICE)
    y_val = torch.FloatTensor(costs_n[val_idx]).to(DEVICE)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SurrogateNet(input_dim=X.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.MSELoss()(val_pred, y_val).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            # Compute R² on val set
            with torch.no_grad():
                pred_v = model(X_val).cpu().numpy()
                true_v = y_val.cpu().numpy()
                ss_res = np.sum((true_v - pred_v) ** 2)
                ss_tot = np.sum((true_v - true_v.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} R²={r2:.4f}")

    model.load_state_dict(best_state)

    # Final R²
    model.eval()
    with torch.no_grad():
        pred_v = model(X_val).cpu().numpy()
        true_v = y_val.cpu().numpy()
        ss_res = np.sum((true_v - pred_v) ** 2)
        ss_tot = np.sum((true_v - true_v.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
    print(f"  Final surrogate R²: {r2:.4f}")

    # Return model + normalization stats
    norm_stats = {
        'f_mean': f_mean, 'f_std': f_std,
        'c_mean': c_mean, 'c_std': c_std,
        'cost_mean': cost_mean, 'cost_std': cost_std,
    }
    return model, norm_stats


# =============================================================
# Part 4: NN Forecaster with Surrogate Loss
# =============================================================

class NNForecaster(nn.Module):
    """
    Neural network forecaster: features → (load_24, solar_24).

    Architecture matches LGB's feature set but is differentiable.
    """

    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=48):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return torch.relu(out)  # Non-negative predictions


class DFLForecasterWrapper(BaseForecaster):
    """
    Wraps NNForecaster as a BaseForecaster for MPC evaluation.
    """

    def __init__(self, model, feature_builder, norm_stats, n_buildings):
        self.model = model
        self.feature_builder = feature_builder
        self.norm_stats = norm_stats
        self.n_buildings = n_buildings

    def predict(self, building_idx, current_step, current_hour):
        feat_dict = self.feature_builder.build_features(
            building_idx, current_step, current_hour
        )
        feat_cols = get_feature_columns()
        feat_array = np.array([[feat_dict[c] for c in feat_cols]])

        # Normalize
        feat_n = (feat_array - self.norm_stats['x_mean']) / self.norm_stats['x_std']
        feat_t = torch.FloatTensor(feat_n).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(feat_t).cpu().numpy()[0]

        # Denormalize
        pred = pred * self.norm_stats['y_std'] + self.norm_stats['y_mean']
        pred = np.maximum(pred, 0)

        load_pred = pred[:24]
        solar_pred = pred[24:]

        # Night solar constraint
        for h in range(24):
            target_hour = (current_hour + h + 1) % 24
            if target_hour < 7 or target_hour >= 20:
                solar_pred[h] = 0.0

        return load_pred, solar_pred

    def update(self, building_idx, load, solar, observation=None):
        self.feature_builder.update(building_idx, load, solar)

    def reset(self):
        self.feature_builder.reset()


def build_nn_training_data():
    """
    Build training data for NN forecaster using existing pipeline.

    Returns features (X) and targets (y = load_24 + solar_24 concatenated).
    """
    X, y_load, y_solar, days = build_training_data()

    # Concatenate load and solar targets into single 48-dim output
    y = np.concatenate([y_load.values, y_solar.values], axis=1)  # (n, 48)

    return X.values, y, days


def train_nn_forecaster_mse(X, y, days, epochs=100, lr=1e-3, batch_size=512):
    """Pre-train NN forecaster with MSE loss."""
    # Split
    train_mask = days <= 300
    val_mask = days > 300

    # Normalize
    x_mean, x_std = X[train_mask].mean(0), X[train_mask].std(0) + 1e-8
    y_mean, y_std = y[train_mask].mean(0), y[train_mask].std(0) + 1e-8

    X_n = (X - x_mean) / x_std
    y_n = (y - y_mean) / y_std

    X_train = torch.FloatTensor(X_n[train_mask]).to(DEVICE)
    y_train = torch.FloatTensor(y_n[train_mask]).to(DEVICE)
    X_val = torch.FloatTensor(X_n[val_mask]).to(DEVICE)
    y_val = torch.FloatTensor(y_n[val_mask]).to(DEVICE)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = NNForecaster(input_dim=X.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.MSELoss()(val_pred, y_val).item()

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            print(f"  MSE Epoch {epoch+1:3d}: val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)

    norm_stats = {
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std,
    }
    return model, norm_stats


def fine_tune_with_surrogate(forecaster_model, surrogate_model, surrogate_norm,
                              X, y, days, forecaster_norm,
                              epochs=50, lr=1e-4, batch_size=256,
                              mse_weight=0.5):
    """
    Fine-tune NN forecaster using surrogate cost as loss.

    Instead of minimizing MSE(pred, true), minimize surrogate(pred).
    The surrogate predicts MPC cost from forecasts.
    """
    # Load price/carbon for context
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')
    root_dir = schema['root_directory']
    price = pd.read_csv(f'{root_dir}/pricing.csv')['electricity_pricing'].values
    carbon = pd.read_csv(f'{root_dir}/carbon_intensity.csv')['carbon_intensity'].values

    train_mask = days <= 300
    X_n = (X - forecaster_norm['x_mean']) / forecaster_norm['x_std']

    X_train = torch.FloatTensor(X_n[train_mask]).to(DEVICE)

    # For each sample, build context (price_24 + carbon_24 + soc)
    # We need the hour of each sample to get the right price slice
    # Use the 'hour' feature (column index depends on feature_cols)
    feature_cols = get_feature_columns()
    hour_idx = feature_cols.index('hour') if 'hour' in feature_cols else 0

    hours = X[train_mask, hour_idx].astype(int)
    day_indices = days[train_mask]

    contexts = []
    for i in range(len(hours)):
        h = hours[i]
        d = day_indices[i]
        t = d * 24 + h
        p24 = price[t+1:t+25] if t + 25 <= len(price) else np.pad(
            price[t+1:], (0, max(0, 24 - len(price) + t + 1)), constant_values=0.25)
        c24 = carbon[t+1:t+25] if t + 25 <= len(carbon) else np.pad(
            carbon[t+1:], (0, max(0, 24 - len(carbon) + t + 1)), constant_values=0.4)
        if len(p24) < 24:
            p24 = np.pad(p24, (0, 24 - len(p24)), constant_values=0.25)
        if len(c24) < 24:
            c24 = np.pad(c24, (0, 24 - len(c24)), constant_values=0.4)
        contexts.append(np.concatenate([p24, c24, [0.5]]))

    contexts = np.array(contexts)
    ctx_n = (contexts - surrogate_norm['c_mean']) / surrogate_norm['c_std']
    ctx_train = torch.FloatTensor(ctx_n).to(DEVICE)

    # Also keep MSE regularization to prevent catastrophic forgetting
    y_n = (y - forecaster_norm['y_mean']) / forecaster_norm['y_std']
    y_train = torch.FloatTensor(y_n[train_mask]).to(DEVICE)

    # Include targets in DataLoader so MSE matches after shuffle
    train_ds = TensorDataset(X_train, ctx_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Freeze surrogate
    surrogate_model.eval()
    for p in surrogate_model.parameters():
        p.requires_grad = False

    # Pre-compute tensors on device
    y_mean_t = torch.FloatTensor(forecaster_norm['y_mean']).to(DEVICE)
    y_std_t = torch.FloatTensor(forecaster_norm['y_std']).to(DEVICE)
    f_mean_t = torch.FloatTensor(surrogate_norm['f_mean']).to(DEVICE)
    f_std_t = torch.FloatTensor(surrogate_norm['f_std']).to(DEVICE)

    # Fine-tune forecaster
    optimizer = optim.Adam(forecaster_model.parameters(), lr=lr)

    # mse_weight controls balance: 0=pure DFL, 1=pure MSE

    for epoch in range(epochs):
        forecaster_model.train()
        total_loss = 0
        total_surr = 0
        total_mse = 0
        n_batches = 0

        for xb, cb, yb in train_dl:
            # Forward through forecaster
            pred_n = forecaster_model(xb)  # normalized predictions

            # Denormalize for surrogate input
            pred_denorm = pred_n * y_std_t + y_mean_t

            # Load and solar predictions (surrogate expects load(24) + solar(24))
            pred_load = pred_denorm[:, :24]  # (batch, 24)
            pred_solar = pred_denorm[:, 24:]  # (batch, 24)

            # Normalize for surrogate
            forecast_48 = torch.cat([pred_load, pred_solar], dim=1)
            forecast_n = (forecast_48 - f_mean_t) / f_std_t

            # Surrogate input
            surr_input = torch.cat([forecast_n, cb], dim=1)
            surr_cost = surrogate_model(surr_input)

            # Surrogate loss: minimize predicted cost
            surr_loss = surr_cost.mean()

            # MSE loss (regularization)
            mse_loss = nn.MSELoss()(pred_n, yb)

            # Combined loss
            loss = (1 - mse_weight) * surr_loss + mse_weight * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forecaster_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_surr += surr_loss.item()
            total_mse += mse_loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  DFL Epoch {epoch+1:3d}: loss={total_loss/n_batches:.4f} "
                  f"surr={total_surr/n_batches:.4f} mse={total_mse/n_batches:.4f}")

    return forecaster_model


# =============================================================
# Part 5: Main Pipeline
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-windows', type=int, default=150,
                        help='Number of time windows for surrogate data')
    parser.add_argument('--n-perturbations', type=int, default=20,
                        help='Perturbations per window')
    parser.add_argument('--surrogate-epochs', type=int, default=200)
    parser.add_argument('--mse-epochs', type=int, default=100)
    parser.add_argument('--dfl-epochs', type=int, default=50)
    parser.add_argument('--mse-weight', type=float, default=0.5,
                        help='MSE regularization weight in DFL (0=pure DFL, 1=pure MSE)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run full MPC evaluation')
    parser.add_argument('--skip-data-gen', action='store_true',
                        help='Skip data generation, load from file')
    parser.add_argument('--eval-only', action='store_true',
                        help='Load saved models and only run MPC evaluation')
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("PHASE 3: DECISION-FOCUSED RETRAINING (DFR)")
    print("=" * 70)

    os.makedirs('models', exist_ok=True)

    # =====================================================
    # Eval-only mode: load saved models and evaluate
    # =====================================================
    if args.eval_only:
        print("\n--- Loading saved models for evaluation ---")
        mse_ckpt = torch.load('models/nn_forecaster_mse.pt', weights_only=False)
        dfl_ckpt = torch.load('models/nn_forecaster_dfl.pt', weights_only=False)

        nn_model = NNForecaster(input_dim=mse_ckpt['input_dim']).to(DEVICE)
        nn_model.load_state_dict(mse_ckpt['model'])
        nn_norm = mse_ckpt['norm']

        dfl_model = NNForecaster(input_dim=dfl_ckpt['input_dim']).to(DEVICE)
        dfl_model.load_state_dict(dfl_ckpt['model'])

        print("\n--- MPC Evaluation ---")
        for model_name, model_obj in [('NN-MSE', nn_model), ('NN-DFL', dfl_model)]:
            print(f"\n  Evaluating {model_name}...")
            weighted_score = 0
            for phase in PHASES:
                N = len(phase['buildings'])
                fb = OnlineFeatureBuilder(N, phase['sim_start'], phase['buildings'])
                forecaster = DFLForecasterWrapper(model_obj, fb, nn_norm, N)
                metrics = run_mpc_phase(
                    phase['buildings'], phase['sim_start'], phase['sim_end'],
                    forecaster, n_scenarios=5, action_smooth_weight=0.1,
                    reopt_interval=8, verbose=True
                )
                weighted_score += phase['weight'] * metrics['score']
                print(f"    {phase['name']}: {metrics['score']:.4f}")
            print(f"  {model_name} Weighted Score: {weighted_score:.4f}")

        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed / 60:.1f} minutes")
        return

    # =====================================================
    # Step 1: Generate surrogate training data
    # =====================================================
    data_path = 'models/surrogate_data.npz'
    if args.skip_data_gen and os.path.exists(data_path):
        print("\n--- Loading surrogate data from cache ---")
        data = np.load(data_path)
        forecasts = data['forecasts']
        contexts = data['contexts']
        costs = data['costs']
        true_data = data['true_data']
    else:
        print("\n--- Step 1: Generating surrogate training data ---")
        forecasts, contexts, costs, true_data = generate_surrogate_data(
            n_windows=args.n_windows,
            n_perturbations=args.n_perturbations,
        )
        np.savez(data_path, forecasts=forecasts, contexts=contexts,
                 costs=costs, true_data=true_data)

    print(f"  Data shape: forecasts={forecasts.shape}, costs={costs.shape}")
    print(f"  Cost range: {costs.min():.3f} - {costs.max():.3f}, "
          f"mean={costs.mean():.3f}, std={costs.std():.3f}")

    # =====================================================
    # Step 2: Train surrogate
    # =====================================================
    print("\n--- Step 2: Training surrogate model ---")
    surrogate, surr_norm = train_surrogate(
        forecasts, contexts, costs, epochs=args.surrogate_epochs
    )

    torch.save({
        'model': surrogate.state_dict(),
        'norm': surr_norm,
    }, 'models/surrogate.pt')
    print("  Surrogate saved to models/surrogate.pt")

    # =====================================================
    # Step 3: Build NN forecaster training data
    # =====================================================
    print("\n--- Step 3: Building NN forecaster training data ---")
    X, y, days = build_nn_training_data()
    print(f"  X: {X.shape}, y: {y.shape}")

    # =====================================================
    # Step 4: Pre-train NN with MSE
    # =====================================================
    print("\n--- Step 4: Pre-training NN forecaster (MSE) ---")
    nn_model, nn_norm = train_nn_forecaster_mse(
        X, y, days, epochs=args.mse_epochs
    )

    torch.save({
        'model': nn_model.state_dict(),
        'norm': nn_norm,
        'input_dim': X.shape[1],
    }, 'models/nn_forecaster_mse.pt')
    print("  MSE model saved")

    # =====================================================
    # Step 5: Fine-tune with surrogate loss (DFL)
    # =====================================================
    print("\n--- Step 5: Fine-tuning with surrogate loss (DFL) ---")

    # Clone MSE model for DFL training
    dfl_model = NNForecaster(input_dim=X.shape[1]).to(DEVICE)
    dfl_model.load_state_dict(nn_model.state_dict())

    dfl_model = fine_tune_with_surrogate(
        dfl_model, surrogate, surr_norm,
        X, y, days, nn_norm,
        epochs=args.dfl_epochs,
        mse_weight=args.mse_weight,
    )

    torch.save({
        'model': dfl_model.state_dict(),
        'norm': nn_norm,
        'input_dim': X.shape[1],
    }, 'models/nn_forecaster_dfl.pt')
    print("  DFL model saved")

    # =====================================================
    # Step 6: Evaluate (optional)
    # =====================================================
    if args.evaluate:
        print("\n--- Step 6: MPC Evaluation ---")

        for model_name, model_obj in [('NN-MSE', nn_model), ('NN-DFL', dfl_model)]:
            print(f"\n  Evaluating {model_name}...")
            weighted_score = 0
            for phase in PHASES:
                N = len(phase['buildings'])
                fb = OnlineFeatureBuilder(N, phase['sim_start'], phase['buildings'])

                forecaster = DFLForecasterWrapper(model_obj, fb, nn_norm, N)

                metrics = run_mpc_phase(
                    phase['buildings'], phase['sim_start'], phase['sim_end'],
                    forecaster, n_scenarios=5, action_smooth_weight=0.1,
                    reopt_interval=8, verbose=True
                )
                weighted_score += phase['weight'] * metrics['score']
                print(f"    {phase['name']}: {metrics['score']:.4f}")

            print(f"  {model_name} Weighted Score: {weighted_score:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
