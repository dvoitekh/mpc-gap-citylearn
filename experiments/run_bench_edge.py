"""
Single-core timing benchmark of the deployable pipeline (edge resource table).

Measures, on the current machine:
  1. Stochastic MPC solve time (problem construction + CLARABEL solve) for
     I = 1 / 5 / 15 buildings, S = 5 scenarios, H = 24, via the real
     OnlineMPC._solve_mpc code path.
  2. LightGBM 24 h forecast latency per building (48 direct estimator calls,
     the same path as LGBForecaster.predict).
  3. OLS online-correction update time.
  4. Peak resident memory after model load.

Everything runs single-threaded (OMP_NUM_THREADS=1). The benchmark stubs the
CityLearn import of online_mpc.py, so on the target device only
numpy/pandas/cvxpy/clarabel/lightgbm/scikit-learn/joblib are required.

Usage (from the repository root):
    python -m experiments.run_bench_edge --prepare   # once, needs CityLearn: writes experiments/bench_data.npz
    python -m experiments.run_bench_edge             # prints a JSON report to stdout

To benchmark another device, copy this file, experiments/bench_data.npz, the
mpcgap/ package and models/lgb_models.pkl to it, keeping the same layout.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import types
import platform
import resource

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def make_stub_modules():
    """Stub citylearn.data so that mpcgap.online_mpc imports without CityLearn."""
    citylearn = types.ModuleType("citylearn")
    citylearn_data = types.ModuleType("citylearn.data")

    class DataSet:  # never used: we call _solve_mpc directly, not reset()
        @staticmethod
        def get_schema(name):
            raise RuntimeError("not available in benchmark")

    citylearn_data.DataSet = DataSet
    citylearn.data = citylearn_data
    sys.modules["citylearn"] = citylearn
    sys.modules["citylearn.data"] = citylearn_data


def bench_mpc(data, repeats=3):
    make_stub_modules()
    sys.path.insert(0, os.path.join(HERE, ".."))
    from mpcgap.online_mpc import OnlineMPC

    results = {}
    rng = np.random.RandomState(0)
    for n_b in (1, 5, 15):
        mpc = OnlineMPC(forecaster=None, n_buildings=n_b, n_scenarios=5,
                        action_smooth_weight=0.1, random_seed=2022)
        load = data["load"][:n_b]
        solar = data["solar"][:n_b]
        init_soc = np.full(n_b, 0.5)
        prev_action = np.zeros(n_b)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            actions = mpc._solve_mpc(init_soc, load, solar,
                                     data["price"], data["carbon"],
                                     prev_action=prev_action)
            times.append(time.perf_counter() - t0)
            assert actions is not None, "solver failed"
        results[f"I={n_b}"] = {
            "times_s": [round(t, 3) for t in times],
            "median_s": round(float(np.median(times)), 3),
        }
    return results


def bench_lgb(model_path, repeats=20):
    import joblib

    t0 = time.perf_counter()
    model_data = joblib.load(model_path)
    load_time = time.perf_counter() - t0
    load_model = model_data["load_model"]
    solar_model = model_data["solar_model"]
    n_feat = len(model_data["feature_columns"])

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_mb / (1024 * 1024 if sys.platform == "darwin" else 1024)

    rng = np.random.RandomState(0)
    feat = rng.uniform(0.0, 2.0, size=(1, n_feat))
    # warmup
    for est in load_model.estimators_[:2]:
        est.predict(feat)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.array([est.predict(feat)[0] for est in load_model.estimators_])
        np.array([est.predict(feat)[0] for est in solar_model.estimators_])
        times.append(time.perf_counter() - t0)
    return {
        "model_load_s": round(load_time, 2),
        "rss_after_load_mb": round(rss_mb, 1),
        "n_features": n_feat,
        "predict_per_building_ms": {
            "median": round(float(np.median(times)) * 1000, 1),
            "min": round(float(np.min(times)) * 1000, 1),
            "max": round(float(np.max(times)) * 1000, 1),
        },
    }


def bench_ols(repeats=1000):
    rng = np.random.RandomState(0)
    real = rng.uniform(0, 3, 24)
    pred = rng.uniform(0, 3, 24)
    pr_sum = np.zeros(24)
    sq_sum = np.zeros(24)
    t0 = time.perf_counter()
    for _ in range(repeats):
        pr = real * pred
        p2 = pred ** 2
        pr_sum += pr
        sq_sum += p2
        denom = np.where(sq_sum > 1e-6, sq_sum, 1.0)
        np.clip(pr_sum / denom, 0.5, 2.0)
    return {"update_us": round((time.perf_counter() - t0) / repeats * 1e6, 1)}


def prepare_bench_data(path):
    """Extract one 24 h window (first day of Phase 3, its 15 buildings) from the
    CityLearn dataset so that the benchmark itself does not need CityLearn."""
    import pandas as pd
    sys.path.insert(0, os.path.join(HERE, ".."))
    from citylearn.data import DataSet
    from mpcgap.evaluate_full import PHASES
    from mpcgap.data import load_building_data

    load_data, solar_data = load_building_data()
    phase = PHASES[2]
    t0 = phase["sim_start"]
    idx = [int(b.split("_")[1]) - 1 for b in phase["buildings"]]
    load = np.array([load_data[i][t0:t0 + 24] for i in idx])
    solar = np.array([solar_data[i][t0:t0 + 24] for i in idx])
    root = DataSet.get_schema("citylearn_challenge_2022_phase_all")["root_directory"]
    price = pd.read_csv(f"{root}/pricing.csv")["electricity_pricing"].values[t0:t0 + 24]
    carbon = pd.read_csv(f"{root}/carbon_intensity.csv")["carbon_intensity"].values[t0:t0 + 24]
    np.savez(path, load=load, solar=solar, price=price, carbon=carbon)
    print(f"wrote {path}: load/solar {load.shape}, price/carbon {price.shape}")


def main():
    data_path = os.path.join(HERE, "bench_data.npz")
    if "--prepare" in sys.argv:
        prepare_bench_data(data_path)
        return
    if not os.path.exists(data_path):
        sys.exit("bench_data.npz not found; run with --prepare first")
    data = np.load(data_path)
    report = {
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        import cvxpy
        report["cvxpy"] = cvxpy.__version__
        import clarabel
        report["clarabel"] = getattr(clarabel, "__version__", "?")
        import lightgbm
        report["lightgbm"] = lightgbm.__version__
        import sklearn
        report["sklearn"] = sklearn.__version__
        report["numpy"] = np.__version__
    except Exception as e:
        report["version_probe_error"] = str(e)

    report["lgb"] = bench_lgb(os.path.join(HERE, "..", "models", "lgb_models.pkl"))
    report["ols"] = bench_ols()
    report["mpc"] = bench_mpc(data)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
