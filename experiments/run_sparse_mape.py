"""Phase-2 forecast accuracy of the sparse-lag LightGBM model vs the full model.

Usage (from the repository root): python -m experiments.run_sparse_mape
Output: results/verify_sparse_mape.json
"""
import json
from mpcgap.data import load_building_data
from experiments.run_forecast_metrics import compute_forecast_metrics
from mpcgap.lgb_forecaster import LGBForecaster
from mpcgap.evaluate_full import PHASES

ph = PHASES[1]; start, end = ph['sim_start'], ph['sim_end']; N = 5
load_data, solar_data = load_building_data()
load_p2 = {i: load_data[i][start:end] for i in range(N)}
solar_p2 = {i: solar_data[i][start:end] for i in range(N)}
names = [f'Building_{i}' for i in range(1, 6)]
out = {}
for tag, path in [('full', 'models/lgb_models.pkl'), ('sparse', 'models/lgb_models_sparse.pkl')]:
    fc = LGBForecaster(N, sim_start=start, building_names=names, model_path=path)
    out[tag] = compute_forecast_metrics(fc, load_p2, solar_p2, N)
    print(tag, {k: round(float(v), 3) for k, v in out[tag].items()}, flush=True)
json.dump(out, open('results/verify_sparse_mape.json', 'w'), indent=2)
