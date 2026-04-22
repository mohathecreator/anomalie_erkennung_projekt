import sys
from pathlib import Path

# Add 01_src to import path so sibling modules like data_utils are resolvable.
PROJECT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from data_utils import read_data, sensor_cols
from iso_forest_model import fit_pipeline
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

train_path = r"02_data\train\train_FD001.txt"
train_df = read_data(train_path)

threshold_std = 0.01
useful_sensors = [col for col in sensor_cols if train_df[col].std() > threshold_std]

max_cycles = train_df.groupby("unit_id")["cycles"].max().reset_index()
max_cycles.columns = ["unit_id", "max_cycle"]
train_df = train_df.merge(max_cycles, on="unit_id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycles"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[useful_sensors])

y_true = (train_df["RUL"] <= 30).astype(int)

param_grid = {
    "n_estimators": [100, 200],
    "contamination": [0.03, 0.05, 0.08],
    "max_samples": [128, 256],
    "window": [10, 15, 20],
    "z_threshold": [-1.5, -2.0, -2.5],
}

results = []
best_result = None
best_df = None

for n_estimators, contamination, max_samples, window, z_threshold in product(
    param_grid["n_estimators"],
    param_grid["contamination"],
    param_grid["max_samples"],
    param_grid["window"],
    param_grid["z_threshold"],
):
    candidate_df, _, _, _ = fit_pipeline(
        train_df,
        X_scaled,
        n_estimators,
        contamination,
        max_samples,
        window,
        z_threshold,
    )

    y_pred = candidate_df["anomaly_flag"].astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    row = {
        "n_estimators": n_estimators,
        "contamination": contamination,
        "max_samples": max_samples,
        "window": window,
        "z_threshold": z_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    results.append(row)

    if best_result is None or (f1, recall, precision) > (best_result["f1"], best_result["recall"], best_result["precision"]):
        best_result = row
        best_df = candidate_df

results_df = pd.DataFrame(results).sort_values(by=["f1", "recall", "precision"], ascending=False)
print("Top 10 Gridsearch-Ergebnisse:")
print(results_df.head(10).to_string(index=False))

print("\nBeste Parameter:")
for k, v in best_result.items():
    print(f"{k}: {v}")
