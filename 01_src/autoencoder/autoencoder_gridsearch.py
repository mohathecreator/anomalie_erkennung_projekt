import sys
from pathlib import Path

PROJECT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from data_utils import read_data, sensor_cols
from autoencoder_model import fit_pipeline
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

train_path = r"02_data\train\train_FD001.txt"
train_df = read_data(train_path)

threshold = 0.01
useful_sensors = [col for col in sensor_cols if train_df[col].std() > threshold]

max_cycles = train_df.groupby("unit_id")["cycles"].max().reset_index()
max_cycles.columns = ["unit_id", "max_cycle"]
train_df = train_df.merge(max_cycles, on="unit_id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycles"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[useful_sensors])
train_df[useful_sensors] = X_scaled

healthy_filter = train_df["cycles"] <= train_df["max_cycle"] * 0.3
healthy_df = train_df[healthy_filter]

param_grid = {
    "window_size": [20, 30],
    "hidden_size": [16, 32],
    "lr": [0.001, 0.0005],
    "epochs": [30, 50],
    "threshold": [0.2, 0.3, 0.4],
}

search_results = []
best_result = None
best_results_df = None

for window_size, hidden_size, lr, epochs, threshold in product(
    param_grid["window_size"],
    param_grid["hidden_size"],
    param_grid["lr"],
    param_grid["epochs"],
    param_grid["threshold"],
):
    candidate_df, _ = fit_pipeline(
        train_df,
        healthy_df,
        useful_sensors,
        window_size,
        hidden_size,
        lr,
        epochs,
        threshold,
    )
    if candidate_df is None:
        continue

    y_true = (candidate_df["RUL"] <= 30).astype(int)
    y_pred = candidate_df["anomaly_flag"].astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    row = {
        "window_size": window_size,
        "hidden_size": hidden_size,
        "lr": lr,
        "epochs": epochs,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    search_results.append(row)

    if best_result is None or (f1, recall, precision) > (
        best_result["f1"],
        best_result["recall"],
        best_result["precision"],
    ):
        best_result = row
        best_results_df = candidate_df

results_overview_df = pd.DataFrame(search_results).sort_values(
    by=["f1", "recall", "precision"], ascending=False
)
print("Top 10 Gridsearch-Ergebnisse:")
print(results_overview_df.head(10).to_string(index=False))

print("\nBeste Parameter:")
print(
    f"window_size={best_result['window_size']}, "
    f"hidden_size={best_result['hidden_size']}, "
    f"lr={best_result['lr']}, "
    f"epochs={best_result['epochs']}, "
    f"threshold={best_result['threshold']}"
)

results_df = best_results_df

first_anomaly = (
    results_df[results_df["anomaly_flag"] == 1]
    .groupby("unit_id")["RUL"]
    .max()
)

print(first_anomaly.describe())

y_true = (results_df["RUL"] <= 30).astype(int)
y_pred = (results_df["anomaly_flag"]).astype(int)

print(f"Precision: {precision_score(y_true, y_pred)}, Recall: {recall_score(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")
