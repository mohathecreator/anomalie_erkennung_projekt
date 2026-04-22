import sys
sys.path.append(r"C:\Programmieren\anomalie_erkennung_projekt\src")
from data_utils import read_data, sensor_cols
from autoencoder.autoencoder_model import LSTMAutoencoder, create_sequences, run_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

#Change file path, according your own path
train_path = r"C:\Programmieren\anomalie_erkennung_projekt\data\train\train_FD001.txt"
test_path = r"C:\Programmieren\anomalie_erkennung_projekt\data\test\test_FD001.txt"

train_df = read_data(train_path)
test_df = read_data(test_path)

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
    candidate_df = run_pipeline(
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
    .max()  # max RUL beim ersten Alarm = wie früh wird erkannt
)

print(first_anomaly.describe())

y_true = (results_df["RUL"] <= 30).astype(int)
y_pred = (results_df["anomaly_flag"]).astype(int)

print(f"Precision: {precision_score(y_true, y_pred)}, Recall: {recall_score(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")

fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=False)

for i, unit_id in enumerate(range(1, 6)):
    unit = results_df[results_df["unit_id"] == unit_id].sort_values("cycles")
    axes[i][0].plot(unit["cycles"], unit["RUL"])
    axes[i][0].set_ylabel(f"Unit {unit_id} RUL")
    axes[i][1].plot(unit["cycles"], unit["reconstruction_error"])
    axes[i][1].set_ylabel("Reconstruction Error")

plt.tight_layout()
plt.savefig("lstm_rul_vs_reconstruction.png", dpi=150, bbox_inches="tight")
plt.close()

first_anomaly.hist(bins=20)
plt.xlabel("RUL beim ersten Alarm")
plt.savefig("lstm_first_anomaly_histogram.png", dpi=150, bbox_inches="tight")
plt.close()