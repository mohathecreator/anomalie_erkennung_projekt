import sys
sys.path.append(r"C:\Programmieren\anomalie_erkennung_projekt\src")
from data_utils import read_data, sensor_cols
from autoencoder_model import fit_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

train_path = r"C:\Programmieren\anomalie_erkennung_projekt\data\train\train_FD001.txt"

train_df = read_data(train_path)

threshold_std = 0.01
useful_sensors = [col for col in sensor_cols if train_df[col].std() > threshold_std]

max_cycles = train_df.groupby("unit_id")["cycles"].max().reset_index()
max_cycles.columns = ["unit_id", "max_cycle"]
train_df = train_df.merge(max_cycles, on="unit_id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycles"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[useful_sensors])
train_df[useful_sensors] = X_scaled

healthy_filter = train_df["cycles"] <= train_df["max_cycle"] * 0.3
healthy_df = train_df[healthy_filter]

window_size = 20
hidden_size = 32
lr = 0.001
epochs = 50
threshold = 0.3

results_df, model = fit_pipeline(train_df, healthy_df, useful_sensors, window_size, hidden_size, lr, epochs, threshold)

y_true = (results_df["RUL"] <= 30).astype(int)
y_pred = results_df["anomaly_flag"].astype(int)

first_anomaly = (
    results_df[results_df["anomaly_flag"] == 1]
    .groupby("unit_id")["RUL"]
    .max()
)

print(first_anomaly.describe())
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