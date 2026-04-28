import sys
from pathlib import Path

PROJECT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from data_utils import read_data, sensor_cols
from iso_forest_model import fit_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

PLOTS_DIR = Path(__file__).resolve().parents[2] / "03_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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

n_estimators = 100
contamination = 0.03
max_samples = 128
window = 10
z_threshold = -2.5

train_df, model, score_mean, score_std = fit_pipeline(
    train_df, X_scaled, n_estimators, contamination, max_samples, window, z_threshold
)

y_true = (train_df["RUL"] <= 30).astype(int)
y_pred = train_df["anomaly_flag"].astype(int)

first_anomaly = (
    train_df[train_df["anomaly_flag"] == 1]
    .groupby("unit_id")["RUL"]
    .max()
)

print(first_anomaly.describe())
print(f"Precision: {precision_score(y_true, y_pred)}, Recall: {recall_score(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")

fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=False)
for i, unit_id in enumerate(range(1, 6)):
    unit = train_df[train_df["unit_id"] == unit_id].sort_values("cycles")
    axes[0][i].plot(unit["cycles"], unit["RUL"])
    axes[0][i].set_ylabel(f"Unit {unit_id} RUL")
    axes[1][i].plot(unit["cycles"], unit["anomaly_score"])
    axes[1][i].set_ylabel("Anomaly Score")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "iso_forest_rul_vs_score.png", dpi=150, bbox_inches="tight")
plt.close()

first_anomaly.hist(bins=20)
plt.xlabel("RUL beim ersten Alarm")
plt.savefig(PLOTS_DIR / "iso_forest_first_anomaly_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
