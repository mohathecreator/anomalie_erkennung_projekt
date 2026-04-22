import sys
from pathlib import Path

PROJECT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from data_utils import read_data, sensor_cols
from iso_forest_model import fit_pipeline, predict_pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_path = r"02_data\train\train_FD001.txt"
test_path = r"02_data\test\test_FD001.txt"
rul_path = r"02_data\RUL\RUL_FD001.txt"

train_df = read_data(train_path)
test_df = read_data(test_path)
rul_df = pd.read_csv(rul_path, header=None, names=["RUL"])
rul_df["unit_id"] = rul_df.index + 1

threshold_std = 0.01
useful_sensors = [col for col in sensor_cols if train_df[col].std() > threshold_std]

max_cycles = train_df.groupby("unit_id")["cycles"].max().reset_index()
max_cycles.columns = ["unit_id", "max_cycle"]
train_df = train_df.merge(max_cycles, on="unit_id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycles"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df[useful_sensors])
X_test_scaled = scaler.transform(test_df[useful_sensors])

n_estimators = 100
contamination = 0.03
max_samples = 128
window = 10
z_threshold = -2.5

_, model, score_mean, score_std = fit_pipeline(
    train_df, X_scaled, n_estimators, contamination, max_samples, window, z_threshold
)

test_df = predict_pipeline(test_df, X_test_scaled, model, score_mean, score_std, window, z_threshold)
test_df = test_df.merge(rul_df, on="unit_id")

units_with_alarm = test_df[test_df["anomaly_flag"] == 1]["unit_id"].nunique()
print(f"Units mit Alarm: {units_with_alarm} / 100")

alarmed_units = test_df[test_df["anomaly_flag"] == 1]["unit_id"].unique()
print("RUL der Units MIT Alarm:")
print(rul_df[rul_df["unit_id"].isin(alarmed_units)]["RUL"].describe())
print("RUL der Units OHNE Alarm:")
print(rul_df[~rul_df["unit_id"].isin(alarmed_units)]["RUL"].describe())
